import os
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from loguru import logger
from tqdm import tqdm
import wandb
from typing import Optional, Tuple, Callable, Dict, List

from nndepth.scene import Frame, Disparity
from nndepth.utils.base_trainer import BaseTrainer
from nndepth.utils.trackers.wandb import WandbTracker
from nndepth.utils import is_distributed_training, is_main_process


class IGEVStereoTrainer(BaseTrainer):
    def __init__(
        self,
        lr: float = 0.0001,
        num_epochs: Optional[int] = 100,
        max_steps: Optional[int] = 100000,
        weight_decay: float = 0.0001,
        epsilon: float = 1e-8,
        dtype: str = "bfloat16",
        gradient_accumulation_steps: Optional[int] = 1,
        **kwargs,
    ):
        """
        Trainer for IGEV Stereo Model

        Args:
            lr (float): learning rate
            max_steps (int): number of steps to train
            num_epochs (int): number of epochs to train
            weight_decay (float): weight decay
            epsilon (float): epsilon for Adam optimizer
            gradient_accumulation_steps (int): number of steps to accumulate gradients
            dtype (str): data type for training
        """
        assert num_epochs is not None or max_steps is not None, "Either num_epochs or max_steps must be provided"
        assert (
            gradient_accumulation_steps is None or gradient_accumulation_steps > 0
        ), "gradient_accumulation_steps must be greater than 0"

        super().__init__(**kwargs)
        self.lr = lr
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if dtype == "bfloat16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                logger.warning("Bfloat16 is not supported on this machine. Using float32 instead.")
                self.dtype = torch.float32
        else:
            self.dtype = dtype

    @torch.no_grad()
    def predict_and_get_visualization(
        self, model: nn.Module, sample: Dict[str, List[Frame]], device: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get visualization of 1 example for debugging purposes

        Args:
            model (nn.Module): model
            val_dataloader (DataLoader): validation data loader

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: left image, ground truth disparity, predicted disparity
        """
        left_frame, right_frame = sample["left"][0], sample["right"][0]

        left_tensor, right_tensor = left_frame.data[None], right_frame.data[None]
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            m_outputs = model(left_tensor.to(device), right_tensor.to(device))
            disp_pred = m_outputs[-1]["up_disp"][0]
            disp_pred = Disparity(data=disp_pred, disp_sign="negative")

        left_image = left_frame.data.cpu().numpy().squeeze().transpose(1, 2, 0)
        left_image = (left_image * 127.5 + 127.5).astype(np.uint8)
        disp_gt = left_frame.disparity
        disp_gt = disp_gt.resize(disp_pred.data.shape[-2:], method="maxpool")
        min_disp, max_disp = 0, disp_gt.data.abs().max().item()

        disp_gt_image = disp_gt.get_view(min=min_disp, max=max_disp, cmap="magma")
        disp_pred_image = disp_pred.get_view(min=min_disp, max=max_disp, cmap="magma")

        return left_image, disp_gt_image, disp_pred_image

    def process_input(self, batch: List[Frame], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assert the input frames

        Args:
            left_frame (Frame): left frame
            right_frame (Frame): right frame
        """
        frames = [f.data for f in batch]
        if batch[0].disparity is not None:
            labels = [f.disparity.data for f in batch]
        else:
            labels = None
        frames = torch.stack(frames, dim=0).to(device)
        if labels is not None:
            labels = torch.stack(labels, dim=0).to(device)
        return frames, labels

    def train(self,
              model: nn.Module,
              criterion: Callable,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              optimizer: optim.Optimizer,
              scheduler: optim.lr_scheduler._LRScheduler,
              scaler: torch.cuda.amp.GradScaler,
              tracker: WandbTracker,
              device: str,
              ) -> None:
        """
        Train the model

        Args:
            model (nn.Module): model to train
            criterion (Callable): criterion function
            train_dataloader (DataLoader): training data loader
            val_dataloader (DataLoader): validation data loader
            optimizer (optim.Optimizer): optimizer
            scheduler (optim.lr_scheduler._LRScheduler): scheduler
            tracker (WandbTracker): tracker
        """
        logger.info("Start training")

        if self.max_steps is not None:
            self.num_epochs = self.max_steps // (len(train_dataloader) // self.gradient_accumulation_steps) + 1
        elif self.num_epochs is not None:
            self.max_steps = len(train_dataloader) * self.num_epochs

        # Placeholders for metrics
        losses = []
        l_epe = []
        l_percent_0_5 = []
        l_percent_1 = []
        l_percent_3 = []
        l_percent_5 = []

        current_epoch = self.current_steps * self.gradient_accumulation_steps // len(train_dataloader)
        if isinstance(self.val_interval, int):
            val_interval_steps = self.val_interval
        elif isinstance(self.val_interval, float) and self.val_interval <= 1:
            val_interval_steps = int(self.val_interval * len(train_dataloader))
        else:
            raise ValueError("val_interval must be either int or float <= 1")

        loss = torch.Tensor([0])
        for epoch in range(current_epoch, self.num_epochs):
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{self.num_epochs}")
            for i_batch, batch in enumerate(pbar):
                pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
                left_frames, left_labels = self.process_input(batch["left"], device)
                right_frames, _ = self.process_input(batch["right"], device)

                with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                    outputs = model(left_frames, right_frames)
                    loss, metrics = criterion(left_labels, outputs)
                    loss = loss / self.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (i_batch + 1) % self.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.current_steps += 1

                # Log metrics
                losses.append(loss.item() * self.gradient_accumulation_steps)
                l_epe.append(metrics["epe"])
                l_percent_0_5.append(metrics["0.5px"])
                l_percent_1.append(metrics["1px"])
                l_percent_3.append(metrics["3px"])
                l_percent_5.append(metrics["5px"])

                if (i_batch + 1) % self.log_interval == 0:
                    tracker.log(
                        {
                            "train/step": self.current_steps,
                            "train/epoch": epoch,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/loss": np.mean(losses),
                            "train/epe": np.mean(l_epe),
                            "train/percent_0_5px": np.mean(l_percent_0_5),
                            "train/percent_1px": np.mean(l_percent_1),
                            "train/percent_3px": np.mean(l_percent_3),
                            "train/percent_5px": np.mean(l_percent_5),
                        }
                    )
                    losses = []
                    l_epe = []
                    l_percent_0_5 = []
                    l_percent_1 = []
                    l_percent_3 = []
                    l_percent_5 = []

                # Validation
                if (i_batch + 1) % val_interval_steps == 0:
                    model.eval()
                    results = self.evaluate(model, criterion, val_dataloader, device)

                    # Infer on 1 example to log for debugging purposes
                    image, disp_gt, disp_pred = self.predict_and_get_visualization(model, next(iter(val_dataloader)), device)

                    tracker.log(
                        {
                            "val/loss": results["loss"],
                            "val/epe": results["epe"],
                            "val/percent_0_5px": results["percent_0_5px"],
                            "val/percent_1px": results["percent_1px"],
                            "val/percent_3px": results["percent_3px"],
                            "val/percent_5px": results["percent_5px"],
                            "val/left_image": wandb.Image(image),
                            "val/GT": wandb.Image(disp_gt),
                            "val/Prediction": wandb.Image(disp_pred),
                        }
                    )

                    if is_distributed_training():
                        model_state_dict = getattr(model.module, '_orig_mod', model.module).state_dict()
                    else:
                        model_state_dict = getattr(model, '_orig_mod', model).state_dict()

                    if (is_distributed_training() and is_main_process()) or not is_distributed_training():
                        # Save checkpoint if best
                        topk_cp_dir, old_topk_cp = self.is_topk_checkpoint(
                            epoch, self.current_steps, results["loss"], "loss", condition="min"
                        )
                        if topk_cp_dir:
                            logger.info(f"loss: {results['loss']} in top {self.save_best_k_cp} checkpoints. Saving...")

                            self.save_checkpoint(
                                os.path.join(self.artifact_dir, topk_cp_dir),
                                model_state_dict,
                                optimizer.state_dict(),
                                scheduler.state_dict(),
                            )

                            if old_topk_cp:
                                shutil.rmtree(os.path.join(self.artifact_dir, old_topk_cp))

                        # Remove old checkpoint & Save latest checkpoint
                        old_latest_cp_dir = self.get_latest_checkpoint_from_dir(self.artifact_dir)
                        if old_latest_cp_dir:
                            shutil.rmtree(os.path.join(self.artifact_dir, old_latest_cp_dir))
                        latest_cp_dir = self.get_latest_checkpoint_name(self.current_steps)

                        self.save_checkpoint(
                            os.path.join(self.artifact_dir, latest_cp_dir),
                            model_state_dict,
                            optimizer.state_dict(),
                            scheduler.state_dict(),
                        )

                    # Back to training mode
                    model.train()

                if self.current_steps >= self.max_steps:
                    logger.info("Training finished. Save the last checkpoint")
                    self.save_checkpoint(
                        os.path.join(self.artifact_dir, self.get_latest_checkpoint_name(self.current_steps)),
                        model.module.state_dict() if is_distributed_training() else model.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                    )
                    break

    @torch.no_grad()
    def evaluate(self, model: nn.Module, criterion: Callable, dataloader: DataLoader, device: str) -> dict:
        """
        Evaluate IGEV Stereo model

        Args:
            model (nn.Module): model to evaluate
            criterion (Callable): criterion function
            dataloader (DataLoader): data loader

        Returns:
            dict: evaluation metrics: loss, epe, percent_0_5, percent_1, percent_3, percent_5
        """
        losses = []
        epe = []
        percent_0_5 = []
        percent_1 = []
        percent_3 = []
        percent_5 = []

        for i_batch, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if self.num_val_samples != -1 and i_batch >= self.num_val_samples:
                break
            left_frames, left_labels = self.process_input(batch["left"], device)
            right_frames, _ = self.process_input(batch["right"], device)

            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                # Forward
                m_outputs = model(left_frames, right_frames)
                loss, metrics = criterion(left_labels, m_outputs)

            # Metrics
            losses.append(loss.detach().item())
            epe.append(metrics["epe"])
            percent_0_5.append(metrics["0.5px"])
            percent_1.append(metrics["1px"])
            percent_3.append(metrics["3px"])
            percent_5.append(metrics["5px"])

        return {
            "loss": np.mean(losses),
            "epe": np.mean(epe),
            "percent_0_5px": np.mean(percent_0_5),
            "percent_1px": np.mean(percent_1),
            "percent_3px": np.mean(percent_3),
            "percent_5px": np.mean(percent_5),
        }
