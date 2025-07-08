import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
import wandb
from typing import Tuple, Dict, List

from nndepth.scene import Frame, Disparity
from nndepth.utils.base_trainer import BaseTrainer
from nndepth.models.raft_stereo import RAFTLoss
from nndepth.utils.distributed_training import is_main_process, is_dist_initialized
from nndepth.utils.common import get_model_state_dict


class RAFTTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.0001,
        weight_decay: float = 0.0001,
        epsilon: float = 1e-8,
        dtype: str = "bfloat16",
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        """
        Trainer for RAFT Stereo Model

        Args:
            lr (float): learning rate
            max_steps (int): number of steps to train
            num_epochs (int): number of epochs to train
            weight_decay (float): weight decay
            epsilon (float): epsilon for Adam optimizer
            dtype (str): data type for training
            device (torch.device): device to train on
        """
        super().__init__(**kwargs)
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.device = device
        if dtype == "bfloat16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                logger.warning("Bfloat16 is not supported on this machine. Using float32 instead.")
                self.dtype = torch.float32
        else:
            self.dtype = dtype

        # Init loss, optimizer, scheduler
        self.criterion = RAFTLoss(gamma=0.8, max_flow=1000)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=epsilon,
        )
        # Gradient scaler
        self.scaler = torch.amp.GradScaler(enabled=(self.dtype == torch.bfloat16))
        if self.max_steps is not None:
            self.scheduler = lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=lr,
                total_steps=self.max_steps + 100,
                anneal_strategy="linear",
                pct_start=0.05,
                cycle_momentum=False,
            )

        if self.resume:
            self.resume_from_checkpoint(
                {"model": self.model},
                {"optimizer": self.optimizer, "scheduler": self.scheduler},
                use_safetensors={"model": False, "optimizer": False, "scheduler": False},
                load_args={"model": {"strict": True}},
                device=self.device,
            )

    @torch.no_grad()
    def predict_and_get_visualization(
        self, model: nn.Module, sample: Dict[str, List[Frame]]
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
            m_outputs = model(left_tensor.to(self.device), right_tensor.to(self.device))
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

    def process_input(self, batch: List[Frame]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assert the input frames

        Args:
            batch (List[Frame]): batch of frames

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: processed frames and labels
        """
        frames = [f.data for f in batch]
        if batch[0].disparity is not None:
            labels = [f.disparity.data for f in batch]
        else:
            labels = None
        frames = torch.stack(frames, dim=0).to(self.device)
        if labels is not None:
            labels = torch.stack(labels, dim=0).to(self.device)
        return frames, labels

    def save_topk_checkpoint(self, results: dict):
        """
        Save topk checkpoint

        Args:
            results (dict): results of the evaluation
        """
        model_state_dict = get_model_state_dict(self.model)
        optimizer_state_dict = self.optimizer.state_dict()
        scheduler_state_dict = self.scheduler.state_dict()

        # Save checkpoint if best
        topk_cp_dir, old_topk_cp = self.is_topk_checkpoint(
            results["loss"], "loss", condition="min"
        )
        if topk_cp_dir:
            logger.info(
                f"loss: {results['loss']} in top {self.save_best_k_cp} checkpoints."
                + f" Saving at {os.path.join(self.artifact_dir, topk_cp_dir)}."
            )

            self.save_checkpoint(
                os.path.join(self.artifact_dir, topk_cp_dir),
                {"model": model_state_dict, "optimizer": optimizer_state_dict, "scheduler": scheduler_state_dict},
                use_safetensors={"model": False, "optimizer": False, "scheduler": False},
            )

            if old_topk_cp:
                logger.info(f"Removing old checkpoint {os.path.join(self.artifact_dir, old_topk_cp)}.")
                shutil.rmtree(os.path.join(self.artifact_dir, old_topk_cp))

        # Remove old checkpoint & Save latest checkpoint
        old_latest_cp_dir = self.get_latest_checkpoint_from_dir(self.artifact_dir)
        if old_latest_cp_dir:
            logger.info(f"Removing old latest checkpoint {os.path.join(self.artifact_dir, old_latest_cp_dir)}...")
            shutil.rmtree(os.path.join(self.artifact_dir, old_latest_cp_dir))
        latest_cp_dir = self.get_latest_checkpoint_name(self.current_step)

        self.save_checkpoint(
            os.path.join(self.artifact_dir, latest_cp_dir),
            {"model": model_state_dict, "optimizer": optimizer_state_dict, "scheduler": scheduler_state_dict},
            use_safetensors={"model": False, "optimizer": False, "scheduler": False},
        )
        logger.info(f"Saved latest checkpoint {latest_cp_dir}.")

    def train(self,
              model: nn.Module,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              ) -> None:
        """
        Train the model

        Args:
            model (nn.Module): model to train
            train_dataloader (DataLoader): training data loader
            val_dataloader (DataLoader): validation data loader
        """
        logger.info("Start training")

        if self.max_steps is not None:
            self.num_epochs = self.max_steps // (len(train_dataloader) // self.gradient_accumulation_steps) + 1
        elif self.num_epochs is not None:
            self.max_steps = len(train_dataloader) * self.num_epochs // self.gradient_accumulation_steps

        if is_main_process():
            pbar = tqdm(total=len(train_dataloader))
            desc = "Epoch {current_epoch}/{num_epochs} - Step {current_step}/{max_steps} - Loss: {loss:.4f}"

        training_metrics = {"loss": []}
        loss = torch.Tensor([0])
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            pbar.reset()
            for i_batch, batch in enumerate(train_dataloader):
                if is_main_process():
                    pbar.update(1)
                    pbar.set_description(
                        desc.format(
                            current_epoch=epoch,
                            num_epochs=self.num_epochs,
                            current_step=self.current_step,
                            max_steps=self.max_steps,
                            loss=loss.item(),
                        )
                    )

                left_frames, left_labels = self.process_input(batch["left"])
                right_frames, _ = self.process_input(batch["right"])

                with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                    outputs = model(left_frames, right_frames)
                    loss, metrics = self.criterion(left_labels, outputs)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (i_batch + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.current_step += 1

                # Log metrics
                training_metrics["loss"].append(loss.item() * self.gradient_accumulation_steps)
                for key, value in metrics.items():
                    if key not in training_metrics:
                        training_metrics[key] = []
                    training_metrics[key].append(value)
                if (self.current_step + 1) % self.log_interval == 0:
                    log_data = {}
                    for key, value in training_metrics.items():
                        log_data[key] = np.mean(value)
                    log_data["train/lr"] = self.scheduler.get_last_lr()[0]
                    self.log_training(log_data)
                    training_metrics = {"loss": []}

                # Validation
                if self.reach_eval_interval(len(train_dataloader)):
                    logger.info("Validation started")
                    if is_dist_initialized():
                        dist.barrier()
                    model.eval()
                    results = self.evaluate(model, val_dataloader)

                    # Infer on 1 example to log for debugging purposes
                    image, disp_gt, disp_pred = self.predict_and_get_visualization(
                        model,
                        next(iter(val_dataloader)),
                    )

                    log_data = {}
                    for key, value in results.items():
                        log_data[f"val/{key}"] = np.mean(value)
                    log_data["val/left_image"] = wandb.Image(image)
                    log_data["val/GT"] = wandb.Image(disp_gt)
                    log_data["val/Prediction"] = wandb.Image(disp_pred)
                    self.log_training(log_data)

                    if is_main_process():
                        self.save_topk_checkpoint(results)

                    # Back to training mode
                    model.train()
                    if is_dist_initialized():
                        dist.barrier()

                if self.current_step >= self.max_steps:
                    logger.info("Training finished. Saving the last checkpoint")
                    self.save_checkpoint(
                        os.path.join(self.artifact_dir, self.get_latest_checkpoint_name(self.current_step)),
                        {
                            "model": get_model_state_dict(self.model),
                            "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                        },
                        use_safetensors={"model": False, "optimizer": False, "scheduler": False},
                    )
                    if is_main_process():
                        pbar.close()

                    logger.info("Training finished")
                    return

    @torch.no_grad()
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> dict:
        """
        Evaluate RAFT-based model

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
            if self.num_val_samples is not None and i_batch >= self.num_val_samples:
                break
            left_frames, left_labels = self.process_input(batch["left"])
            right_frames, _ = self.process_input(batch["right"])

            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                # Forward
                m_outputs = model(left_frames, right_frames)
                loss, metrics = self.criterion(left_labels, m_outputs)

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
