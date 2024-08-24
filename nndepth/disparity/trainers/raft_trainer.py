import os
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from accelerate import Accelerator
from loguru import logger
from tqdm import tqdm
import wandb
from typing import Optional, Tuple, Callable, Dict

import aloscene

from nndepth.utils.base_trainer import BaseTrainer
from nndepth.utils.trackers.wandb import WandbTracker


class RAFTTrainer(BaseTrainer):
    def __init__(
        self,
        lr: float = 0.0001,
        num_epochs: Optional[int] = 100,
        max_steps: Optional[int] = 100000,
        weight_decay: float = 0.0001,
        epsilon: float = 1e-8,
        gradient_accumulation_steps: Optional[int] = 1,
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
            gradient_accumulation_steps (int): number of steps to accumulate gradients
            val_interval (Union[float, int]): interval to validate
            log_interval (int): interval to log
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

        self.accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
        self.is_prepared = False

    def predict_and_get_visualization(
        self, model: nn.Module, sample: Dict[str, aloscene.Frame]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get visualization of 1 example for debugging purposes

        Args:
            model (nn.Module): model
            val_dataloader (DataLoader): validation data loader

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: left image, ground truth disparity, predicted disparity
        """

        def get_disp_image(
            disp: aloscene.Disparity, min_disp: Optional[float] = None, max_disp: Optional[float] = None
        ) -> np.ndarray:
            disp_image = disp.__get_view__(min_disp, max_disp).image
            disp_image = (disp_image * 255).astype(np.uint8)
            return disp_image

        left_frame, right_frame = sample["left"], sample["right"]
        left_tensor, right_tensor = left_frame.as_tensor(), right_frame.as_tensor()
        m_outputs = model(left_tensor, right_tensor)
        disp_pred = m_outputs[-1]["up_disp"][0]
        disp_pred = aloscene.Disparity(disp_pred, camera_side="left", disp_format="signed", names=("C", "H", "W"))

        left_image = left_frame.norm255().as_tensor().cpu().numpy().squeeze().transpose(1, 2, 0).astype(np.uint8)
        disp_gt = left_frame.disparity[0]
        disp_gt.names = ("C", "H", "W")  # Set names to avoid error in resize
        disp_gt = disp_gt.resize(disp_pred.shape[-2:], mode="nearest")
        min_disp, max_disp = disp_gt.min(), disp_gt.max()

        disp_gt_image = get_disp_image(disp_gt, min_disp, max_disp)
        disp_pred_image = get_disp_image(disp_pred, min_disp, max_disp)

        return left_image, disp_gt_image, disp_pred_image

    def assert_input(self, left_frame: aloscene.Frame, right_frame: aloscene.Frame):
        """
        Assert the input frames

        Args:
            left_frame (aloscene.Frame): left frame
            right_frame (aloscene.Frame): right frame
        """
        assert left_frame.disparity is not None, "Left frame must have disparity"
        assert (
            left_frame.normalization == "minmax_sym" and right_frame.normalization == "minmax_sym"
        ), f"frames must be minmax_sym normalized. Found {left_frame.normalization} and {right_frame.normalization}"
        assert left_frame.names == ("B", "C", "H", "W") and right_frame.names == (
            "B",
            "C",
            "H",
            "W",
        ), "frames must have names ('B', 'C', 'H', 'W'). Found {left_frame.names} and {right_frame.names}"

    def prepare(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
    ) -> Tuple[nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler, DataLoader, DataLoader]:
        """
        Prepare the model, criterion, optimizer, scheduler, train and validation dataloaders using the accelerator
        """
        # Prepare
        model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, scheduler
        )
        self.is_prepared = True
        logger.info("Finish accelerator preparation")
        return model, train_dataloader, val_dataloader, optimizer, scheduler

    def resume_from_checkpoint(self, dir_path: str):
        """
        Resume training from the checkpoint

        Args:
            dir_path (str): path to the directory
        """
        self.load_state(dir_path)

        # Load model, criterion, optimizer, scheduler, train and val dataloaders
        assert self.is_prepared, "Prepare the trainer before resuming. See `prepare` method"
        self.accelerator.load_state(dir_path)
        logger.info("Accelerator state is resumed from checkpoint !")

    def train(
        self,
        model: nn.Module,
        criterion: Callable,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        tracker: WandbTracker,
    ):
        """
        Train the model

        Args:
            model (nn.Module): model to train
            train_dataloader (DataLoader): training data loader
            val_dataloader (DataLoader): validation data loader
            optimizer (optim.Optimizer): optimizer
        """
        assert self.is_prepared, "Prepare the trainer before training. See `prepare` method"

        logger.info("Start training")

        if self.max_steps is not None:
            self.num_epochs = self.max_steps // len(train_dataloader) + 1
        elif self.num_epochs is not None:
            self.max_steps = len(train_dataloader) * self.num_epochs

        # Placeholders for metrics
        losses = []
        l_epe = []
        l_percent_0_5 = []
        l_percent_1 = []
        l_percent_3 = []
        l_percent_5 = []

        current_epoch = self.current_steps // len(train_dataloader)
        if isinstance(self.val_interval, int):
            val_interval_steps = self.val_interval
        elif isinstance(self.val_interval, float) and self.val_interval <= 1:
            val_interval_steps = int(self.val_interval * len(train_dataloader))
        else:
            raise ValueError("val_interval must be either int or float <= 1")

        finish_training = False

        # Start training
        with self.accelerator.accumulate(model):
            for epoch in range(current_epoch, self.num_epochs):
                # Skip first batches if resuming
                train_dataloader = self.accelerator.skip_first_batches(
                    train_dataloader, self.current_steps % len(train_dataloader)
                )

                for i_batch, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{self.num_epochs}")):
                    left_frame, right_frame = batch["left"], batch["right"]
                    self.assert_input(left_frame, right_frame)

                    left_tensor, right_tensor = left_frame.as_tensor(), right_frame.as_tensor()

                    # Forward
                    m_outputs = model(left_tensor, right_tensor)
                    disp_gt = left_frame.disparity.as_tensor()
                    loss, metrics = criterion(disp_gt, m_outputs)

                    # Backward
                    self.accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # Metrics
                    losses.append(loss.detach().item())
                    l_epe.append(metrics["epe"])
                    l_percent_0_5.append(metrics["0.5px"])
                    l_percent_1.append(metrics["1px"])
                    l_percent_3.append(metrics["3px"])
                    l_percent_5.append(metrics["5px"])

                    # Accumulate steps & Log & Evaluate if needed
                    if i_batch > 0 and i_batch % self.gradient_accumulation_steps == 0:
                        self.current_steps += 1

                        # Log
                        if self.current_steps > 0 and self.current_steps % self.log_interval == 0:
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
                        if self.current_steps > 0 and self.current_steps % val_interval_steps == 0:
                            model.eval()
                            results = self.evaluate(model, criterion, val_dataloader)

                            # Infer on 1 example to log for debugging purposes
                            image, disp_gt, disp_pred = self.predict_and_get_visualization(
                                model, next(iter(val_dataloader))
                            )

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

                            # Save checkpoint if best
                            topk_cp_dir, old_topk_cp = self.is_topk_checkpoint(
                                epoch, self.current_steps, results["loss"], "loss", condition="min"
                            )
                            if topk_cp_dir:
                                logger.info(
                                    f"loss: {results['loss']} in top {self.save_best_k_cp} checkpoints. Saving..."
                                )
                                self.accelerator.save_state(
                                    os.path.join(self.artifact_dir, topk_cp_dir), safe_serialization=False
                                )
                                # Remove old checkpoint
                                if old_topk_cp:
                                    shutil.rmtree(os.path.join(self.artifact_dir, old_topk_cp))
                            else:
                                logger.info(f"loss: {results['loss']} not in top {self.save_best_k_cp} checkpoints")

                            # Remove old checkpoint & Save latest checkpoint
                            old_latest_cp_dir = self.get_latest_checkpoint_from_dir(self.artifact_dir)
                            if old_latest_cp_dir:
                                shutil.rmtree(os.path.join(self.artifact_dir, old_latest_cp_dir))
                            latest_cp_dir = self.get_latest_checkpoint_name(self.current_steps)
                            self.accelerator.save_state(
                                os.path.join(self.artifact_dir, latest_cp_dir), safe_serialization=False
                            )

                            # Check if training is finished
                            if self.current_steps >= self.max_steps:
                                finish_training = True
                                break

                            # Back to training mode
                            model.train(True)

                if finish_training:
                    break

        # Save the last checkpoint
        old_latest_cp_dir = self.get_latest_checkpoint_from_dir(self.artifact_dir)
        if old_latest_cp_dir:
            shutil.rmtree(os.path.join(self.artifact_dir, old_latest_cp_dir))
        latest_cp_dir = self.get_latest_checkpoint_name(self.current_steps)
        self.accelerator.save_state(os.path.join(self.artifact_dir, latest_cp_dir), safe_serialization=False)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, criterion: Callable, dataloader: DataLoader) -> dict:
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
            if self.num_val_samples != -1 and i_batch >= self.num_val_samples:
                break
            left_frame, right_frame = batch["left"], batch["right"]
            left_frame, right_frame = left_frame.as_tensor(), right_frame.as_tensor()

            # Forward
            m_outputs = model(left_frame, right_frame)
            disp_gt = left_frame.disparity.as_tensor()
            loss, metrics = criterion(disp_gt, m_outputs)

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
