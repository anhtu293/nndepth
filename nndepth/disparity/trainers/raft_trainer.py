import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from accelerate import Accelerator
from loguru import logger
from tqdm import tqdm
from typing import Optional, Tuple

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
        gradient_accumulation_steps: Optional[int] = None,
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

    def prepare(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
    ) -> Tuple[nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler, DataLoader, DataLoader]:
        """
        Prepare the model, criterion, optimizer, scheduler, train and validation dataloaders using the accelerator
        """
        # Prepare
        model, criterion, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            model, criterion, optimizer, train_dataloader, val_dataloader, scheduler
        )
        logger.info("Finish accelerator preparation")
        return model, criterion, optimizer, scheduler, train_dataloader, val_dataloader

    def train(
        self,
        model: nn.Module,
        criterion: nn.Module,
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
        logger.info("Start training")

        if self.num_epochs is not None:
            self.max_steps = len(train_dataloader) * self.num_epochs
        elif self.max_steps is not None:
            self.num_epochs = self.max_steps // len(train_dataloader) + 1

        # Train loop
        losses = []
        l_epe = []
        l_percent_0_5 = []
        l_percent_1 = []
        l_percent_3 = []
        l_percent_5 = []

        current_epoch = self.total_steps // len(train_dataloader)
        if isinstance(self.val_interval, int):
            val_interval_steps = self.val_interval
        elif isinstance(self.val_interval, float) and self.val_interval <= 1:
            val_interval_steps = int(self.val_interval * len(train_dataloader))
        else:
            raise ValueError("val_interval must be either int or float <= 1")

        with self.accelerator.accumulate(model):
            for epoch in range(current_epoch, self.num_epochs):
                # Skip first batches if resuming
                train_dataloader = self.accelerator.skip_first_batches(
                    train_dataloader, self.total_steps % len(train_dataloader)
                )

                for i_batch, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{self.num_epochs}")):

                    left_frame, right_frame = batch["left"], batch["right"]
                    left_frame, right_frame = left_frame.as_tensor(), right_frame.as_tensor()

                    # Forward
                    m_outputs = model(left_frame, right_frame)
                    disp_gt = left_frame.disparity.as_tensor()
                    loss = criterion(disp_gt, m_outputs)

                    # Backward
                    self.accelerator.backward(loss)
                    if i_batch > 0 and i_batch % self.gradient_accumulation_steps == 0:
                        self.total_steps += 1
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # Metrics
                    losses.append(loss.item())
                    epe, percent_0_5, percent_1, percent_3, percent_5 = self.compute_metrics(disp_gt, m_outputs)
                    l_epe.append(epe)
                    l_percent_0_5.append(percent_0_5)
                    l_percent_1.append(percent_1)
                    l_percent_3.append(percent_3)
                    l_percent_5.append(percent_5)

                    # Log
                    if self.total_steps > 0 and self.total_steps % self.log_interval == 0:
                        tracker.log(
                            {
                                "train/step": self.total_steps,
                                "train/epoch": epoch,
                                "train/loss": np.mean(losses),
                                "train/epe": np.mean(l_epe),
                                "train/percent_0_5": np.mean(l_percent_0_5),
                                "train/percent_1": np.mean(l_percent_1),
                                "train/percent_3": np.mean(l_percent_3),
                                "train/percent_5": np.mean(l_percent_5),
                            }
                        )
                        losses = []
                        l_epe = []
                        l_percent_0_5 = []
                        l_percent_1 = []
                        l_percent_3 = []
                        l_percent_5 = []

                    # Validation
                    if self.total_steps > 0 and self.total_steps % val_interval_steps == 0:
                        model.eval()
                        results = self.validate(model, criterion, val_dataloader)
                        tracker.log(
                            {
                                "val/loss": results["loss"],
                                "val/epe": results["epe"],
                                "val/percent_0_5": results["percent_0_5"],
                                "val/percent_1": results["percent_1"],
                                "val/percent_3": results["percent_3"],
                                "val/percent_5": results["percent_5"],
                            }
                        )

                        # Save checkpoint if best
                        if results["loss"] > self.checkpoint_infos[-1]["metric"]:
                            _ = self.checkpoint_infos.pop(-1)
                            self.checkpoint_infos.append(
                                {
                                    "epoch": epoch,
                                    "steps": self.total_steps,
                                    "metric": results["loss"],
                                    "metric_name": "loss",
                                }
                            )
                            self.checkpoint_infos = sorted(self.checkpoint_infos, key=lambda x: x["metric"])
                            cp_dir = self.get_checkpoint_name(epoch, self.total_steps, results["loss"], "loss")
                            self.accelerator.save_state(cp_dir)

                        model.train(True)
