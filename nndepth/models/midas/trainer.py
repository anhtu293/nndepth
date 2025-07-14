import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
from typing import Tuple
from loguru import logger

from nndepth.scene import Frame, Depth
from nndepth.utils import BaseTrainer, is_main_process, is_dist_initialized, get_model_state_dict
from nndepth.models.midas.loss import MiDasLoss, scale_shift_estimation, ssi_depth, normalize_01_depth


class MiDasTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        lr_decay_every_epochs: int = 10,
        viz_log_interval: int = 500,
        trimmed_loss_coef: float = 1.0,
        gradient_reg_coef: float = 0.1,
        dtype: str = "bfloat16",
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        super(MiDasTrainer, self).__init__(**kwargs)
        self.model = model
        self.lr = lr
        self.lr_decay_every_epochs = lr_decay_every_epochs
        self.viz_log_interval = viz_log_interval
        self.trimmed_loss_coef = trimmed_loss_coef
        self.gradient_reg_coef = gradient_reg_coef
        self.device = device

        if dtype == "bfloat16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                logger.warning("Bfloat16 is not supported on this machine. Using float32 instead.")
                self.dtype = torch.float32
        else:
            self.dtype = dtype

        self.optimizer = self.build_optimizer()
        self.loss_fn = self.build_loss_fn()
        self.scheduler = self.build_scheduler()

        self.scaler = GradScaler()

    def build_optimizer(self) -> optim.Optimizer:
        params = [{
            "params": [p for n, p in self.model.named_parameters() if "encoder" in n],
            "lr": 1e-4,
        }, {
            "params": [p for n, p in self.model.named_parameters() if "encoder" not in n],
            "lr": self.lr,
        }]
        return optim.Adam(params, weight_decay=1e-5)

    def build_loss_fn(self) -> nn.Module:
        return MiDasLoss(
            trimmed_loss_coef=self.trimmed_loss_coef,
            gradient_reg_coef=self.gradient_reg_coef,
        )

    def build_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_decay_every_epochs, gamma=0.1)

    def save_topk_checkpoint(self, metrics: dict):
        """
        Save topk checkpoint

        Args:
            metrics (dict): metrics of the evaluation
        """
        topk_cp_dir, old_topk_cp = self.is_topk_checkpoint(metrics["val/loss"], "loss", condition="min")
        if topk_cp_dir:
            logger.info(
                f"loss: {metrics['val/loss']} in top {self.save_best_k_cp} checkpoints."
                + f" Saving at {os.path.join(self.artifact_dir, topk_cp_dir)}."
            )
            state_dicts = {
                "model": get_model_state_dict(self.model),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
            }
            self.save_checkpoint(
                os.path.join(self.artifact_dir, topk_cp_dir),
                state_dicts,
                use_safetensors={"model": True, "optimizer": False, "scheduler": False, "scaler": False},
            )

            if old_topk_cp:
                logger.info(f"Removing old checkpoint {os.path.join(self.artifact_dir, old_topk_cp)}.")
                shutil.rmtree(os.path.join(self.artifact_dir, old_topk_cp))

        # Remove old checkpoint & Save latest checkpoint
        old_latest_cp_dir = self.get_latest_checkpoint_from_dir(self.artifact_dir)
        if old_latest_cp_dir:
            logger.info(f"Removing old latest checkpoint {os.path.join(self.artifact_dir, old_latest_cp_dir)}...")
            shutil.rmtree(os.path.join(self.artifact_dir, old_latest_cp_dir))

        # Save latest checkpoint
        latest_cp_dir = self.get_latest_checkpoint_name(self.current_step)
        state_dicts = {
            "model": get_model_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        self.save_checkpoint(
            os.path.join(self.artifact_dir, latest_cp_dir),
            state_dicts,
            use_safetensors={"model": True, "optimizer": False, "scheduler": False, "scaler": False},
        )

    def log_viz(self, batch: Frame, outputs: torch.Tensor, stage: str = "train"):
        """
        Log visualization of the model

        Args:
            batch (Frame): batch of data
            outputs (torch.Tensor): outputs of the model
        """

        frame = (batch.data[0] * 127.5 + 127.5).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        gt = batch.depth.data[:1].cpu().to(torch.float32).detach()
        valid_mask = batch.depth.valid_mask[:1].cpu().detach()
        pred_depth = outputs[:1].cpu().to(torch.float32).detach()

        gt_depth_view = Depth(gt[0], valid_mask[0]).get_view(cmap="magma")
        pred_depth_view = Depth(pred_depth[0], valid_mask[0]).get_view(cmap="magma")

        # Error map
        gt_scale, gt_shift = scale_shift_estimation(gt, valid_mask)
        gt_ssi = ssi_depth(gt, gt_scale, gt_shift)
        gt_normalized = normalize_01_depth(gt_ssi, valid_mask)
        pred_ssi = ssi_depth(pred_depth, gt_scale, gt_shift)
        pred_normalized = normalize_01_depth(pred_ssi, valid_mask)

        abs_dev = torch.abs(pred_normalized - gt_normalized)
        min_abs_dev = abs_dev[valid_mask].min()
        max_abs_dev = abs_dev[valid_mask].max()
        abs_dev = (abs_dev - min_abs_dev) / (max_abs_dev - min_abs_dev)
        abs_dev[~valid_mask] = 0
        abs_dev = abs_dev.cpu().numpy().squeeze()
        abs_dev = (abs_dev * 255).astype(np.uint8)
        abs_dev = np.stack([abs_dev, abs_dev, abs_dev], axis=-1)

        # Log image
        self.log_visualization(
            {
                f"{stage}_image/frame": frame,
                f"{stage}_image/gt": gt_depth_view,
                f"{stage}_image/pred": pred_depth_view,
                f"{stage}_image/error_map": abs_dev,
            }
        )

    def _common_step(self, batch: Frame) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Common step for training and evaluation

        Args:
            batch (Frame): batch of data
        """
        batch_tensor = batch.data.to(self.device)
        gt_tensor = batch.depth.data.to(self.device)
        valid_mask = batch.depth.valid_mask.to(self.device)
        outputs = self.model(batch_tensor)
        loss, metrics = self.loss_fn(outputs, gt_tensor, valid_mask)
        return loss, metrics, outputs

    def train_step(self, batch: Frame, no_sync: bool = False) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Train step for the model

        Args:
            batch (Frame): batch of data
            no_sync (bool): whether to use no_sync

        Returns:
            Tuple[torch.Tensor, dict, torch.Tensor]: loss, metrics, outputs
        """
        if no_sync:
            with self.model.no_sync():
                loss, metrics, outputs = self._common_step(batch)
        else:
            loss, metrics, outputs = self._common_step(batch)
        loss /= self.gradient_accumulation_steps
        return loss, metrics, outputs

    def val_step(self, batch: Frame) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Val step for the model

        Args:
            batch (Frame): batch of data

        Returns:
            Tuple[torch.Tensor, dict, torch.Tensor]: loss, metrics, outputs
        """
        return self._common_step(batch)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *args,
        **kwargs,
    ):
        """
        Train the model

        Args:
            train_loader (DataLoader): train data loader
            val_loader (DataLoader): val data loader
        """
        logger.info("Starting training...")
        is_main_rank = is_main_process()

        if is_main_rank:
            pbar = tqdm(total=len(train_loader))
            desc = "Epoch {current_epoch}/{total_epochs} - Step {current_step}/{total_steps} - Loss {loss:.4f}"

        if self.max_steps is None:
            self.max_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        if self.num_epochs is None:
            self.num_epochs = self.max_steps // len(train_loader) * self.gradient_accumulation_steps

        training_metrics = {}
        for self.current_epoch in range(self.num_epochs):
            if is_main_rank:
                pbar.reset()

            for i_batch, batch in enumerate(train_loader):
                with autocast(device_type="cuda", dtype=self.dtype):
                    no_sync = (i_batch % self.gradient_accumulation_steps) != 0 and is_dist_initialized()
                    loss, metrics, outputs = self.train_step(batch, no_sync)

                if is_main_rank:
                    pbar.update(1)
                    pbar.set_description(
                        desc.format(
                            current_epoch=self.current_epoch,
                            total_epochs=self.num_epochs,
                            current_step=self.current_step,
                            total_steps=len(train_loader),
                            loss=loss.detach().cpu().item() * self.gradient_accumulation_steps,
                        )
                    )
                    for key, value in metrics.items():
                        key = f"train/{key}"
                        if key not in training_metrics:
                            training_metrics[key] = []
                        training_metrics[key].append(value)

                self.scaler.scale(loss).backward()
                if (i_batch % self.gradient_accumulation_steps) == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.current_step += 1
                else:
                    continue

                if (self.current_step % self.log_interval) == 0:
                    for key, value in training_metrics.items():
                        training_metrics[key] = np.mean(value)
                    self.log_metrics(training_metrics)
                    training_metrics = {}

                if (self.viz_log_interval > 0) and (self.current_step % self.viz_log_interval) == 0:
                    self.log_viz(batch, outputs, stage="train")

                if self.reach_eval_interval(len(val_loader)):
                    if is_dist_initialized():
                        dist.barrier()

                    self.model.eval()
                    metrics = self.evaluate(val_loader)
                    self.log_metrics(metrics)
                    self.save_topk_checkpoint(metrics)

                    if is_dist_initialized():
                        dist.barrier()

                    self.model.train()

                if self.current_step == self.max_steps:
                    break

            # Step the scheduler after the last batch of the epoch
            self.scheduler.step()

        if is_main_rank:
            pbar.close()

        logger.info("Training finished. Saving the final checkpoint...")
        self.save_checkpoint(
            dir_path=os.path.join(self.artifact_dir, self.get_latest_checkpoint_name(self.current_step)),
            state_dicts={
                "model": get_model_state_dict(self.model),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            use_safetensors={"model": True, "optimizer": False, "scheduler": False, "scaler": False},
        )

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        """
        Evaluate MiDas model

        Args:
            dataloader (DataLoader): data loader

        Returns:
            dict: metrics of the evaluation
        """
        val_metrics = {}
        for i_batch, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if self.num_val_samples is not None and i_batch >= self.num_val_samples:
                break

            with autocast(device_type="cuda", dtype=self.dtype):
                _, metrics, outputs = self.val_step(batch)

            for key, value in metrics.items():
                key = f"val/{key}"
                if key not in val_metrics:
                    val_metrics[key] = []
                val_metrics[key].append(value)

        self.log_viz(batch, outputs, stage="val")

        for key, value in val_metrics.items():
            val_metrics[key] = np.mean(value)

        return val_metrics
