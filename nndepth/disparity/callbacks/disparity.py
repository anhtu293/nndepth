import pytorch_lightning as pl
import numpy as np
import torch
from typing import List, Dict

import aloscene
from alonet.common.logger import log_image


class DisparityVisualizationCallback(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()
        self.datamodule = datamodule
        self.names = datamodule.val_names
        self.val_frames_list = self._init_val_frames()
        if len(self.names) != len(self.val_frames_list):
            raise Exception("Number of dataset names and images should not be different")

    def _init_val_frames(self):
        """
        Read the first image of val dataset
        """
        val_loaders = self.datamodule.val_dataloader()
        val_frames_list = [next(iter(val_loaders))]
        return val_frames_list

    def get_disp_img(self, disp, vmin, vmax):
        # disp = replace_infinite_and_nan(disp)
        disp_img = disp.__get_view__(min_disp=vmin, max_disp=vmax).image
        disp_img = (disp_img * 255).astype(np.uint8)
        return disp_img

    def log_disp_validation(
        self,
        frames: List[Dict[str, aloscene.Frame]],
        disparities: List[aloscene.Disparity],
        trainer: pl.trainer.trainer.Trainer,
        name: str,
    ):
        # Log ground-truth
        disp_gt = frames[0]["left"].disparity.detach().cpu()
        disp_min = 0
        disp_max = disp_gt.abs().max().item()
        disp_gt_img = self.get_disp_img(disp_gt, disp_min, disp_max)
        log_image(trainer, f"{name}/disp/gt/disp_gt", [{"image": disp_gt_img}])

        # Log each iterations
        for it, disp_pred in enumerate(disparities):
            disp_pred = disp_pred[0].detach().cpu()
            disp_pred_img = self.get_disp_img(disp_pred, disp_min, disp_max)
            log_image(trainer, f"{name}/disp/pred_iter/disp_iter{it}", [{"image": disp_pred_img}])

        # Log final disp
        disp_final = disparities[-1]
        disp_pred_img = self.get_disp_img(disp_final[0].detach().cpu(), disp_min, disp_max)
        log_image(trainer, f"{name}/disp/pred_final/disp_final", [{"image": disp_pred_img}])

    def log_images_validation(self, frames: aloscene.Frame, trainer: pl.trainer.trainer.Trainer, name: str):
        for camera in ["left", "right"]:
            frame = frames[0][camera]  # first batch
            frame = frame.detach().norm255().cpu().type(torch.uint8).numpy().transpose(1, 2, 0)

            # log image
            log_name = f"{name}/images/{camera}"
            log_data = [{"image": frame}]
            log_image(trainer, log_name, log_data)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.logger is None:
            return

        for name, val_frames in zip(self.names, self.val_frames_list):
            # send data to pl_module device
            for idx, frame in enumerate(val_frames):
                for camera in ["left", "right"]:
                    if frame[camera].device != pl_module.device:
                        val_frames[idx][camera] = frame[camera].to(pl_module.device)

            disparities = pl_module(val_frames, only_last=False)
            disparities = pl_module.inference(disparities, only_last=False)

            self.log_images_validation(val_frames, trainer, name)
            self.log_disp_validation(val_frames, disparities, trainer, name)
