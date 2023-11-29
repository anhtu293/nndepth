from typing import List, Dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import torch.optim as optim
import yaml

import alonet
import aloscene

from nndepth.disparity import MODELS
from nndepth.disparity.criterion import DisparityCriterion
from nndepth.disparity.callbacks import DisparityVisualizationCallback


class LitDisparityModel(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        alonet.common.pl_helpers.params_update(self, args, kwargs)
        self.model_name = self.model_config.split("/")[-1].split(".")[0]
        self.args = args
        self.model = self.build_model()
        self.criterion = self.build_criterion()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("LitDisparityModule")
        parser.add_argument("--model_config", required=True, help="Path to model json config file")
        parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
        return parent_parser

    def build_model(self):
        with open(self.model_config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        for cls in MODELS:
            if self.model_name == cls.__name__:
                return cls(**config)
        raise NotImplementedError(f"{self.model_name} is not supported !")

    def build_criterion(self):
        return DisparityCriterion()

    def assert_input(self, frames, inference=False):
        assert all(
            key in frames for key in ["left", "right"]
        ), "data loaders should output dict with 'left' and 'right' keys"
        for camera in ["left", "right"]:
            frame = frames[camera]
            assert (
                frame.normalization == "minmax_sym"
            ), f"frame.normalization should minmax_sym, not '{frame.normalization}'"
            assert frame.names in [
                tuple("BCHW"),
                tuple("BTCHW"),
            ], f"frame.names should be ('B', ['T'],'C','H','W'), not: '{frame.names}'"

            if inference:
                continue

            if camera == "left":
                assert frame.disparity is not None, "A disparity label should be attached to the frame"

    def forward(self, frames, **kwargs):
        frames = aloscene.batch_list([{key: f[key] for key in ["left", "right"]} for f in frames])
        self.assert_input(frames, inference=True)
        frame1 = frames["left"]
        frame2 = frames["right"]
        # run forward pass model
        m_outputs = self.model(frame1, frame2, **kwargs)
        return m_outputs

    def inference(self, m_outputs, **kwargs):
        return self.model.inference(m_outputs, **kwargs)

    def training_step(self, frames: List[Dict[str, aloscene.Frame]], batch_idx):
        frames = aloscene.batch_list([{key: f[key] for key in ["left", "right"]} for f in frames])
        self.assert_input(frames, inference=True)
        frame1 = frames["left"]
        frame2 = frames["right"]
        # run forward pass model
        m_outputs = self.model(frame1, frame2)
        loss, metrics, epe_per_iter = self.criterion(m_outputs, frame1)
        outputs = {"loss": loss, "metrics": metrics, "epe_per_iter": epe_per_iter}
        return outputs

    def validation_step(self, frames: List[Dict[str, aloscene.Frame]], batch_idx):
        frames = aloscene.batch_list([{key: f[key] for key in ["left", "right"]} for f in frames])
        self.assert_input(frames, inference=True)
        frame1 = frames["left"]
        frame2 = frames["right"]
        # run forward pass model
        m_outputs = self.model(frame1, frame2)
        loss, metrics, epe_per_iter = self.criterion(m_outputs, frame1, compute_per_iter=False)
        self.log("val_loss", loss.detach())
        outputs = {"val_loss": loss, "metrics": metrics, "epe_per_iter": epe_per_iter}
        return outputs

    def configure_optimizers(self, weight_decay=1e-4, epsilon=1e-8):
        params = self.model.parameters()
        optimizer = optim.AdamW(params, lr=self.lr, weight_decay=weight_decay, eps=epsilon)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.lr,
            self.args.max_steps + 100,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy="linear",
        )
        scheduler_config = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def callbacks(self, data_loader):
        """Given a data_loader, this method will return the default callbacks
        of the training loop.
        """
        data_loader.setup()
        metrics_callback = alonet.callbacks.MetricsCallback(val_names=data_loader.val_names)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        disp_viz = DisparityVisualizationCallback(data_loader)
        return [metrics_callback, lr_monitor, disp_viz]

    def run_train(
        self,
        data_loader,
        args,
        project="disparity",
        expe_name="disparity",
        callbacks: List = None,
    ):
        callbacks = self.callbacks(data_loader) if callbacks is None else callbacks
        alonet.common.pl_helpers.run_pl_training(
            self,
            data_loader=data_loader,
            callbacks=callbacks,
            args=args,
            project=project,
            expe_name=expe_name,
        )

    def run_validation(
        self,
        data_loader,
        args,
        project="disparity",
        expe_name="disparity",
        callbacks: list = None,
    ):
        """Validate the model using pytorch lightning"""
        # Set the default callbacks if not provide.
        callbacks = callbacks if callbacks is not None else self.callbacks(data_loader)

        alonet.common.pl_helpers.run_pl_validate(
            # Trainer, data & callbacks
            lit_model=self,
            data_loader=data_loader,
            callbacks=callbacks,
            # Project info
            args=args,
            project=project,
            expe_name=expe_name,
        )
