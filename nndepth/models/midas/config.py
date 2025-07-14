from typing import Annotated, Optional

from nndepth.utils import BaseConfiguration, BaseTrainingConfig
from nndepth.data.dataloaders.configs import MultiDSDepthDataConfig


class MBNetV3DepthModelConfig(BaseConfiguration):
    """Configuration for Midas model."""

    weights: Annotated[Optional[str], "Path to pretrained weights"] = None
    strict_load: Annotated[bool, "Whether to strictly load the weights"] = True
    feature_channels: Annotated[int, "The number of feature channels"] = 64


class MidasTrainingConfig(BaseTrainingConfig):
    """Configuration for Midas training."""
    lr: Annotated[float, "Learning rate for training"] = 1e-4
    lr_decay_every_epochs: Annotated[int, "The number of steps to decay the learning rate"] = 10
    viz_log_interval: Annotated[int, "The number of steps to log the visualization"] = 500
    trimmed_loss_coef: Annotated[float, "The coefficient for trimmed loss"] = 1.0
    gradient_reg_coef: Annotated[float, "The coefficient for gradient regularization"] = 0.1
    dtype: Annotated[str, "The data type for training"] = "bfloat16"


class MBNetV3MidasTrainingConfig(BaseConfiguration):
    model: Annotated[MBNetV3DepthModelConfig, "The model configuration"] = MBNetV3DepthModelConfig()
    trainer: Annotated[MidasTrainingConfig, "The trainer configuration"] = MidasTrainingConfig()
    data: Annotated[MultiDSDepthDataConfig, "The data configuration"] = MultiDSDepthDataConfig()
