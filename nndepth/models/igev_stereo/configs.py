"""
Configuration classes for IGEV Stereo model.
These classes inherit from BaseConfiguration and use type annotations for automatic CLI generation.
"""

from typing import Annotated, Optional
from nndepth.utils import BaseConfiguration, BaseTrainingConfig
from nndepth.data.dataloaders.configs import TartanairDisparityDataConfig


class BaseIGEVStereoModelConfig(BaseConfiguration):
    """Configuration for IGEV Stereo model parameters."""

    update_cls: Annotated[str, "The class name of the update block to use"] = "basic_update_block"
    cv_groups: Annotated[int, "The number of groups to split the cost volume into"] = 8
    iters: Annotated[int, "Number of iterations"] = 6
    hidden_dim: Annotated[int, "The hidden dimension of the update block"] = 64
    context_dim: Annotated[int, "The context dimension of the update block"] = 64
    corr_levels: Annotated[int, "The number of correlation levels to compute"] = 4
    corr_radius: Annotated[int, "The radius of the correlation window"] = 4
    tracing: Annotated[bool, "Whether to enable tracing for ONNX exportation"] = False
    include_preprocessing: Annotated[bool, "Whether to include preprocessing steps in tracing"] = False
    weights: Annotated[Optional[str], "Path to pretrained weights"] = None
    strict_load: Annotated[bool, "Whether to strictly load the weights"] = True


class IGEVStereoMBNetModelConfig(BaseIGEVStereoModelConfig):
    """Configuration for IGEV Stereo MBNet model."""
    update_cls: Annotated[str, "The class name of the update block to use"] = "basic_update_block"
    cv_groups: Annotated[int, "The number of groups to split the cost volume into"] = 8
    iters: Annotated[int, "Number of iterations"] = 6
    hidden_dim: Annotated[int, "The hidden dimension of the update block"] = 64
    context_dim: Annotated[int, "The context dimension of the update block"] = 64
    corr_levels: Annotated[int, "The number of correlation levels to compute"] = 4
    corr_radius: Annotated[int, "The radius of the correlation window"] = 4
    tracing: Annotated[bool, "Whether to enable tracing for ONNX exportation"] = False
    include_preprocessing: Annotated[bool, "Whether to include preprocessing steps in tracing"] = False
    weights: Annotated[Optional[str], "Path to pretrained weights"] = None
    strict_load: Annotated[bool, "Whether to strictly load the weights"] = True


class IGEVStereoTrainerConfig(BaseTrainingConfig):
    """Configuration for IGEV Stereo trainer."""

    lr: Annotated[float, "Learning rate for training"] = 0.0001
    weight_decay: Annotated[float, "Weight decay for training"] = 0.0001
    epsilon: Annotated[float, "Epsilon for optimizer"] = 1e-08
    dtype: Annotated[str, "Data type for training"] = "bfloat16"


class BaseIGEVStereoTrainingConfig(BaseConfiguration):
    """Configuration for Base IGEV Stereo training."""

    model: Annotated[BaseIGEVStereoModelConfig, "Model configuration"] = BaseIGEVStereoModelConfig()
    data: Annotated[TartanairDisparityDataConfig, "Data configuration"] = TartanairDisparityDataConfig()
    trainer: Annotated[IGEVStereoTrainerConfig, "Trainer configuration"] = IGEVStereoTrainerConfig()


class MBNetIGEVStereoTrainingConfig(BaseIGEVStereoTrainingConfig):
    """Configuration for MBNet IGEV Stereo training."""

    model: Annotated[IGEVStereoMBNetModelConfig, "Model configuration"] = IGEVStereoMBNetModelConfig()
    data: Annotated[TartanairDisparityDataConfig, "Data configuration"] = TartanairDisparityDataConfig()
    trainer: Annotated[IGEVStereoTrainerConfig, "Trainer configuration"] = IGEVStereoTrainerConfig()
