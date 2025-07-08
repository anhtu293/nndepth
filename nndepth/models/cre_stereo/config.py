"""
Configuration classes for CRE Stereo model.
These classes inherit from BaseConfiguration and use type annotations for automatic CLI generation.
"""

from typing import Annotated, Optional
from nndepth.utils import BaseConfiguration, BaseTrainingConfig
from nndepth.data.dataloaders.configs import TartanairDisparityDataConfig


class BaseCREStereoModelConfig(BaseConfiguration):
    """Configuration for CRE Stereo model parameters."""

    fnet_cls: Annotated[str, "Feature extraction network class"] = "basic_encoder"
    update_cls: Annotated[str, "Update block class"] = "basic_update_block"
    iters: Annotated[int, "Number of iterations"] = 12
    max_disp: Annotated[int, "Maximum disparity value"] = 192
    num_fnet_channels: Annotated[int, "Number of channels in feature network"] = 256
    hidden_dim: Annotated[int, "Hidden dimension size"] = 128
    context_dim: Annotated[int, "Context dimension size"] = 128
    search_num: Annotated[int, "Number of search iterations"] = 9
    mixed_precision: Annotated[bool, "Use mixed precision training"] = False
    test_mode: Annotated[bool, "Run model in test mode"] = False
    tracing: Annotated[bool, "Enable tracing"] = False
    include_preprocessing: Annotated[bool, "Include preprocessing steps"] = False
    weights: Annotated[Optional[str], "Path to pretrained weights"] = None
    strict_load: Annotated[bool, "Strict loading of weights"] = True


class CRETrainerConfig(BaseTrainingConfig):
    """Configuration for CRE Stereo trainer."""

    lr: Annotated[float, "Learning rate for training"] = 0.0001
    weight_decay: Annotated[float, "Weight decay for training"] = 0.0001
    epsilon: Annotated[float, "Epsilon for optimizer"] = 1e-08
    dtype: Annotated[str, "Data type for training"] = "bfloat16"
    device: Annotated[str, "Device for training"] = "cuda"


class BaseCRETrainingConfig(BaseConfiguration):
    """Configuration for Base CRE Stereo training."""

    model: Annotated[BaseCREStereoModelConfig, "Model configuration"] = BaseCREStereoModelConfig()
    data: Annotated[TartanairDisparityDataConfig, "Data configuration"] = TartanairDisparityDataConfig()
    trainer: Annotated[CRETrainerConfig, "Trainer configuration"] = CRETrainerConfig()
