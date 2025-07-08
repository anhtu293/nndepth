"""
Configuration classes for RAFT Stereo model.
These classes inherit from BaseConfiguration and use type annotations for automatic CLI generation.
"""

from typing import Annotated, List, Optional, Tuple
from nndepth.utils import BaseConfiguration, BaseTrainingConfig
from nndepth.data.dataloaders.configs import TartanairDisparityDataConfig


class BaseRAFTStereoModelConfig(BaseConfiguration):
    """Configuration for RAFT Stereo model parameters."""

    iters: int = 12
    fnet_dim: int = 256
    hidden_dim: int = 128
    context_dim: int = 64
    corr_levels: int = 4
    corr_radius: int = 4
    tracing: bool = False
    include_preprocessing: bool = False
    weights: Optional[str] = None
    strict_load: bool = True


class RAFTTrainerConfig(BaseTrainingConfig):
    """Configuration for RAFT Stereo trainer."""

    lr: Annotated[float, "Learning rate for training"] = 0.0001
    weight_decay: Annotated[float, "Weight decay for training"] = 0.0001
    epsilon: Annotated[float, "Epsilon for optimizer"] = 1e-08
    dtype: Annotated[str, "Data type for training"] = "bfloat16"
    device: Annotated[str, "Device for training"] = "cuda"


class RepViTRAFTStereoModelConfig(BaseRAFTStereoModelConfig):
    """Configuration for Coarse2FineGroupRepViTRAFTStereo model parameters."""

    num_groups: Annotated[int, "Number of groups"] = 4
    downsample_ratios: Annotated[List[Tuple[int, int]], "Downsample ratios for each stage"] = [
        (2, 2), (2, 2), (2, 2), (2, 2)
    ]
    ffn_exp_ratios: Annotated[List[float], "FFN expansion ratios"] = [1.0, 3.0, 3.0, 4.0]
    num_blocks_per_stage: Annotated[List[int], "Number of blocks per stage"] = [4, 4, 6, 2]
    patch_size: Annotated[int, "Patch size"] = 7
    stem_strides: Annotated[List[Tuple[int, int]], "Stem strides for each stage"] = [(2, 2), (2, 2), (1, 1)]
    token_mixer_types: Annotated[List[str], "Token mixer types"] = ["repmixer", "repmixer", "repmixer", "attention"]


class BaseRAFTTrainingConfig(BaseConfiguration):
    """Configuration for Base RAFT Stereo training."""

    model: Annotated[BaseRAFTStereoModelConfig, "Model configuration"] = BaseRAFTStereoModelConfig()
    data: Annotated[TartanairDisparityDataConfig, "Data configuration"] = TartanairDisparityDataConfig()
    trainer: Annotated[RAFTTrainerConfig, "Trainer configuration"] = RAFTTrainerConfig()


class RepViTRAFTStereoTrainingConfig(BaseConfiguration):
    """Configuration for RepViTRAFTStereo training."""

    model: Annotated[RepViTRAFTStereoModelConfig, "Model configuration"] = RepViTRAFTStereoModelConfig()
    data: Annotated[TartanairDisparityDataConfig, "Data configuration"] = TartanairDisparityDataConfig()
    trainer: Annotated[RAFTTrainerConfig, "Trainer configuration"] = RAFTTrainerConfig()
