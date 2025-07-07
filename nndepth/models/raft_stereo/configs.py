"""
Configuration classes for RAFT Stereo model.
These classes inherit from BaseConfiguration and use type annotations for automatic CLI generation.
"""

from typing import Annotated, List, Optional, Tuple
from nndepth.utils import BaseConfiguration


class BaseRAFTStereoModelConfig(BaseConfiguration):
    """Configuration for RAFT Stereo model parameters."""

    # Type annotations automatically generate CLI arguments
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


class TartanairDataConfig(BaseConfiguration):
    """Configuration for TartanAir dataset."""

    # Type annotations automatically generate CLI arguments
    dataset_dir: Annotated[str, "Path to the TartanAir dataset"] = "/data/tartanair"
    HW: Annotated[List[int], "Height and width of the images"] = [480, 640]
    batch_size: Annotated[int, "Batch size for training"] = 6
    num_workers: Annotated[int, "Number of workers for data loading"] = 8
    train_envs: List[str] = [
        "abandonedfactory", "amusement", "carwelding", "endofworld", "gascola",
        "hospital", "japanesealley", "neighborhood", "ocean", "office", "office2",
        "oldtown", "seasidetown", "seasonsforest", "seasonsforest_winter",
        "soulcity", "westerndesert"
    ]
    val_envs: List[str] = ["abandonedfactory_night"]


class RAFTTrainerConfig(BaseConfiguration):
    """Configuration for RAFT Stereo trainer."""

    # Type annotations automatically generate CLI arguments
    lr: Annotated[float, "Learning rate for training"] = 0.0001
    num_epochs: Annotated[int, "Number of epochs for training"] = 100
    max_steps: Annotated[int, "Maximum number of steps for training"] = 100000
    weight_decay: Annotated[float, "Weight decay for training"] = 0.0001
    epsilon: Annotated[float, "Epsilon for optimizer"] = 1e-08
    gradient_accumulation_steps: Annotated[int, "Number of gradient accumulation steps"] = 4
    workdir: Annotated[str, "Path to the working directory"] = "/weights"
    project_name: Annotated[str, "Name of the project"] = "raft_stereo"
    experiment_name: Annotated[str, "Name of the experiment"] = "BaseRaftStereo"
    val_interval: Annotated[float, "Validation interval"] = 10.0
    log_interval: Annotated[int, "Logging interval"] = 50
    num_val_samples: Annotated[int, "Number of validation samples"] = 10
    save_best_k_cp: Annotated[int, "Number of best checkpoints to save"] = 3


class Coarse2FineGroupRepViTRAFTStereoModelConfig(BaseRAFTStereoModelConfig):
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
    data: Annotated[TartanairDataConfig, "Data configuration"] = TartanairDataConfig()
    trainer: Annotated[RAFTTrainerConfig, "Trainer configuration"] = RAFTTrainerConfig()


class RepViTRAFTStereoTrainingConfig(BaseRAFTTrainingConfig):
    """Configuration for RepViTRAFTStereo training."""

    model: Annotated[Coarse2FineGroupRepViTRAFTStereoModelConfig, "Model configuration"] = Coarse2FineGroupRepViTRAFTStereoModelConfig()
    data: Annotated[TartanairDataConfig, "Data configuration"] = TartanairDataConfig()
    trainer: Annotated[RAFTTrainerConfig, "Trainer configuration"] = RAFTTrainerConfig()
