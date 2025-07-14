from typing import Annotated, List
from nndepth.utils import BaseDataloaderConfig, BaseConfiguration


class DimlDepthDatasetConfig(BaseConfiguration):
    """Configuration for DIML dataset."""

    diml_dataset_dir: Annotated[str, "Path to the DIML dataset"] = "/data/diml"
    diml_scenes: Annotated[List[str], "List of scenes"] = None
    diml_outdoor_conf_threshold: Annotated[float, "Threshold for outdoor confidence"] = 0.6


class HRWSIDatasetConfig(BaseConfiguration):
    """Configuration for HRWSI dataset."""

    hrwsi_dataset_dir: Annotated[str, "Path to the HRWSI dataset"] = "/data/hrwsi"


class HypersimDatasetConfig(BaseConfiguration):
    """Configuration for Hypersim dataset."""

    hypersim_dataset_dir: Annotated[str, "Path to the Hypersim dataset"] = "/data/hypersim"
    hypersim_val_sequences: Annotated[List[str], "List of val sequences"] = ["ai_055_001", "ai_055_002", "ai_055_003", "ai_055_004", "ai_055_005", "ai_055_006", "ai_055_007", "ai_055_008", "ai_055_009", "ai_055_010"]


class TartanairDatasetConfig(BaseConfiguration):
    """Configuration for Tartanair dataset."""

    tartanair_dataset_dir: Annotated[str, "Path to the Tartanair dataset"] = "/data/tartanair"
    tartanair_train_envs: Annotated[List[str], "List of train environments"] = [
        "abandonedfactory", "amusement", "carwelding", "endofworld", "gascola",
        "hospital", "japanesealley", "neighborhood", "ocean", "office", "office2",
        "oldtown", "seasidetown", "seasonsforest", "seasonsforest_winter",
        "soulcity", "westerndesert"
    ]
    tartanair_val_envs: Annotated[List[str], "List of val environments"] = ["abandonedfactory_night"]


class MultiDSDepthDataConfig(BaseDataloaderConfig):
    """Configuration for MultiDSDepth dataset."""

    dataset_names: Annotated[List[str], "List of dataset names"] = ["hrwsi", "tartanair", "diml", "hypersim"]
    HW: Annotated[List[int], "Height and width of the images"] = [384, 1248]
    hrwsi_kwargs: Annotated[HRWSIDatasetConfig, "Keyword arguments for HRWSI dataset"] = HRWSIDatasetConfig()
    tartanair_kwargs: Annotated[TartanairDatasetConfig, "Keyword arguments for Tartanair dataset"] = TartanairDatasetConfig()
    diml_kwargs: Annotated[DimlDepthDatasetConfig, "Keyword arguments for DIML dataset"] = DimlDepthDatasetConfig()
    hypersim_kwargs: Annotated[HypersimDatasetConfig, "Keyword arguments for Hypersim dataset"] = HypersimDatasetConfig()
    no_augmentation: Annotated[bool, "Whether to use augmentation"] = False
