from typing import Annotated, List
from nndepth.utils import BaseDataloaderConfig


class HypersimDepthDataConfig(BaseDataloaderConfig):
    """Configuration for Hypersim dataset."""

    dataset_dir: Annotated[str, "Path to the Hypersim dataset"] = "/data/hypersim"
    HW: Annotated[List[int], "Height and width of the images"] = [384, 1248]
    val_sequences: Annotated[List[str], "List of val sequences"] = [
        "ai_055_001",
        "ai_055_002",
        "ai_055_003",
        "ai_055_004",
        "ai_055_005",
        "ai_055_006",
        "ai_055_007",
        "ai_055_008",
        "ai_055_009",
        "ai_055_010",
    ]
    no_augmentation: Annotated[bool, "Whether to use augmentation"] = False
