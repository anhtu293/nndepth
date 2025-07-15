from typing import Annotated, List
from nndepth.utils import BaseDataloaderConfig


class DimlDepthDataConfig(BaseDataloaderConfig):
    """Configuration for Diml Depth dataset."""

    dataset_dir: Annotated[str, "Path to the Diml Depth dataset"] = "/data/diml_depth"
    HW: Annotated[List[int], "Height and width of the images"] = [384, 1248]
    scenes: Annotated[List[str], "List of scenes"] = None
    outdoor_conf_threshold: Annotated[float, "Threshold for outdoor confidence"] = 0.6
    no_augmentation: Annotated[bool, "Whether to use augmentation"] = False
