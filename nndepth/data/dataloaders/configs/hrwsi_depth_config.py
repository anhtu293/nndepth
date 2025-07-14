from typing import Annotated, List
from nndepth.utils import BaseDataloaderConfig


class HRWSIDataConfig(BaseDataloaderConfig):
    """Configuration for HRWSI dataset."""

    dataset_dir: Annotated[str, "Path to the HRWSI dataset"] = "/data/hrwsi"
    HW: Annotated[List[int], "Height and width of the images"] = [384, 1248]
    no_augmentation: Annotated[bool, "Whether to use augmentation"] = False
