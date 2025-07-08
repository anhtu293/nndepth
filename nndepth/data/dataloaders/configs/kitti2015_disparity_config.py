from typing import Annotated, List
from nndepth.utils import BaseDataloaderConfig


class Kitti2015DisparityDataConfig(BaseDataloaderConfig):
    """Configuration for Kitti 2015 dataset."""

    dataset_dir: Annotated[str, "Path to the Kitti 2015 dataset"] = "/data/kitti/stereo_2015"
    HW: Annotated[List[int], "Height and width of the images"] = [384, 1248]
