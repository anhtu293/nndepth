from typing import Annotated, List
from nndepth.utils import BaseDataloaderConfig


class TartanairDisparityDataConfig(BaseDataloaderConfig):
    """Configuration for TartanAir dataset."""

    dataset_dir: Annotated[str, "Path to the TartanAir dataset"] = "/data/tartanair"
    HW: Annotated[List[int], "Height and width of the images"] = [480, 640]
    train_envs: List[str] = [
        "abandonedfactory", "amusement", "carwelding", "endofworld", "gascola",
        "hospital", "japanesealley", "neighborhood", "ocean", "office", "office2",
        "oldtown", "seasidetown", "seasonsforest", "seasonsforest_winter",
        "soulcity", "westerndesert"
    ]
    val_envs: List[str] = ["abandonedfactory_night"]
