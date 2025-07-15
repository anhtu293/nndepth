from typing import Annotated, List
from nndepth.utils import BaseDataloaderConfig


class TartanairDepthDataConfig(BaseDataloaderConfig):
    """Configuration for Tartanair dataset."""

    dataset_dir: Annotated[str, "Path to the Tartanair dataset"] = "/data/tartanair"
    HW: Annotated[List[int], "Height and width of the images"] = [384, 1248]
    train_envs: Annotated[List[str], "List of training environments"] = [
        "abandonedfactory", "amusement", "carwelding", "endofworld", "gascola",
        "hospital", "japanesealley", "neighborhood", "ocean", "office", "office2",
        "oldtown", "seasidetown", "seasonsforest", "seasonsforest_winter",
        "soulcity", "westerndesert"
    ]
    val_envs: Annotated[List[str], "List of validation environments"] = ["abandonedfactory_night"]
    no_augmentation: Annotated[bool, "Whether to use augmentation"] = False
