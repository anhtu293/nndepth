from torch import nn
import yaml
from typing import Union, Tuple


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def init_from_config(cls, config: Union[dict, str]) -> Tuple["BaseModel", dict]:
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**config), config
