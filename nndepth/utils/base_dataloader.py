from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import Tuple, Union
import yaml


class BaseDataLoader(ABC):
    def __init__(
        self,
        batch_size: int = 5,
        num_workers: int = 8,
    ):
        """
        Base class for all data loaders
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

    @classmethod
    def init_from_config(cls, config: Union[dict, str]) -> Tuple["BaseDataLoader", dict]:
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**config), config

    def setup(self):
        self.train_dataloader = self.setup_train_dataloader()
        self.val_dataloader = self.setup_val_dataloader()

    @abstractmethod
    def setup_train_dataloader(self) -> DataLoader:
        """
        Return a new dataloader for the training dataset
        """
        pass

    @abstractmethod
    def setup_val_dataloader(self) -> DataLoader:
        """
        Return a new dataloader for the validation dataset
        """
        pass
