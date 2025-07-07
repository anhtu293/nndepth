from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


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
