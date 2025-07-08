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
        self._train_dataloader = None
        self._val_dataloader = None

    @property
    def train_dataloader(self) -> DataLoader:
        """
        Get the training dataloader
        """
        if self._train_dataloader is None:
            raise ValueError("Train dataloader is not set")
        return self._train_dataloader

    @property
    def val_dataloader(self) -> DataLoader:
        """
        Get the validation dataloader
        """
        if self._val_dataloader is None:
            raise ValueError("Validation dataloader is not set")
        return self._val_dataloader

    @train_dataloader.setter
    def train_dataloader(self, value: DataLoader):
        """
        Set the training dataloader
        """
        self._train_dataloader = value

    @val_dataloader.setter
    def val_dataloader(self, value: DataLoader):
        """
        Set the validation dataloader
        """
        self._val_dataloader = value

    def setup(self, stage: str = "train"):
        if stage == "train":
            self._train_dataloader = self.setup_train_dataloader()
            self._val_dataloader = self.setup_val_dataloader()
        elif stage == "val":
            self._val_dataloader = self.setup_val_dataloader()
        else:
            raise ValueError(f"Invalid stage: {stage}")

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
