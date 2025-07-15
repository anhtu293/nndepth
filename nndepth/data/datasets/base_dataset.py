from torch.utils.data import Dataset
from typing import Callable, Any
from loguru import logger


class BaseDataset(Dataset):
    """
    Base class for all datasets.
    """

    LOADING_RETRY_LIMIT = 10

    def __init__(self, transform_fn: Callable = None, **kwargs):
        super().__init__(**kwargs)
        self.transform_fn = transform_fn

    def get_item(self, idx: int) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    def __getitem__(self, idx: int) -> Any:
        nb_retry = 0
        while nb_retry < self.LOADING_RETRY_LIMIT:
            try:
                item = self.get_item(idx)
                if self.transform_fn is not None:
                    item = self.transform_fn(item)
                return item
            except Exception as e:
                nb_retry += 1
                idx = (idx + 1) % len(self)
                logger.info(f"Error while loading item {idx}: {e}. Retry with item {idx}")

        logger.error(Exception("Error while loading data. Retry limit reached"))
        return None
