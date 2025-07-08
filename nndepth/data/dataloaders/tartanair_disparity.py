from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import List, Tuple

from nndepth.scene import Frame
from nndepth.utils.base_dataloader import BaseDataLoader
from nndepth.data.datasets import TartanairDataset

from nndepth.utils import is_dist_initialized


class TartanairDisparityDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset_dir: str = "/data/tartanair",
        HW: Tuple[int, int] = [384, 496],
        train_envs: List[str] = ["abandonedfactory"],
        val_envs: List[str] = ["abandonedfactory_night"],
        **kwargs,
    ):
        """
        DataLoader for training disparity on Tartanair dataset

        Args:
            dataset_dir (str): path to Tartanair dataset
            batch_size (int): batch size
            num_workers (int): number of workers
            HW (Tuple[int, int]): image size
            train_envs (List[str]): list of training environments
            val_envs (List[str]): list of validation environments
        """
        super().__init__(**kwargs)
        self.dataset_dir = dataset_dir
        self.HW = HW
        self.train_envs = train_envs
        self.val_envs = val_envs

    def train_transform(self, frame: Frame):
        """train transform without augmentation"""
        frame = frame.resize(self.HW)
        frame.data = (frame.data - 127.5) / 127.5
        return frame

    def val_transform(self, frame: Frame):
        frame = frame.resize(self.HW)
        frame.data = (frame.data - 127.5) / 127.5
        return frame

    def collate_fn(self, subset: str):
        assert subset in ["train", "val"]

        def collate(batch):
            frames = {"left": [], "right": []}
            transform_fn = self.train_transform if subset == "train" else self.val_transform
            for f in batch:
                if f is None:
                    continue
                # Remove temporal dimension & Transform
                frames["left"].append(transform_fn(f["left"][0]))
                frames["right"].append(transform_fn(f["right"][0]))

            return frames

        return collate

    def setup_train_dataloader(self) -> DataLoader:
        self.train_dataset = TartanairDataset(
            dataset_dir=self.dataset_dir,
            envs=self.train_envs,
            cameras=["left", "right"],
            labels=["disparity"],
        )
        sampler = (
            DistributedSampler(self.train_dataset) if is_dist_initialized() else RandomSampler(self.train_dataset)
        )
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=True,
            collate_fn=self.collate_fn(subset="train"),
        )
        return dataloader

    def setup_val_dataloader(self):
        self.val_dataset = TartanairDataset(
            dataset_dir=self.dataset_dir,
            envs=self.val_envs,
            cameras=["left", "right"],
            labels=["disparity"],
        )
        sampler = (
            DistributedSampler(self.val_dataset) if is_dist_initialized() else RandomSampler(self.val_dataset)
        )
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            sampler=sampler,
            collate_fn=self.collate_fn(subset="val"),
        )
        return dataloader
