from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from typing import List, Tuple, Callable

from nndepth.scene import Frame, Depth
from nndepth.utils.base_dataloader import BaseDataLoader
from nndepth.data.datasets import TartanairDataset
from nndepth.data.augmentations import RandomHorizontalFlip, RandomResizedCrop, Compose

from nndepth.utils import is_dist_initialized


class TartanairDepthDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset_dir: str = "/data/tartanair",
        HW: Tuple[int, int] = [384, 496],
        train_envs: List[str] = ["abandonedfactory"],
        val_envs: List[str] = ["abandonedfactory_night"],
        no_augmentation: bool = False,
        **kwargs,
    ):
        """
        DataLoader for training depth on Tartanair dataset

        Args:
            dataset_dir (str): path to Tartanair dataset
            batch_size (int): batch size
            num_workers (int): number of workers
            HW (Tuple[int, int]): image size
            train_envs (List[str]): list of training environments
            val_envs (List[str]): list of validation environments
            no_augmentation (bool): whether to use augmentation
            **kwargs: other arguments for BaseDataLoader
        """
        super().__init__(**kwargs)
        self.dataset_dir = dataset_dir
        self.HW = HW
        self.train_envs = train_envs
        self.val_envs = val_envs
        self.no_augmentation = no_augmentation

        if not no_augmentation:
            self.train_augmentation = Compose([
                RandomResizedCrop(self.HW, p=1),
                RandomHorizontalFlip(p=0.5),
            ])

    def train_transform(self, frame: Frame) -> Frame:
        if not self.no_augmentation:
            frame = self.train_augmentation(frame)
        frame.data = (frame.data - 127.5) / 127.5
        frame.depth = frame.depth.inverse(clip_max=100, clip_min=0.01)
        return frame

    def val_transform(self, frame: Frame) -> Frame:
        frame = frame.resize(self.HW)
        frame.data = (frame.data - 127.5) / 127.5
        frame.depth = frame.depth.inverse(clip_max=100, clip_min=0.01)
        return frame

    def collate_fn(self, subset: str) -> Callable:
        assert subset in ["train", "val"]

        def collate(batch) -> Frame:
            transform_fn = self.train_transform if subset == "train" else self.val_transform
            frames = [transform_fn(frame["left"][0]) for frame in batch if frame is not None]

            data = torch.stack([frame.data for frame in frames], dim=0)
            depth = torch.stack([frame.depth.data for frame in frames], dim=0)
            valid_mask = torch.stack([frame.depth.valid_mask for frame in frames], dim=0)
            batched_depth = Depth(data=depth, valid_mask=valid_mask)

            return Frame(data=data, depth=batched_depth)

        return collate

    def setup_train_dataloader(self) -> DataLoader:
        self.train_dataset = TartanairDataset(
            dataset_dir=self.dataset_dir,
            envs=self.train_envs,
            cameras=["left"],
            labels=["depth"],
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
            cameras=["left"],
            labels=["depth"],
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
