import os
import torch
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import List, Tuple, Optional, Callable

from nndepth.scene import Frame, Depth
from nndepth.utils.base_dataloader import BaseDataLoader
from nndepth.data.datasets import HypersimDataset
from nndepth.data.augmentations import RandomHorizontalFlip, RandomResizedCrop, Compose

from nndepth.utils import is_dist_initialized


class HypersimDepthDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset_dir: str = "/data/hypersim",
        HW: Tuple[int, int] = [384, 496],
        val_sequences: Optional[List[str]] = None,
        no_augmentation: bool = False,
        **kwargs,
    ):
        """
        DataLoader for training depth on Hypersim dataset

        Args:
            dataset_dir (str): path to Hypersim dataset
            batch_size (int): batch size
            num_workers (int): number of workers
            HW (Tuple[int, int]): image size
            sequences (Optional[List[str]]): list of sequences to load
            no_augmentation (bool): whether to use augmentation
            **kwargs: other arguments for BaseDataLoader
        """
        super().__init__(**kwargs)
        self.dataset_dir = dataset_dir
        self.HW = HW
        self.val_sequences = val_sequences
        self.no_augmentation = no_augmentation

        if not self.no_augmentation:
            self.train_augmentation = Compose([
                RandomResizedCrop(self.HW, p=1),
                RandomHorizontalFlip(p=0.5),
            ])

    def train_transform(self, frame: Frame) -> Frame:
        if not self.no_augmentation:
            frame = self.train_augmentation(frame)
        else:
            frame = frame.resize(self.HW)
        frame.data = (frame.data - 127.5) / 127.5
        frame.depth = frame.depth.inverse(clip_max=None, clip_min=None)

        return frame

    def val_transform(self, frame: Frame) -> Frame:
        frame = frame.resize(self.HW)
        frame.data = (frame.data - 127.5) / 127.5
        frame.depth = frame.depth.inverse(clip_max=None, clip_min=None)

        return frame

    def collate_fn(self, subset: str) -> Callable:
        assert subset in ["train", "val"]

        def collate(batch) -> Frame:
            transform_fn = self.train_transform if subset == "train" else self.val_transform
            frames = [transform_fn(frame) for frame in batch if frame is not None]

            data = torch.stack([frame.data for frame in frames], dim=0)
            depth = torch.stack([frame.depth.data for frame in frames], dim=0)
            valid_mask = torch.stack([frame.depth.valid_mask for frame in frames], dim=0)
            batched_depth = Depth(data=depth, valid_mask=valid_mask)

            return Frame(data=data, depth=batched_depth)

        return collate

    def setup_train_dataloader(self) -> DataLoader:
        ds_dir = os.path.join(self.dataset_dir, "train")
        sequences = sorted(os.listdir(ds_dir))
        sequences = [seq for seq in sequences if os.path.isdir(os.path.join(ds_dir, seq))]
        sequences = [seq for seq in sequences if seq not in self.val_sequences]

        self.train_dataset = HypersimDataset(
            dataset_dir=self.dataset_dir,
            sequences=sequences,
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
        self.val_dataset = HypersimDataset(
            dataset_dir=self.dataset_dir,
            sequences=self.val_sequences,
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
