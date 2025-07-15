import torch
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple, Callable

from nndepth.scene import Frame, Depth
from nndepth.utils.base_dataloader import BaseDataLoader
from nndepth.data.datasets import HRWSIDataset
from nndepth.data.augmentations import RandomHorizontalFlip, RandomResizedCrop, Compose

from nndepth.utils import is_dist_initialized


class HRWSIDepthDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset_dir: str = "/data/hrwsi",
        HW: Tuple[int, int] = [384, 496],
        no_augmentation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_dir = dataset_dir
        self.HW = HW
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
        min_depth = frame.depth.data[frame.depth.valid_mask].min()
        max_depth = frame.depth.data[frame.depth.valid_mask].max()
        frame.depth.data = (frame.depth.data - min_depth) / (max_depth - min_depth)

        return frame

    def val_transform(self, frame: Frame) -> Frame:
        frame = frame.resize(self.HW)
        frame.data = (frame.data - 127.5) / 127.5
        frame.depth = frame.depth.inverse(clip_max=None, clip_min=None)
        min_depth = frame.depth.data[frame.depth.valid_mask].min()
        max_depth = frame.depth.data[frame.depth.valid_mask].max()
        frame.depth.data = (frame.depth.data - min_depth) / (max_depth - min_depth)

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
        self.train_dataset = HRWSIDataset(
            dataset_dir=self.dataset_dir,
            subset="train",
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
            pin_memory=True,
        )
        return dataloader

    def setup_val_dataloader(self):
        self.val_dataset = HRWSIDataset(
            dataset_dir=self.dataset_dir,
            subset="val",
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
            pin_memory=True,
        )
        return dataloader
