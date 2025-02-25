import torch
from torch.utils.data import RandomSampler, DataLoader
from typing import Tuple

from nndepth.scene import Frame
from nndepth.utils.base_dataloader import BaseDataLoader
from nndepth.data.datasets import KittiStereo2015


class Kitti2015DisparityDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset_dir: str = "/data/kitti/stereo_2015",
        HW: Tuple[int, int] = [375, 1242],
        **kwargs,
    ):
        """
        DataLoader for training disparity on Tartanair dataset

        Args:
            dataset_dir (str): path to Tartanair dataset
            batch_size (int): batch size
            num_workers (int): number of workers
            HW (Tuple[int, int]): image size
        """
        super().__init__(**kwargs)
        self.dataset_dir = dataset_dir
        self.HW = HW

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
        self.train_dataset = KittiStereo2015(
            dataset_dir=self.dataset_dir,
            subset="train",
            sequence_start=10,
            sequence_end=10,
            cameras=["left", "right"],
            labels=["disp_occ"],
        )
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=RandomSampler(self.train_dataset),
            drop_last=True,
            collate_fn=self.collate_fn(subset="train"),
        )
        return dataloader

    def setup_val_dataloader(self):
        self.val_dataset = KittiStereo2015(
            dataset_dir=self.dataset_dir,
            subset="train",  # disparity GT is not available for val split
            sequence_start=10,
            sequence_end=10,
            cameras=["left", "right"],
            labels=["disp_occ"],
        )
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            sampler=RandomSampler(self.val_dataset),
            collate_fn=self.collate_fn(subset="val"),
        )
        return dataloader
