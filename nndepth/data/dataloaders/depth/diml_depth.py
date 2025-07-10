import torch
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import List, Tuple, Optional

from nndepth.scene import Frame, Depth
from nndepth.utils.base_dataloader import BaseDataLoader
from nndepth.data.datasets import DIMLDataset
from nndepth.data.augmentations import RandomHorizontalFlip, RandomResizedCrop, Compose

from nndepth.utils import is_dist_initialized


class DIMLDepthDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset_dir: str = "/data/diml",
        HW: Tuple[int, int] = [384, 496],
        scenes: Optional[List[str]] = None,
        outdoor_conf_threshold: float = 0.6,
        no_augmentation: bool = False,
        **kwargs,
    ):
        """
        DataLoader for training depth on DIML dataset

        Args:
            dataset_dir (str): path to DIML dataset
            batch_size (int): batch size
            num_workers (int): number of workers
            HW (Tuple[int, int]): image size
            scenes (Optional[List[str]]): list of scenes
            outdoor_conf_threshold (float): threshold for outdoor confidence
            no_augmentation (bool): whether to use augmentation
            **kwargs: other arguments for BaseDataLoader
        """
        super().__init__(**kwargs)
        self.dataset_dir = dataset_dir
        self.HW = HW
        self.scenes = scenes
        self.outdoor_conf_threshold = outdoor_conf_threshold
        self.no_augmentation = no_augmentation

        if not self.no_augmentation:
            self.train_augmentation = Compose([
                RandomResizedCrop(self.HW, p=1),
                RandomHorizontalFlip(p=0.5),
            ])

    def train_transform(self, frame: Frame):
        if not self.no_augmentation:
            frame = self.train_augmentation(frame)
        else:
            frame = frame.resize(self.HW)
        frame.data = (frame.data - 127.5) / 127.5
        frame.depth = frame.depth.inverse(clip_max=100, clip_min=0.01)
        return frame

    def val_transform(self, frame: Frame):
        frame = frame.resize(self.HW)
        frame.data = (frame.data - 127.5) / 127.5
        frame.depth = frame.depth.inverse(clip_max=100, clip_min=0.01)
        return frame

    def collate_fn(self, subset: str):
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
        self.train_dataset = DIMLDataset(
            dataset_dir=self.dataset_dir,
            subset="train",
            scenes=self.scenes,
            outdoor_conf_threshold=self.outdoor_conf_threshold,
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
        self.val_dataset = DIMLDataset(
            dataset_dir=self.dataset_dir,
            scenes=self.scenes,
            outdoor_conf_threshold=self.outdoor_conf_threshold,
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
        )
        return dataloader


if __name__ == "__main__":
    import numpy as np
    import cv2
    dataloader = DIMLDepthDataLoader(
        dataset_dir="/data/diml",
        HW=(384, 496),
        outdoor_conf_threshold=0.6,
        no_augmentation=False,
    )
    dataloader.setup("train")

    for batch in dataloader.train_dataloader:
        image = batch.data[0].permute(1, 2, 0).cpu().numpy()
        depth = batch.depth.data[0]
        valid_mask = batch.depth.valid_mask[0]

        depth = Depth(data=depth, valid_mask=valid_mask)
        image = (image * 127.5 + 127.5).astype(np.uint8)
        depth = depth.inverse(clip_max=100, clip_min=0.01)
        depth_view = depth.get_view()

        image = np.concatenate([image, depth_view], axis=1)
        cv2.imwrite("frame.png", image)
        break
