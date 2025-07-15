import os
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler
from typing import Callable, Dict, List, Optional, Tuple

from nndepth.scene import Frame, Depth

from nndepth.utils import BaseDataLoader, is_dist_initialized
from nndepth.data.datasets import HRWSIDataset, TartanairDataset, DIMLDataset, HypersimDataset
from nndepth.data.augmentations import RandomHorizontalFlip, RandomResizedCrop, Compose


class MultiDatasetsDepthDataLoader(BaseDataLoader):
    """
    DataLoader for training depth on multiple datasets.
    """

    DATASETS_CLS = {
        "hrwsi": HRWSIDataset,
        "tartanair": TartanairDataset,
        "diml": DIMLDataset,
        "hypersim": HypersimDataset,
    }

    def __init__(
        self,
        dataset_names: List[str],
        HW: Tuple[int, int],
        hrwsi_kwargs: Optional[Dict] = None,
        tartanair_kwargs: Optional[Dict] = None,
        diml_kwargs: Optional[Dict] = None,
        hypersim_kwargs: Optional[Dict] = None,
        no_augmentation: bool = False,
        **kwargs,
    ):
        """
        Args:
            dataset_names (List[str]): List of dataset names to load.
                Available datasets are: "hrwsi", "tartanair", "diml", "hypersim".
            HW (Tuple[int, int]): Height and width of the images.
            hrwsi_kwargs (Optional[Dict]): Keyword arguments for HRWSI dataset.
            tartanair_kwargs (Optional[Dict]): Keyword arguments for Tartanair dataset.
            diml_kwargs (Optional[Dict]): Keyword arguments for DIML dataset.
            hypersim_kwargs (Optional[Dict]): Keyword arguments for Hypersim dataset.
            no_augmentation (bool): Whether to apply augmentation to the data.
            **kwargs: Keyword arguments for BaseDataLoader.

        Examples:
        >>> from nndepth.data.dataloaders.depth.multi_datasets_depth import MultiDatasetsDepthDataLoader
        >>> dataloader = MultiDatasetsDepthDataLoader(
                dataset_names=["hrwsi", "tartanair", "diml", "hypersim"],
            )
        """
        super().__init__(**kwargs)

        self.HW = HW
        self.datasets_kwargs = {
            "hrwsi": hrwsi_kwargs,
            "tartanair": tartanair_kwargs,
            "diml": diml_kwargs,
            "hypersim": hypersim_kwargs,
        }
        self.dataset_names = dataset_names
        self.no_augmentation = no_augmentation

        if not self.no_augmentation:
            self.train_augmentation = Compose([
                RandomResizedCrop(self.HW, p=1),
                RandomHorizontalFlip(p=0.5),
            ])

    def init_diml_dataset(self, subset: str) -> DIMLDataset:
        """
        Initialize the DIML dataset.
        """
        assert subset in ["train", "val"]
        return DIMLDataset(
            dataset_dir=self.datasets_kwargs["diml"]["diml_dataset_dir"],
            subset=subset,
            scenes=self.datasets_kwargs["diml"]["diml_scenes"],
            outdoor_conf_threshold=self.datasets_kwargs["diml"]["diml_outdoor_conf_threshold"],
        )

    def init_hrwsi_dataset(self, subset: str) -> HRWSIDataset:
        """
        Initialize the HRWSI dataset.

        Args:
            subset (str): "train" or "val".
        """
        assert subset in ["train", "val"]

        return HRWSIDataset(
            dataset_dir=self.datasets_kwargs["hrwsi"]["hrwsi_dataset_dir"],
            subset=subset,
        )

    def init_hypersim_dataset(self, subset: str) -> HypersimDataset:
        """
        Initialize the Hypersim dataset.
        """
        assert subset in ["train", "val"]

        if subset == "train":
            ds_dir = self.datasets_kwargs["hypersim"]["hypersim_dataset_dir"]
            sequences = sorted(os.listdir(ds_dir))
            sequences = [seq for seq in sequences if os.path.isdir(os.path.join(ds_dir, seq))]
            sequences = [
                seq for seq in sequences if seq not in self.datasets_kwargs["hypersim"]["hypersim_val_sequences"]
            ]
        else:
            sequences = self.datasets_kwargs["hypersim"]["hypersim_val_sequences"]

        return HypersimDataset(
            dataset_dir=self.datasets_kwargs["hypersim"]["hypersim_dataset_dir"],
            sequences=sequences,
            labels=["depth"],
        )

    def init_tartanair_dataset(self, subset: str) -> TartanairDataset:
        """
        Initialize the Tartanair dataset.
        """
        assert subset in ["train", "val"]
        if subset == "train":
            envs = self.datasets_kwargs["tartanair"]["tartanair_train_envs"]
        else:
            envs = self.datasets_kwargs["tartanair"]["tartanair_val_envs"]

        return TartanairDataset(
            dataset_dir=self.datasets_kwargs["tartanair"]["tartanair_dataset_dir"],
            envs=envs,
            cameras=["left"],
            labels=["depth"],
            transform_fn=lambda x: x["left"][0],
        )

    def init_dataset(self, dataset_name: str, subset: str) -> Dataset:
        """
        Initialize the dataset.
        """
        if dataset_name == "hrwsi":
            return self.init_hrwsi_dataset(subset)
        elif dataset_name == "tartanair":
            return self.init_tartanair_dataset(subset)
        elif dataset_name == "diml":
            return self.init_diml_dataset(subset)
        elif dataset_name == "hypersim":
            return self.init_hypersim_dataset(subset)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

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

    def setup_train_dataloader(self):
        datasets = [self.init_dataset(dataset_name, "train") for dataset_name in self.dataset_names]
        dataset = ConcatDataset(datasets)
        sampler = (
            DistributedSampler(dataset) if is_dist_initialized() else RandomSampler(dataset)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            collate_fn=self.collate_fn(subset="train"),
            pin_memory=True,
        )
        return dataloader

    def setup_val_dataloader(self):
        datasets = [self.init_dataset(dataset_name, "val") for dataset_name in self.dataset_names]
        dataset = ConcatDataset(datasets)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            collate_fn=self.collate_fn(subset="val"),
            pin_memory=True,
        )
        return dataloader
