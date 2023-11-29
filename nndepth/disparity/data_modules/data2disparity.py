import yaml
from argparse import Namespace
from torch.utils.data import SequentialSampler, RandomSampler
import pytorch_lightning as pl
from typing import Tuple
import alonet


class Data2DisparityModel(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 5,
        num_workers: int = 8,
        sequential: bool = False,
        HW: Tuple[int, int] = [384, 496],
        args: Namespace = None,
        **kwargs
    ):
        default_cfg = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "sequential": sequential,
            "HW": HW,
        }
        if args is not None and args.data_config is not None:
            with open(args.data_config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            self.config = {**default_cfg, **config}
        else:
            self.config = default_cfg

        for key, val in self.config.items():
            setattr(self, key, val)
        super().__init__()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Data2DisparityModel")
        parser.add_argument("--data_config", type=str, default=None, help="Path to data config file")
        return parent_parser

    def train_transform(self, frames):
        """train transform without augmentation, but crop for multiple of 8"""
        transformed = {}
        for key in ["left", "right"]:
            frame = frames[key]
            frame = frame.resize(self.HW)
            frame = frame.norm_minmax_sym()
            transformed[key] = frame

        for key in frames:
            if key not in ["left", "right"]:
                transformed[key] = frames[key]

        return transformed

    def val_transform(self, frames):
        transformed = {}
        for key in ["left", "right"]:
            frame = frames[key]
            frame = frame.resize(self.HW)
            frame = frame.norm_minmax_sym()
            transformed[key] = frame

        for key in frames:
            if key not in ["left", "right"]:
                transformed[key] = frames[key]

        return transformed

    def train_dataloader(self):
        """
        Return a new dataloader for the training dataset
        """
        sampler = SequentialSampler if self.sequential else RandomSampler
        return self.train_dataset.train_loader(
            batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler
        )

    def val_dataloader(self):
        """
        Return a new dataloader for the validation dataset
        """
        return self.val_dataset.train_loader(batch_size=1, num_workers=self.num_workers, sampler=RandomSampler)

    def prepare_data(self):
        pass

    def _setup_train_dataset(self):
        """
        Setup the train dataset and assign it to self._train_dataset
        """
        raise NotImplementedError("Should be implemented in child class.")

    def _setup_val_dataset(self):
        """
        Setup the val dataset and assign it to self._val_dataset
        """
        raise NotImplementedError("Should be implemented in child class.")

    def setup(self, stage=None):
        self._setup_train_dataset()
        self._setup_val_dataset()
