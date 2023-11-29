from argparse import Namespace
from torch.utils.data import SequentialSampler, RandomSampler
from typing import List

from nndepth.datasets import TartanairDataset
from nndepth.disparity.data_modules.data2disparity import Data2DisparityModel


class Tartanair2DisparityModel(Data2DisparityModel):
    def __init__(
        self,
        train_envs: List[str] = ["abandonedfactory"],
        val_envs: List[str] = ["abandonedfactory_night"],
        args: Namespace = None,
        **kwargs
    ):
        super().__init__(args=args)
        self.val_names = ["Tartanair"]
        default_cfg = {"train_envs": train_envs, "val_envs": val_envs}
        self.config = {**default_cfg, **self.config}
        self.train_envs = self.config["train_envs"]
        self.val_envs = self.config["val_envs"]

    def _setup_train_dataset(self):
        self.train_dataset = TartanairDataset(
            envs=self.train_envs,
            cameras=["left", "right"],
            labels=["disp"],
            transform_fn=lambda f: self.train_transform(Tartanair2DisparityModel.adapt(f)),
            ignore_errors=True,
        )

    def _setup_val_dataset(self):
        self.val_dataset = TartanairDataset(
            envs=self.val_envs,
            cameras=["left", "right"],
            labels=["disp"],
            transform_fn=lambda f: self.val_transform(Tartanair2DisparityModel.adapt(f)),
            ignore_errors=True,
        )

    @classmethod
    def adapt(cls, frames):
        for key in ["left", "right"]:
            frames[key] = frames[key][0]
        return frames

    @staticmethod
    def add_argparse_args(parent_parser, inherit_args=True):
        if inherit_args:
            parent_parser = super(Tartanair2DisparityModel, Tartanair2DisparityModel).add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("Tartain2DisparityModel")
        parser.add_argument("--train_envs", type=str, nargs="+", default=["neighborhood"])
        parser.add_argument("--val_envs", type=str, nargs="+", default=["office"])
        return parent_parser

    def train_dataloader(self):
        sampler = SequentialSampler if self.sequential else RandomSampler
        return self.train_dataset.train_loader(
            batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler
        )

    def val_dataloader(self):
        return self.val_dataset.train_loader(batch_size=1, num_workers=self.num_workers, sampler=SequentialSampler)
