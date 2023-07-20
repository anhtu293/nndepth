import torch
from torch.utils.data import SequentialSampler, RandomSampler

import aloscene
import alonet

from nndepth.datasets import TartanairDataset
from nndepth.disparity.data_modules.data2disparity import Data2DisparityModel


class Tartanair2DisparityModel(Data2DisparityModel):
    def __init__(self, args, **kwargs):
        alonet.common.pl_helpers.params_update(self, args, kwargs)
        self.val_names = ["Tartanair"]
        self.train_envs = args.train_envs
        self.val_envs = args.val_envs
        super().__init__(args)

    def _setup_train_dataset(self):
        self.train_dataset = TartanairDataset(
            envs=self.train_envs,
            cameras=["left", "right"],
            labels=["disp"],
            sequence_size=self.sequence_size,
            transform_fn=lambda f: self.train_transform(Tartanair2DisparityModel.adapt(f)),
            ignore_errors=True,
        )

    def _setup_val_dataset(self):
        self.val_dataset = TartanairDataset(
            envs=self.val_envs,
            cameras=["left", "right"],
            labels=["disp"],
            sequence_size=self.sequence_size,
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
