from torch.utils.data import SequentialSampler, RandomSampler
import pytorch_lightning as pl

import alonet


class Data2DisparityModel(pl.LightningDataModule):
    def __init__(self, args, **kwargs):
        alonet.common.pl_helpers.params_update(self, args, kwargs)
        self.batch_size = args.batch_size
        self.train_on_val = args.train_on_val
        self.num_workers = args.num_workers
        self.sequential = args.sequential_sampler
        self.sequence_size = max(1, args.sequence_size)
        self.HW = tuple(args.HW)
        self.args = args
        super().__init__()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Data2DisparityModel")
        parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
        parser.add_argument("--train_on_val", action="store_true")
        parser.add_argument("--num_workers", type=int, default=8, help="num_workers to use on the dataset")
        parser.add_argument("--sequential_sampler", action="store_true", help="sample data sequentially (no shuffle)")
        parser.add_argument("--sequence_size", default=0, type=int, help="Size of the desired sequence")
        parser.add_argument("--HW", type=int, default=[368, 496], nargs=2, help="Size H W of resized frame")

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
