import os
from torch.utils.data import DataLoader

from nndepth.disparity.data_loaders import TartanairDisparityDataLoader, Kitti2015DisparityDataLoader


def test_TartanairDisparityDataLoader():
    datadir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(datadir, "../../samples/tartanair")
    dataloader = TartanairDisparityDataLoader(
        dataset_dir=datadir,
        HW=[480, 640],
        train_envs=["abandonedfactory"],
        val_envs=["abandonedfactory"],
        batch_size=1,
        num_workers=1,
    )
    dataloader.setup()
    assert dataloader.train_dataloader is not None and dataloader.val_dataloader is not None
    assert isinstance(dataloader.train_dataloader, DataLoader) and isinstance(dataloader.val_dataloader, DataLoader)


def test_Kitti2015DisparityDataLoader():
    datadir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(datadir, "../../samples/kitti-stereo-2015")
    dataloader = Kitti2015DisparityDataLoader(
        dataset_dir=datadir,
        HW=[375, 1242],
        batch_size=1,
        num_workers=1,
    )
    dataloader.setup()
    assert dataloader.train_dataloader is not None and dataloader.val_dataloader is not None
    assert isinstance(dataloader.train_dataloader, DataLoader) and isinstance(dataloader.val_dataloader, DataLoader)
