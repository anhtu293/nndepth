import os
import torch
from torch.utils.data import DataLoader

from nndepth.data.dataloaders import TartanairDisparityDataLoader, Kitti2015DisparityDataLoader
from nndepth.data.datasets import TartanairDataset, KittiStereo2015
from nndepth.scene import Frame, Depth, Disparity


def test_tartanair():
    datadir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(datadir, "../../samples/tartanair")
    dataset = TartanairDataset(
        dataset_dir=datadir,
        sequence_size=1,
        sequence_skip=0,
        labels=["depth", "disparity"],
        cameras=["left", "right"],
        envs=["abandonedfactory"],
        sequences=[["P000"]],
    )
    frame = dataset[0]
    assert isinstance(frame, dict) and "left" in frame and "right" in frame
    assert isinstance(frame["left"], list) and isinstance(frame["left"][0], Frame)
    assert isinstance(frame["right"], list) and isinstance(frame["right"][0], Frame)
    assert isinstance(frame["left"][0].data, torch.Tensor)
    assert isinstance(frame["left"][0].disparity, Disparity)
    assert isinstance(frame["left"][0].depth, Depth)
    assert isinstance(frame["right"][0].data, torch.Tensor)


def test_kitti2015():
    datadir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(datadir, "../../samples/kitti-stereo-2015")
    dataset = KittiStereo2015(
        dataset_dir=datadir,
        subset="train",
        sequence_start=10,
        sequence_end=10,
        cameras=["left", "right"],
        labels=["disp_occ"]
    )
    frame = dataset[0]
    assert isinstance(frame, dict) and "left" in frame and "right" in frame
    assert isinstance(frame["left"], list) and isinstance(frame["left"][0], Frame)
    assert isinstance(frame["right"], list) and isinstance(frame["right"][0], Frame)
    assert isinstance(frame["left"][0].data, torch.Tensor)
    assert isinstance(frame["left"][0].disparity, Disparity)
    assert isinstance(frame["right"][0].data, torch.Tensor)


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
