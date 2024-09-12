import torch
from nndepth.disparity.models import BaseRAFTStereo, CREStereoBase, Coarse2FineGroupRepViTRAFTStereo, IGEVStereoMBNet


def test_BaseRAFTStereo():
    model = BaseRAFTStereo()
    left = torch.rand((1, 3, 480, 640))
    right = torch.rand((1, 3, 480, 640))
    outputs = model(left, right)
    assert isinstance(outputs, list)


def test_Coarse2FineGroupRepViTRAFTStereo():
    model = Coarse2FineGroupRepViTRAFTStereo(corr_levels=1)
    left = torch.rand((1, 3, 384, 512))
    right = torch.rand((1, 3, 384, 512))
    outputs = model(left, right)
    assert isinstance(outputs, list)


def test_CREStereoBase():
    model = CREStereoBase()
    left = torch.rand((1, 3, 480, 640))
    right = torch.rand((1, 3, 480, 640))
    outputs = model(left, right)
    assert isinstance(outputs, list)


def test_IGEVStereoMBNet():
    model = IGEVStereoMBNet()
    left = torch.rand((1, 3, 480, 640))
    right = torch.rand((1, 3, 480, 640))
    outputs = model(left, right)
    assert isinstance(outputs, list)
