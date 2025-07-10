import torch

from nndepth.scene import Frame

from .base import BaseAugmentation


class RandomHorizontalFlip(BaseAugmentation):
    def apply(self, frame: Frame) -> Frame:
        frame.data = torch.flip(frame.data, dims=[-1])
        if frame.disparity is not None:
            frame.disparity.data = torch.flip(frame.disparity.data, dims=[-1])
            frame.disparity.disp_sign = "positive" if frame.disparity.disp_sign == "negative" else "negative"
            if frame.disparity.occlusion is not None:
                frame.disparity.occlusion = torch.flip(frame.disparity.occlusion, dims=[-1])
        if frame.depth is not None:
            frame.depth.data = torch.flip(frame.depth.data, dims=[-1])
            frame.depth.valid_mask = torch.flip(frame.depth.valid_mask, dims=[-1])
        return frame


class RandomVerticalFlip(BaseAugmentation):
    def apply(self, frame: Frame) -> Frame:
        frame.data = torch.flip(frame.data, dims=[-2])
        if frame.depth is not None:
            frame.depth.data = torch.flip(frame.depth.data, dims=[-2])
            frame.depth.valid_mask = torch.flip(frame.depth.valid_mask, dims=[-2])
        return frame
