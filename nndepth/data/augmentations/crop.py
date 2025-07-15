import torch
from typing import Tuple

from nndepth.scene import Frame

from .base import BaseAugmentation


class RandomCrop(BaseAugmentation):
    def __init__(self, size: Tuple[int, int], p: float = 0.5):
        super().__init__(p)
        self.size = size

    def apply(self, frame: Frame) -> Frame:
        top = torch.randint(0, frame.data.shape[-2] - self.size[0] + 1, (1,))
        left = torch.randint(0, frame.data.shape[-1] - self.size[1] + 1, (1,))
        frame.data = frame.data[:, top:top + self.size[0], left:left + self.size[1]]
        if frame.disparity is not None:
            frame.disparity.data = frame.disparity.data[:, top:top + self.size[0], left:left + self.size[1]]
            if frame.disparity.occlusion is not None:
                frame.disparity.occlusion = frame.disparity.occlusion[
                    :,
                    top:top + self.size[0],
                    left:left + self.size[1],
                ]
        if frame.depth is not None:
            frame.depth.data = frame.depth.data[:, top:top + self.size[0], left:left + self.size[1]]
            frame.depth.valid_mask = frame.depth.valid_mask[:, top:top + self.size[0], left:left + self.size[1]]
        return frame


class RandomResizedCrop(BaseAugmentation):
    def __init__(self, size: Tuple[int, int], p: float = 0.5):
        super().__init__(p)
        self.size = size

    def apply(self, frame: Frame) -> Frame:
        if self.size[0] < frame.data.shape[-2]:
            crop_H = torch.randint(self.size[0], frame.data.shape[-2] + 1, (1,))
        else:
            crop_H = frame.data.shape[-2]
        if self.size[1] < frame.data.shape[-1]:
            crop_W = torch.randint(self.size[1], frame.data.shape[-1] + 1, (1,))
        else:
            crop_W = frame.data.shape[-1]
        top = torch.randint(0, frame.data.shape[-2] - crop_H + 1, (1,))
        left = torch.randint(0, frame.data.shape[-1] - crop_W + 1, (1,))
        frame.data = frame.data[:, top:top + crop_H, left:left + crop_W]
        if frame.disparity is not None:
            frame.disparity.data = frame.disparity.data[:, top:top + crop_H, left:left + crop_W]
            if frame.disparity.occlusion is not None:
                frame.disparity.occlusion = frame.disparity.occlusion[:, top:top + crop_H, left:left + crop_W]
        if frame.depth is not None:
            frame.depth.data = frame.depth.data[:, top:top + crop_H, left:left + crop_W]
            frame.depth.valid_mask = frame.depth.valid_mask[:, top:top + crop_H, left:left + crop_W]
        frame = frame.resize(self.size)
        return frame
