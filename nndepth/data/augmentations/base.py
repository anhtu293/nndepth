from abc import ABC, abstractmethod
import torch

from nndepth.scene import Frame


class BaseAugmentation(ABC):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, frame: Frame):
        if torch.rand(1) < self.p:
            return self.apply(frame)
        return frame

    @abstractmethod
    def apply(self, frame: Frame) -> Frame:
        pass
