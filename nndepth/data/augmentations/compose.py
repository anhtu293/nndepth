from typing import List

from .base import BaseAugmentation

from nndepth.scene import Frame


class Compose(BaseAugmentation):
    def __init__(self, augmentations: List[BaseAugmentation]):
        super().__init__(p=1.0)
        self.augmentations = augmentations

    def apply(self, frame: Frame) -> Frame:
        for augmentation in self.augmentations:
            frame = augmentation(frame)
        return frame
