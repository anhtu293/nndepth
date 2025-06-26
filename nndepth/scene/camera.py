import torch
from typing import Optional, Union, Tuple


class Camera:
    def __init__(self, intrinsic: Optional[torch.Tensor] = None, extrinsic: Optional[torch.Tensor] = None):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic

    def resize(self, size: Union[int, Tuple[int, int]], method: str = 'bilinear', **kwargs) -> 'Camera':
        if self.intrinsic is not None:
            resize_ratio_w = size[1]
            resize_ratio_h = size[0]
            cam_intrinsic = self.intrinsic.clone()
            cam_intrinsic[..., 0, 0] *= resize_ratio_w  # fx
            cam_intrinsic[..., 1, 1] *= resize_ratio_h  # fy
            cam_intrinsic[..., 0, 2] *= resize_ratio_w  # x0
            cam_intrinsic[..., 1, 2] *= resize_ratio_h  # y0
            return Camera(cam_intrinsic, self.extrinsic)
        else:
            return self
