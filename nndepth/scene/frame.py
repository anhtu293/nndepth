import torch
from torch.nn.functional import interpolate
from tensordict import tensorclass
from typing import Optional, Tuple

from .disparity import Disparity
from .depth import Depth


@tensorclass
class Frame:
    """
    A class representing a single frame in a scene.

    Attributes:
        image (torch.Tensor): The image data as a tensor.
        disparity (Optional[Disparity]): Disparity information for the frame.
        planar_depth (Optional[Depth]): Depth map for the frame.
        cam_intrinsic (Optional[torch.Tensor]): Camera intrinsic parameters.
        cam_extrinsic (Optional[torch.Tensor]): Camera extrinsic parameters.
        pose (Optional[torch.Tensor]): Pose of the frame.
        baseline (Optional[float]): Baseline distance for stereo setups.
        camera (Optional[str]): Identifier for the camera used.

    Methods:
        resize(size: Tuple[int, int], align_corners=True,
               disparity_resize_method: str = "interpolate",
               depth_resize_method: str = "interpolate",
               disparity_resize_kwargs: dict = {},
               depth_resize_kwargs: dict = {}) -> Frame:
            Resizes the frame's image, disparity, and depth data.
    """
    image: torch.Tensor
    disparity: Optional[Disparity] = None
    planar_depth: Optional[Depth] = None
    cam_intrinsic: Optional[torch.Tensor] = None
    cam_extrinsic: Optional[torch.Tensor] = None
    pose: Optional[torch.Tensor] = None
    baseline: Optional[float] = None
    camera: Optional[str] = None

    def _resize_cam_intrinsic(self, size: Tuple[int, int]) -> torch.Tensor:
        resize_ratio_w = size[1]
        resize_ratio_h = size[0]
        cam_intrinsic = self.cam_intrinsic.clone()
        cam_intrinsic[..., 0, 0] *= resize_ratio_w  # fx
        cam_intrinsic[..., 1, 1] *= resize_ratio_h  # fy
        cam_intrinsic[..., 0, 2] *= resize_ratio_w  # x0
        cam_intrinsic[..., 1, 2] *= resize_ratio_h  # y0
        return cam_intrinsic

    def resize(
        self,
        size: Tuple[int, int],
        align_corners=True,
        disparity_resize_method: str = "interpolate",
        depth_resize_method: str = "interpolate",
        disparity_resize_kwargs: dict = {},
        depth_resize_kwargs: dict = {},
    ):
        # Resize image
        if len(self.batch_size) == 0:
            resized_img = interpolate(self.image[None], size=size, mode="bilinear", align_corners=align_corners)[0]
        else:
            resized_img = interpolate(self.image, size, mode="bilinear", align_corners=align_corners)

        # Resize disparity
        if self.disparity is not None:
            disparity = self.disparity.resize(size, disparity_resize_method, **disparity_resize_kwargs)
        else:
            disparity = None

        # Resize depth
        if self.planar_depth is not None:
            depth = self.planar_depth.resize(size, depth_resize_method, **depth_resize_kwargs)
        else:
            depth = None

        if self.cam_intrinsic is not None:
            cam_intrinsic = self._resize_cam_intrinsic(size)
        else:
            cam_intrinsic = None

        new_frame = Frame(
            image=resized_img,
            disparity=disparity,
            planar_depth=depth,
            cam_intrinsic=cam_intrinsic,
            cam_extrinsic=self.cam_extrinsic,
            baseline=self.baseline,
            camera=self.camera,
            batch_size=self.batch_size
        )
        return new_frame
