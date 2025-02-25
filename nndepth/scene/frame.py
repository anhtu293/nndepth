import torch
from torch.nn.functional import interpolate
from typing import Optional, Tuple

from .disparity import Disparity
from .depth import Depth
from .camera import Camera


class Frame:
    def __init__(
        self,
        data: torch.Tensor,
        disparity: Optional[Disparity] = None,
        depth: Optional[Depth] = None,
        camera: Optional[Camera] = None,
        camera_id: Optional[str] = None,
        pose: Optional[torch.Tensor] = None,
    ):
        """
        A class representing a single frame in a scene.

        Attributes:
            data (torch.Tensor): Image data tensor
            disparity (Optional[Disparity]): Disparity map for the frame
            depth (Optional[Depth]): Depth map for the frame
            camera (Optional[Camera]): Camera parameters (intrinsic, extrinsic)
            camera_id (Optional[str]): Identifier for the camera
            pose (Optional[torch.Tensor]): 4x4 pose matrix in world coordinates

        Methods:
            resize(size: Tuple[int, int], align_corners=True,
                    disparity_resize_method: str = "interpolate",
                    depth_resize_method: str = "interpolate",
                    disparity_resize_kwargs: dict = {},
                    depth_resize_kwargs: dict = {}) -> Frame:
                Resizes the frame's image, disparity, and depth data.
        """
        self.data = data
        self.disparity = disparity
        self.depth = depth
        self.camera = camera
        self.camera_id = camera_id
        self.pose = pose

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
        ndim = self.data.ndim
        assert ndim <= 4, "Only support resize tensor with 3 or 4 dimensions"

        missing_dim = 4 - ndim
        for _ in range(missing_dim):
            self.data = self.data[None]

        # Resize image
        resized_img = interpolate(self.data, size, mode="bilinear", align_corners=align_corners)
        for _ in range(missing_dim):
            resized_img = resized_img[0]

        # Resize disparity
        if self.disparity is not None:
            disparity = self.disparity.resize(size, disparity_resize_method, **disparity_resize_kwargs)
        else:
            disparity = None

        # Resize depth
        if self.depth is not None:
            depth = self.depth.resize(size, depth_resize_method, **depth_resize_kwargs)
        else:
            depth = None

        if self.camera is not None:
            new_camera = self.camera.resize(size)
        else:
            new_camera = None

        new_frame = Frame(
            data=resized_img,
            disparity=disparity,
            depth=depth,
            camera=new_camera,
            pose=self.pose,
        )
        return new_frame
