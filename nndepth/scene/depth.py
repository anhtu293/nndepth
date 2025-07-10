import torch
from torch.nn.functional import interpolate, max_pool2d
import numpy as np
import matplotlib
import matplotlib.cm
from typing import Optional, Tuple, Union, List


def maxpool_depth(depth: torch.Tensor, size: Tuple[int, int], **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply maxpool to depth

    Args:
        depth (torch.Tensor): Input depth tensor
        size (Tuple[int, int]): Target size (H, W) for the output depth
        **kwargs (dict): Additional keyword arguments for interpolation

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - new_depth (torch.Tensor): The resized depth tensor
            - indices (torch.Tensor): The indices of the maximum values (useful for resizing occlusion mask)

    Notes:
    This function first applies max pooling to the absolute value of the disparity,
    then adjusts the disp_sign if necessary, and finally interpolates if the resulting
    size doesn't match the target size.
    """
    assert depth.ndim == 4, "Only support resize depth with 4 dimensions"

    current_HW = depth.shape[-2:]
    kernel = (current_HW[0] // size[0], current_HW[1] // size[1])
    new_depth, indices = max_pool2d(depth.abs(), kernel_size=kernel, return_indices=True)
    if new_depth.shape[-2] != size[0] or new_depth.shape[-1] != size[1]:
        new_depth = interpolate(new_depth, size=size, mode="bilinear", **kwargs)
    return new_depth, indices


def minpool_depth(depth: torch.Tensor, size: Tuple[int, int], **kwargs) -> torch.Tensor:
    """Apply minpool to depth

    Args:
        depth (torch.Tensor): Input depth tensor
        size (Tuple[int, int]): Target size (H, W) for the output depth
        **kwargs (dict): Additional keyword arguments for interpolation

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - new_depth (torch.Tensor): The resized depth tensor
            - indices (torch.Tensor): The indices of the maximum values (useful for resizing occlusion mask)

    Notes:
    This function first applies min pooling to the absolute value of the depth,
    then adjusts the sign if necessary, and finally interpolates if the resulting
    size doesn't match the target size.
    """
    assert depth.ndim == 4, "Only support resize depth with 4 dimensions"

    current_HW = depth.shape[-2:]
    kernel = (current_HW[0] // size[0], current_HW[1] // size[1])
    new_depth, indices = max_pool2d(-depth.abs(), kernel_size=kernel, return_indices=True)
    new_depth = -new_depth
    if new_depth.shape[-2] != size[0] or new_depth.shape[-1] != size[1]:
        new_depth = interpolate(new_depth, size=size, mode="bilinear", **kwargs)
    return new_depth, indices


class Depth:
    def __init__(self, data: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, is_inverse: bool = False):
        """
        A class representing depth information for a frame in a scene.

        Attributes:
            data (torch.Tensor): The depth data as a tensor.
            valid_mask (Optional[torch.Tensor]): A mask indicating valid depth values.
            is_inverse (bool): Whether the depth is inverse.

        Methods:
            resize(size: Tuple[int, int], method: str = "interpolate", **resize_kwargs) -> Depth:
                Resizes the depth data and valid mask (if present).

        Notes:
            The depth values are typically in meters, representing the distance from the camera to the object in the
            scene.
            The valid_mask, if provided, indicates which depth values are considered valid or reliable.
        """
        self.data = data
        self.valid_mask = valid_mask
        self.is_inverse = is_inverse

    def resize(self, size: Tuple[int, int], method: str = "interpolate", **resize_kwargs):
        """Resize depth

        Args:
            size (Tuple[int, int]): target size (H, W) in relative coordinates between 0 and 1
            method (str): Method to resize depth. Choices: `interpolate`, `maxpool`, `minpool`.
                `Interpolate`: A pixel in resized depth will be an interpolation of its neighbor pixels. This method
                will create artifact in resized depth map.
            `maxpool`: A pixel in resized depth will be the max value of its neighbor pixels.
                This technique reduce the artifacts. Only available for downsampling.
            `minpool`: A pixel in resized depth will be the min value of its neighbor pixels.
                This technique reduce the artifacts. Only available for downsampling.
            resize_kwargs (dict): Additional arguments for resize methods

        Returns:
            Depth: resized version of depth map
        """
        assert method in ["interpolate", "maxpool", "minpool"], (
            "method must be in [`interpolate`, `maxpool`, `minpool`]"
        )
        assert self.data.ndim <= 4, "Only support resize depth with 3 or 4 dimensions"
        if self.valid_mask is not None:
            assert self.valid_mask.ndim <= 4, "Only support resize valid_mask with 3 or 4 dimensions"

        missing_dim = 4 - self.data.ndim
        for _ in range(missing_dim):
            self.data = self.data[None]

        if self.valid_mask is not None:
            missing_dim = 4 - self.valid_mask.ndim
            for _ in range(missing_dim):
                self.valid_mask = self.valid_mask[None]

        valid_mask = None
        if method == "interpolate":
            data = interpolate(self.data, size, mode="bilinear", **resize_kwargs)
            if self.valid_mask is not None:
                valid_mask = interpolate(
                    self.valid_mask.float(),
                    size,
                    mode="bilinear",
                    **resize_kwargs).type(self.valid_mask.dtype)
        elif method in ["maxpool", "minpool"]:
            if method == "maxpool":
                data, indices = maxpool_depth(self.data, size, **resize_kwargs)
            elif method == "minpool":
                data, indices = minpool_depth(self.data, size, **resize_kwargs)
            if self.valid_mask is not None:
                valid_mask = self.valid_mask.flatten()
                valid_mask = valid_mask[indices.flatten()]
                valid_mask = valid_mask.reshape(data.shape)

        for _ in range(missing_dim):
            data = data[0]

        if valid_mask is not None:
            for _ in range(missing_dim):
                valid_mask = valid_mask[0]

        new_depth = Depth(
            data=data,
            valid_mask=valid_mask,
        )
        return new_depth

    def inverse(self, clip_max: float = None, clip_min: float = None, eps: float = 1e-6) -> "Depth":
        """Inverse the depth

        Args:
            clip_max (float): Maximum depth value to clip after inverse.
            clip_min (float): Minimum depth value to clip after inverse.
            eps (float): Small value to avoid division by zero.

        Returns:
            Depth: Inversed depth
        """

        data = 1 / (self.data + eps)
        if clip_max is not None:
            data = torch.clamp(data, max=clip_max)
        if clip_min is not None:
            data = torch.clamp(data, min=clip_min)
        new_depth = Depth(
            data=data,
            valid_mask=self.valid_mask.clone() if self.valid_mask is not None else None,
            is_inverse=not self.is_inverse,
        )
        return new_depth

    def get_view(
            self, min=None, max=None, cmap="nipy_spectral", reverse=False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get a colored visualization of the depth map.

        Args:
            min (float): Minimum absolute depth value for normalization. If None, uses the minimum value in the data.
            max (float): Maximum absolute depth value for normalization. If None, uses the maximum value in the data.
            cmap (str or matplotlib.colors.Colormap): Colormap to use for visualization. Default is "nipy_spectral".
            If "red2green", a custom colormap from red to white to green is used.
            reverse (bool): If True, reverses the colormap. Default is False.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The colored visualization of the depth map or a list of visualization
            in case of batch depth
        """
        assert self.data.ndim <= 4, "Only support get_view for depth with 3 or 4 dimensions"

        if cmap == "red2green":
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)
        elif isinstance(cmap, str):
            cmap = matplotlib.cm.get_cmap(cmap)
        if self.data.ndim == 3:
            depth = self.data.clone()
            if self.valid_mask is not None:
                depth[self.valid_mask.float() != 1] = depth[self.valid_mask.float() == 1].min()
            depth = depth.permute([1, 2, 0]).cpu().numpy()
            depth = matplotlib.colors.Normalize(vmin=min, vmax=max, clip=True)(depth)
            if reverse:
                depth = 1 - depth
            depth_color = cmap(depth)[:, :, 0, :3]
            depth_color = (depth_color * 255).astype(np.uint8)
        else:
            depth_color = []
            for i in range(self.data.shape[0]):
                depth = self.data[i].clone()
                if self.valid_mask is not None:
                    depth[self.valid_mask[i].float() != 1] = depth[self.valid_mask[i].float() == 1].min()
                depth = depth.permute([1, 2, 0]).cpu().numpy()
                depth = matplotlib.colors.Normalize(vmin=min, vmax=max, clip=True)(depth)
                if reverse:
                    depth = 1 - depth
                depth_color.append((cmap(depth)[:, :, 0, :3] * 255).astype(np.uint8))
        return depth_color
