import torch
from torch.nn.functional import interpolate, max_pool2d
from tensordict import tensorclass
import numpy as np
import matplotlib
import matplotlib.cm
from typing import Optional, Tuple, Union, List


def maxpool_depth(depth: torch.Tensor, size: Tuple[int, int], **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply maxpool to depth

    Parameters
    ----------
    depth : torch.Tensor
        Input depth tensor
    size : Tuple[int, int]
        Target size (H, W) for the output depth
    **kwargs : dict
        Additional keyword arguments for interpolation

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - new_depth: The resized depth tensor
        - indices: The indices of the maximum values (useful for resizing occlusion mask)

    Notes
    -----
    This function first applies max pooling to the absolute value of the disparity,
    then adjusts the disp_sign if necessary, and finally interpolates if the resulting
    size doesn't match the target size.
    """
    current_HW = depth.shape[-2:]
    kernel = (current_HW[0] // size[0], current_HW[1] // size[1])
    new_depth, indices = max_pool2d(depth.abs(), kernel_size=kernel, return_indices=True)
    if new_depth.shape[-2] != size[0] or new_depth.shape[-1] != size[1]:
        new_depth = interpolate(new_depth, size=size, mode="bilinear", **kwargs)
    return new_depth, indices


def minpool_depth(depth: torch.Tensor, size: Tuple[int, int], **kwargs) -> torch.Tensor:
    """Apply minpool to depth

    Parameters
    ----------
    depth : torch.Tensor
        Input depth tensor
    size : Tuple[int, int]
        Target size (H, W) for the output depth
    **kwargs : dict
        Additional keyword arguments for interpolation

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - new_depth: The resized depth tensor
        - indices: The indices of the maximum values (useful for resizing occlusion mask)

    Notes
    -----
    This function first applies min pooling to the absolute value of the depth,
    then adjusts the sign if necessary, and finally interpolates if the resulting
    size doesn't match the target size.
    """
    current_HW = depth.shape[-2:]
    kernel = (current_HW[0] // size[0], current_HW[1] // size[1])
    new_depth, indices = -max_pool2d(-depth.abs(), kernel_size=kernel, return_indices=True)
    if new_depth.shape[-2] != size[0] or new_depth.shape[-1] != size[1]:
        new_depth = interpolate(new_depth, size=size, mode="bilinear", **kwargs)
    return new_depth, indices


@tensorclass
class Depth:
    """
    A class representing depth information for a frame in a scene.

    Attributes:
        data (torch.Tensor): The depth data as a tensor.
        valid_mask (Optional[torch.Tensor]): A mask indicating valid depth values.

    Methods:
        resize(size: Tuple[int, int], method: str = "interpolate", **resize_kwargs) -> Depth:
            Resizes the depth data and valid mask (if present).

    Notes:
        The depth values are typically in meters, representing the distance from the camera to the object in the scene.
        The valid_mask, if provided, indicates which depth values are considered valid or reliable.
    """
    data: torch.Tensor
    valid_mask: Optional[torch.Tensor] = None

    def resize(self, size: Tuple[int, int], method: str = "interpolate", **resize_kwargs):
        """Resize depth

        Parameters
        ----------
        size : tuple of float
            target size (H, W) in relative coordinates between 0 and 1
        method: str
            Method to resize depth. Choices: `interpolate`, `maxpool`, `minpool`.
            `Interpolate`: A pixel in resized depth will be an interpolation of its neighbor pixels. This method
                will create artifact in resized depth map.
            `maxpool`: A pixel in resized depth will be the max value of its neighbor pixels.
                This technique reduce the artifacts. Only available for downsampling.
            `minpool`: A pixel in resized depth will be the min value of its neighbor pixels.
                This technique reduce the artifacts. Only available for downsampling.
        resize_kwargs: Additional arguments for resize methods

        Returns
        -------
        disp_resized : depth
            resized version of depth map
        """
        assert method in ["interpolate", "maxpool", "minpool"], (
            "method must be in [`interpolate`, `maxpool`, `minpool`]"
        )

        valid_mask = None
        if method == "interpolate":
            if (len(self.batch_size) == 0):
                data = interpolate(self.data[None], size, mode="bilinear", **resize_kwargs)[0]
                if self.valid_mask is not None:
                    valid_mask = interpolate(self.valid_mask[None], size, mode="bilinear", **resize_kwargs)[0]
            else:
                data = interpolate(self.data, size, mode="bilinear", **resize_kwargs)
                if self.valid_mask is not None:
                    valid_mask = interpolate(self.valid_mask, size, mode="bilinear", **resize_kwargs)
        elif method in ["maxpool", "minpool"] :
            if method == "maxpool":
                data, indices = maxpool_depth(self.data, size, **resize_kwargs)
            elif method == "minpool":
                data, indices = minpool_depth(self.data, size, **resize_kwargs)
            if self.valid_mask is not None:
                valid_mask = self.valid_mask.flatten(-2)
                valid_mask = valid_mask[indices.flatten(-2)]
                valid_mask = valid_mask.reshape(data.shape)

        new_disp = Depth(
            data=data,
            valid_mask=valid_mask,
        )
        return new_disp

    def get_view(
            self, min=None, max=None, cmap="nipy_spectral", reverse=False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get a colored visualization of the depth map.

        Parameters
        ----------
        min_disp : float, optional
            Minimum absolute depth value for normalization. If None, uses the minimum value in the data.
        max_disp : float, optional
            Maximum absolute depth value for normalization. If None, uses the maximum value in the data.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap to use for visualization. Default is "nipy_spectral".
            If "red2green", a custom colormap from red to white to green is used.
        reverse : bool, optional
            If True, reverses the colormap. Default is False.

        Returns
        -------
        Union[np.ndarray, List[np.ndarray]]
            The colored visualization of the depth map or a list of visualization in case of batch depth
        """
        if cmap == "red2green":
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)
        elif isinstance(cmap, str):
            cmap = matplotlib.cm.get_cmap(cmap)
        if len(self.batch_size) == 0:
            depth = self.data.clone()
            if self.valid_mask is not None:
                depth[self.valid_mask != 1] = 0
            depth = depth.permute([1, 2, 0]).numpy()
            depth = matplotlib.colors.Normalize(vmin=min, vmax=max, clip=True)(depth)
            if reverse:
                depth = 1 - depth
            depth_color = cmap(depth)[:, :, 0, :3]
            depth_color = (depth_color * 255).astype(np.uint8)
        else:
            depth_color = []
            for i in range(self.batch_size[0]):
                depth = self.data[i].clone()
                if self.valid_mask is not None:
                    depth[self.valid_mask[i] != 1] = 0
                depth = depth.permute([1, 2, 0]).numpy()
                depth = matplotlib.colors.Normalize(vmin=min, vmax=max, clip=True)(depth)
                if reverse:
                    depth = 1 - depth
                depth_color.append((cmap(depth)[:, :, 0, :3] * 255).astype(np.uint8))
        return depth_color
