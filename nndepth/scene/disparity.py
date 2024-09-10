import torch
from torch.nn.functional import interpolate, max_pool2d
from tensordict import tensorclass
import numpy as np
import matplotlib
import matplotlib.cm
from typing import Optional, Tuple, Literal, Union, List


def maxpool_disp(
    disp: torch.Tensor,
    size: Tuple[int, int],
    disp_sign: str,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply maxpool to disparity

    Parameters
    ----------
    disp : torch.Tensor
        Input disparity tensor
    size : Tuple[int, int]
        Target size (H, W) for the output disparity
    disp_sign : str
        disp_sign of the disparity, either "positive" or "negative"
    **kwargs : dict
        Additional keyword arguments for interpolation

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - new_disp: The resized disparity tensor
        - indices: The indices of the maximum values (useful for resizing occlusion mask)

    Notes
    -----
    This function first applies max pooling to the absolute value of the disparity,
    then adjusts the disp_sign if necessary, and finally interpolates if the resulting
    size doesn't match the target size.
    """
    current_HW = disp.shape[-2:]
    kernel = (current_HW[0] // size[0], current_HW[1] // size[1])
    new_disp, indices = max_pool2d(disp.abs(), kernel_size=kernel, return_indices=True)
    if disp_sign == "negative":
        new_disp *= -1
    if new_disp.shape[-2] != size[0] or new_disp.shape[-1] != size[1]:
        new_disp = interpolate(new_disp, size=size, mode="bilinear", **kwargs)
    return new_disp, indices


def minpool_disp(disp: torch.Tensor, size: Tuple[int, int], disp_sign: str, **kwargs) -> torch.Tensor:
    """Apply minpool to disparity

    Parameters
    ----------
    disp : torch.Tensor
        Input disparity tensor
    size : Tuple[int, int]
        Target size (H, W) for the output disparity
    disp_sign : str
        disp_sign of the disparity, either "positive" or "negative"
    **kwargs : dict
        Additional keyword arguments for interpolation

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - new_disp: The resized disparity tensor
        - indices: The indices of the maximum values (useful for resizing occlusion mask)

    Notes
    -----
    This function first applies min pooling to the absolute value of the disparity,
    then adjusts the sign if necessary, and finally interpolates if the resulting
    size doesn't match the target size.
    """
    current_HW = disp.shape[-2:]
    kernel = (current_HW[0] // size[0], current_HW[1] // size[1])
    new_disp, indices = -max_pool2d(-disp.abs(), kernel_size=kernel, return_indices=True)
    if disp_sign == "negative":
        new_disp *= -1
    if new_disp.shape[-2] != size[0] or new_disp.shape[-1] != size[1]:
        new_disp = interpolate(new_disp, size=size, mode="bilinear", **kwargs)
    return new_disp, indices


@tensorclass
class Disparity:
    data: torch.Tensor
    disp_sign: Literal["negative", "positive"] = "negative"
    occlusion: Optional[torch.Tensor] = None

    def resize(self, size: Tuple[int, int], method: str = "interpolate", **resize_kwargs):
        """Resize disparity

        Parameters
        ----------
        size : tuple of float
            target size (H, W) in relative coordinates between 0 and 1
        method: str
            Method to resize disparity. Choices: `interpolate`, `maxpool`, `minpool`.
            `Interpolate`: A pixel in resized disparity will be an interpolation of its neighbor pixels. This method
                will create artifact in resized disparity map.
            `maxpool`: A pixel in resized disparity will be the max value of its neighbor pixels. This means that the
                method will prefer points whose depth is small (absolute of disparity is large).
                This technique reduce the artifacts. Only available for downsampling.
            `minpool`: A pixel in resized disparity will be the min value of its neighbor pixels. This means that the
                method will prefer points whose depth is large (absolute of disparity is small).
                This technique reduce the artifacts. Only available for downsampling.
        resize_kwargs: Additional arguments for resize methods

        Returns
        -------
        disp_resized : Disparity
            resized version of disparity map
        """
        assert method in ["interpolate", "maxpool", "minpool"], (
            "method must be in [`interpolate`, `maxpool`, `minpool`]"
        )
        # resize disparity
        W_old = self.data.shape[-1]

        occlusion = None
        if method == "interpolate":
            if (len(self.batch_size) == 0):
                data = interpolate(self.data[None], size, mode="bilinear", **resize_kwargs)[0]
                if self.occlusion is not None:
                    occlusion = interpolate(self.occlusion[None], size, mode="bilinear", **resize_kwargs)[0]
            else:
                data = interpolate(self.data, size, mode="bilinear", **resize_kwargs)
                if self.occlusion is not None:
                    occlusion = interpolate(self.occlusion, size, mode="bilinear", **resize_kwargs)
        elif method in ["maxpool", "minpool"] :
            if method == "maxpool":
                data, indices = maxpool_disp(self.data, size, self.disp_sign, **resize_kwargs)
            elif method == "minpool":
                data, indices = minpool_disp(self.data, size, self.disp_sign, **resize_kwargs)
            if self.occlusion is not None:
                occlusion = self.occlusion.flatten(-2)
                occlusion = occlusion[indices.flatten(-2)]
                occlusion = occlusion.reshape(data.shape)

        # rescale disparity
        W_new = data.shape[-1]
        data = data * W_new / W_old
        new_disp = Disparity(
            data=data,
            disp_sign=self.disp_sign,
            occlusion=occlusion,
        )
        return new_disp

    def get_view(
            self, min=None, max=None, cmap="nipy_spectral", reverse=False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get a colored visualization of the disparity map.

        Parameters
        ----------
        min : float, optional
            Minimum absolute disparity value for normalization. If None, uses the minimum value in the data.
        max : float, optional
            Maximum absolute disparity value for normalization. If None, uses the maximum value in the data.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap to use for visualization. Default is "nipy_spectral".
            If "red2green", a custom colormap from red to white to green is used.
        reverse : bool, optional
            If True, reverses the colormap. Default is False.

        Returns
        -------
        Union[np.ndarray, List[np.ndarray]]
            The colored visualization of the disparity map or a list of visualization in case of batch disparity
        """
        if cmap == "red2green":
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)
        elif isinstance(cmap, str):
            cmap = matplotlib.cm.get_cmap(cmap)
        if len(self.batch_size) == 0:
            disp = self.data.abs()
            disp[self.occlusion == 1] = 0
            disp = disp.permute([1, 2, 0]).numpy()
            disp = matplotlib.colors.Normalize(vmin=min, vmax=max, clip=True)(disp)
            if reverse:
                disp = 1 - disp
            disp_color = cmap(disp)[:, :, 0, :3]
            disp_color = (disp_color * 255).astype(np.uint8)
        else:
            disp_color = []
            for i in range(self.batch_size[0]):
                disp = self.data[i].abs()
                disp[self.occlusion[i] == 1] = 0
                disp = disp.permute([1, 2, 0]).numpy()
                disp = matplotlib.colors.Normalize(vmin=min, vmax=max, clip=True)(disp)
                if reverse:
                    disp = 1 - disp
                disp_color.append((cmap(disp)[:, :, 0, :3] * 255).astype(np.uint8))
        return disp_color
