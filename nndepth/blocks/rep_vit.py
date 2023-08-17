import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple


class SEBlock(nn.Module):
    """Squeeze and Excite module.

    Pytorch implementation of `Squeeze-and-Excitation Networks` -
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels, out_channels=int(in_channels * rd_ratio), kernel_size=1, stride=1, bias=True
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio), out_channels=in_channels, kernel_size=1, stride=1, bias=True
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class RepViTBlock(nn.Module):
    """RepViT Block
    RepViT: Revisiting Mobile CNN From ViT Perspective: https://arxiv.org/pdf/2307.09283.pdf
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_ratio: float = 1.0,
        act_fn: torch.nn.functional = torch.nn.functional.relu,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dw_kernel_size: Union[int, Tuple[int, int]] = 3,
        dw_padding: Union[int, Tuple[int, int]] = 1,
        stride: Union[int, Tuple[int, int]] = 1,
        se_block: bool = False,
        inference=False,
    ):
        super().__init__()
        mid_channels = int(in_channels * exp_ratio)
        self.act_fn = act_fn
        self.inference = inference

        # Depth-wise convolution
        self.conv_dw = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                dw_kernel_size,
                stride=stride,
                padding=dw_padding,
                groups=in_channels,
            ),
            nn.BatchNorm2d(in_channels),
        )
        self.conv_dw_proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                1,
                stride=stride,
                padding=0,
                groups=in_channels,
            ),
            nn.BatchNorm2d(in_channels),
        )
        self.conv2_dw = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        # Squeeze-and-excitation
        if se_block:
            self.se = SEBlock(out_channels)
        else:
            self.se = None

        # Point-wise expansion
        self.conv_pw1 = nn.Conv2d(out_channels, mid_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn_pw1 = nn.BatchNorm2d(mid_channels)
        # self.act1 = nn.ReLU(inplace=True)

        # Point-wise linear projection
        self.conv_pw2 = nn.Conv2d(mid_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn_pw2 = nn.BatchNorm2d(out_channels)
        self.act_pw2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        # Depth-wise convolution
        x1 = self.conv_dw(x)
        x2 = self.conv_dw_proj(x)
        x = F.relu(x1 + x2)
        x = self.conv2_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # Point-wise expansion
        x_pw = self.conv_pw1(x)
        x_pw = self.bn_pw1(x_pw)
        x_pw = self.act_fn(x_pw)

        # Point-wise linear projection
        x_pw = self.conv_pw2(x_pw)
        x_pw = self.bn_pw2(x_pw)
        x_pw = self.act_pw2(x_pw)

        x = x + x_pw

        return x
