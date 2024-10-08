import torch
import torch.nn as nn
from typing import List, Tuple, Union

from nndepth.blocks.residual_block import ResidualBlock


class BasicEncoder(nn.Module):
    def __init__(self, output_dim: int = 128, norm_fn: str = "batch", dropout: float = 0.0):
        """Basic Encoder
        Args:
            output_dim (int): Number of channels of output feature. Default: 128.
            norm_fn (str): Type of normalization layer. Possible values: [`batch`, `group`, `instance`, `none`].
                Default: `batch`
            dropout (float): Dropout rate. Default: 0.0
        """
        super(BasicEncoder, self).__init__()
        assert norm_fn in [
            "batch",
            "group",
            "instance",
            "none",
        ], f"norm_fn must be in [`batch`, `group`, `instance`, `none`]. Found {norm_fn}"

        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64, affine=False)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Module:
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]]) -> torch.Tensor:
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, x.shape[0] // 2, dim=0)

        return x
