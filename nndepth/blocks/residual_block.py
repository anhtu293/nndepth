import torch
import torch.nn as nn


# Ref: https://github.com/princeton-vl/RAFT/blob/master/core/extractor.py
class ResidualBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, norm_fn: str = "group", stride: int = 1):
        """
        Initialize a ResidualBlock.

        Args:
            in_planes (int): Number of input channels.
            planes (int): Number of output channels.
            norm_fn (str): Normalization function to use. Options are "group", "batch", "instance", or "none".
            stride (int): Stride value for the convolutional layers.
        """
        assert norm_fn in [
            "group",
            "batch",
            "isntance",
            "none",
        ], f"norm_fn must be in group, batch, instance, or none, found {norm_fn}"
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes, affine=False)
            self.norm2 = nn.InstanceNorm2d(planes, affine=False)
            self.norm3 = nn.InstanceNorm2d(planes, affine=False)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        x = self.downsample(x)

        return self.relu(x + y)
