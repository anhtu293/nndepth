import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class UpsamplerBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = True):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_bn (bool, optional): Whether to use batch normalization. Defaults to True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(in_channels)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, feat: torch.Tensor, skip_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            feat (torch.Tensor): Feature tensor of shape (B, C, H, W) from the decoder.
            skip_feat (torch.Tensor, optional): Skip feature tensor of shape (B, C, H, W) from the encoder.
                Defaults to None.

        Returns:
            torch.Tensor: Upsampled feature tensor of shape (B, C, H, W).
        """

        output = feat
        if skip_feat is not None:
            skip_feat = self.activation(self.bn1(self.conv1(skip_feat)))
            output = output + skip_feat
        output = self.activation(self.bn2(self.conv2(output)))

        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)
        output = self.activation(self.out_conv(output))
        return output
