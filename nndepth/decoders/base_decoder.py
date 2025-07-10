import torch
import torch.nn as nn
from typing import List

from nndepth.blocks import UpsamplerBlock


class BaseDecoder(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: List[int] | int):
        super().__init__()
        if isinstance(out_channels, int):
            out_channels = [out_channels] * len(in_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.skip_layers = nn.ModuleList()
        for in_channel, out_channel in zip(in_channels, out_channels):
            self.skip_layers.append(nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
            ))

        self.upsampler_layers = nn.ModuleList()
        for channel in out_channels:
            self.upsampler_layers.append(UpsamplerBlock(channel, channel, use_bn=True))

    def forward(self, feats: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        skip_feats = []
        for i, feat in enumerate(feats):
            skip_feats.append(self.skip_layers[i](feat))

        output = skip_feats[-1]
        output = self.upsampler_layers[-1](output)

        for i in range(len(skip_feats) - 2, -1, -1):
            output = self.upsampler_layers[i](output, skip_feats[i])

        return output
