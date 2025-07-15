import torch
import torch.nn as nn
from typing import Optional

from nndepth.decoders import BaseDecoder
from nndepth.encoders import MobilenetV3LargeEncoder

from .base import BaseDepthModel


class MobileNetV3DepthModel(BaseDepthModel):
    def __init__(self, feature_channels: int = 64, weights: Optional[str] = None, strict_load: bool = True, **kwargs):
        self.feature_channels = feature_channels
        self.weights = weights
        self.strict_load = strict_load
        super().__init__(**kwargs)

        if self.weights is not None:
            self.load_weights(self.weights, self.strict_load)

    def build_encoder(self) -> nn.Module:
        return MobilenetV3LargeEncoder(
            pretrained=True,
            feature_hooks=[1, 2, 4, 5],
        )

    def build_decoder(self) -> nn.Module:
        return BaseDecoder(
            in_channels=[24, 40, 112, 160],
            out_channels=[self.feature_channels] * 4,
        )

    def build_last_conv(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=3, padding=1, stride=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.feature_channels, 1, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
        )

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        return self.encoder(x)

    def forward_decoder(self, feats: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        return self.decoder(feats)
