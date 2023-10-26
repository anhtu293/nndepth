import torch
import torch.nn as nn
from typing import Union, List

from timm.models.mobilenetv3 import tf_mobilenetv3_large_100


class MobilenetV3LargeEncoder(nn.Module):
    def __init__(self, pretrained=True, feature_hooks: Union[None, List[int]] = None):
        """MobilenetV3 Large
        Args:
            pretrained (bool): Use pretrained weight of timm. Default: True
            feature_hooks (None or List[int]): indices of backbone stages where we want to hook feature map.
                Default: None.
        """
        super().__init__()
        self.backbone = tf_mobilenetv3_large_100(pretrained=pretrained, features_only=True)
        self.feature_hooks = feature_hooks

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        features = []
        for i, b in enumerate(self.backbone.blocks):
            x = b(x)
            if self.feature_hooks is not None and i in self.feature_hooks:
                features.append(x)
        if self.feature_hooks is None:
            return features[-1]
        else:
            return features
