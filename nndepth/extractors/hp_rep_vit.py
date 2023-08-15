import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

from nndepth.blocks.rep_vit import RepViTBlock


class HPNet(nn.Module):
    """Horizontal Preserved Network
    """
    BASE_NUM_CHANNELS = [32, 64, 128, 256]

    def __init__(self, num_blocks_per_stage=[1, 2, 6, 4], width_multipliers=[1, 1, 1, 2]):
        super().__init__()
        assert len(num_blocks_per_stage) == len(width_multipliers)
        self.num_blocks_per_stage = num_blocks_per_stage
        self.width_multipliers = width_multipliers

        self.stem_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
        )

        self.stem_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
        )

        self.num_channels = 16
        self.stage_0 = self._make_stage(
            channels=int(self.BASE_NUM_CHANNELS[0] * width_multipliers[0]),
            num_blocks=num_blocks_per_stage[0],
            num_se_blocks=0,
            kernel_size=1,
            stride=(1, 1),
            padding=0,
            dw_kernel_size=3,
            dw_padding=1,
        )
        self.stage_1 = self._make_stage(
            channels=int(self.BASE_NUM_CHANNELS[1] * width_multipliers[1]),
            num_blocks=num_blocks_per_stage[1],
            stride=(2, 1),
            num_se_blocks=0,
            kernel_size=1,
            padding=0,
            dw_kernel_size=3,
            dw_padding=1,
        )
        self.stage_2 = self._make_stage(
            channels=int(self.BASE_NUM_CHANNELS[2] * width_multipliers[2]),
            num_blocks=num_blocks_per_stage[2],
            stride=(2, 1),
            num_se_blocks=num_blocks_per_stage[2],
            exp_ratio=2.0,
            kernel_size=1,
            padding=0,
            dw_kernel_size=3,
            dw_padding=1,
        )
        self.stage_3 = self._make_stage(
            channels=int(self.BASE_NUM_CHANNELS[3] * width_multipliers[3]),
            num_blocks=num_blocks_per_stage[3],
            stride=(2, 1),
            num_se_blocks=num_blocks_per_stage[3],
            exp_ratio=2.0,
            kernel_size=1,
            padding=0,
            dw_kernel_size=3,
            dw_padding=1,
        )

    def _make_stage(
        self,
        channels: int,
        num_blocks: int,
        stride: Tuple[int, int],
        num_se_blocks: int,
        exp_ratio: float = 1.0,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dw_kernel_size: Union[int, Tuple[int, int]] = 3,
        dw_padding: Union[int, Tuple[int, int]] = 1,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                RepViTBlock(
                    in_channels=self.num_channels,
                    out_channels=channels,
                    exp_ratio=exp_ratio if i % 2 == 0 else 1.0,
                    kernel_size=kernel_size,
                    padding=padding,
                    dw_kernel_size=dw_kernel_size,
                    dw_padding=dw_padding,
                    stride=strides[i],
                    se_block=i >= (num_blocks - num_se_blocks),
                )
            )
            self.num_channels = channels
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        features = []
        for layer in [self.stem_1, self.stem_2, self.stage_0, self.stage_1, self.stage_2, self.stage_3]:
            x = layer(x)
            features.append(x)

        return features


if __name__ == "__main__":
    model = HPNet()
    x = torch.randn(1, 3, 480, 640)
    y = model(x)
    print([t.shape for t in y[1:]])
