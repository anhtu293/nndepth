from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from nndepth.blocks.gru import SepConvGRU, ConvGRU


class FlowHead(nn.Module):
    """Ref: https://github.com/princeton-vl/RAFT/blob/master/core/update.py"""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, flow_channel: int = 2):
        """
        Initialize the FlowHead module.

        Args:
            input_dim (int): The number of input channels.
            hidden_dim (int): The number of hidden channels.
            flow_channel (int): The number of flow channels.
        """
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, flow_channel, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FlowHead module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv2(self.relu(self.conv1(x)))


class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes: int, hidden_dim: int = 128, flow_channel: int = 2):
        """
        Initialize the BasicMotionEncoder module.

        Args:
            cor_planes (int): The number of input channels for correlation.
            hidden_dim (int): The number of hidden channels.
            flow_channel (int): The number of flow channels.
        """
        super(BasicMotionEncoder, self).__init__()

        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(flow_channel, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, hidden_dim - flow_channel, 3, padding=1)

    def forward(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cor_planes: int,
        context_dim: int = 128,
        gru: str = "sep_conv",
        flow_channel: int = 2,
        spatial_scale: Union[Tuple[int, int], int] = 8,
    ):
        """
        Initialize the BasicUpdateBlock module.

        Args:
            hidden_dim (int): The number of hidden channels.
            cor_planes (int): The number of input channels for correlation.
            context_dim (int): The number of input channels for the GRU.
            gru (str): The type of GRU to use. Options are "sep_conv" or "conv_gru".
            flow_channel (int): The number of flow channels.
            spatial_scale (Union[Tuple[int, int], int]): The spatial scale of the mask.
        """
        super(BasicUpdateBlock, self).__init__()
        GRU_CLS = {"sep_conv": SepConvGRU, "conv_gru": ConvGRU}

        self.encoder = BasicMotionEncoder(cor_planes, hidden_dim=hidden_dim, flow_channel=flow_channel)
        self.gru = GRU_CLS[gru](hidden_dim=hidden_dim, input_dim=context_dim + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=hidden_dim, flow_channel=flow_channel)

        sps = spatial_scale**2 if isinstance(spatial_scale, int) else spatial_scale[0] * spatial_scale[1]
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, sps * 9, 1, padding=0),
        )

    def forward(self, net: torch.Tensor, inp: torch.Tensor, corr: torch.Tensor, flow: torch.Tensor):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat((inp, motion_features), dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow
