import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from nndepth.extractors.mobilenetv3_encoder import MobilenetV3LargeEncoder
from nndepth.blocks.update_block import BasicUpdateBlock
from nndepth.models.igev_stereo.cost_volume import (
    GeometryAwareCostVolume,
    CostVolumeFilterNetwork,
)
from nndepth.utils import load_weights


class IGEVStereoBase(nn.Module):
    """IGEV Stereo: https://arxiv.org/pdf/2303.06615.pdf"""

    SUPPORTED_UPDATE_CLS = {"basic_update_block": BasicUpdateBlock}

    def __init__(
        self,
        update_cls: str = "basic_update_block",
        cv_groups: int = 8,
        iters: int = 12,
        hidden_dim: int = 128,
        context_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        tracing: bool = False,
        include_preprocessing: bool = False,
        weights: Optional[str] = None,
        strict_load: bool = True,
    ):
        """
        Initialize the IGEVStereoBase model.

        Args:
            update_cls (str): The class name of the update block to use. Default is "basic_update_block".
            cv_groups (int): The number of groups to split the cost volume into. Default is 8.
            hidden_dim (int): The hidden dimension of the update block. Default is 128.
            context_dim (int): The context dimension of the update block. Default is 128.
            corr_levels (int): The number of correlation levels to compute. Default is 4.
            corr_radius (int): The radius of the correlation window. Default is 4.
            tracing (bool): Whether to enable tracing for ONNX exportation. Default is False.
            include_preprocessing (bool): Whether to include preprocessing steps in tracing. Default is False.
        """
        super(IGEVStereoBase, self).__init__()
        self.fnet = self._init_fnet()
        self.iters = iters
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.cv_groups = cv_groups
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius

        self.update_block = self.SUPPORTED_UPDATE_CLS[update_cls](
            hidden_dim=self.hidden_dim,
            context_dim=self.context_dim,
            flow_channel=1,
            cor_planes=corr_levels * (corr_radius * 2 + 1) * self.cv_groups * 2,
            spatial_scale=4,
        )
        self.cv_regularizer = self._init_cost_volume_filter()
        self.corr_fn = GeometryAwareCostVolume
        self.cv_squeezer = nn.Conv3d(self.cv_groups, 1, 3, 1, 1)

        # onnx exportation argument
        self.tracing = tracing
        self.include_preprocessing = include_preprocessing

        self.weights = weights
        self.strict_load = strict_load

    def _init_fnet(self):
        raise NotImplementedError("Must be implemented in child class")

    def _init_cost_volume_filter(self):
        raise NotImplementedError("Must be implemented in child class")

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def disp_init(self, fmap: torch.Tensor):
        N, C, H, W = fmap.shape
        _x = torch.zeros([N, 1, H, W], dtype=torch.float32)
        _y = torch.zeros([N, 1, H, W], dtype=torch.float32)
        zero_flow = torch.cat((_x, _y), dim=1).to(fmap.device)
        return zero_flow

    def regress_disparity(self, distribution: torch.Tensor, width: int):
        disp = torch.arange(0, width, dtype=distribution.dtype, device=distribution.device)
        disp = disp.reshape(1, -1, 1, 1)
        return -torch.sum(disp * distribution, dim=1, keepdim=True)

    def initialize_coords(self, fmap1):
        B, C, H, W = fmap1.shape
        coords = torch.arange(W, device=fmap1.device).float()
        coords = coords[None, None, None, :].repeat(B, 1, H, 1)
        return coords

    def convex_upsample(self, flow, mask, rate=4):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        # print(flow.shape, mask.shape, rate)
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(rate * flow, [3, 3], padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, rate * H, rate * W)

    def forward_fnet(self, frame1: torch.Tensor, frame2: torch.Tensor):
        """Forward in backbone. This method must return fmap1, fmap2, cnet1 and guide_features"""
        raise NotImplementedError("Must be implemented in child class")

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor, **kwargs):
        # forward backbone. This method must return fmap1, fmap2, cnet and guide_features(needed to guide cost volume)
        fmap1, fmap2, cnet1, guide_features = self.forward_fnet(frame1, frame2)
        fnet_ds = frame1.shape[-1] // fmap1.shape[-1]
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        C = cnet1.shape[1]
        net, inp = torch.split(cnet1, C // 2, dim=1)
        net = torch.tanh(net)
        inp = F.relu(inp)

        corr = self.corr_fn(
            fmap1,
            fmap2,
            guide_features,
            self.cv_regularizer,
            self.corr_levels,
            self.corr_radius,
            self.cv_groups,
        )
        B, _, H1, W1 = fmap1.shape
        W2 = fmap2.shape[-1]
        geo_aware_cv = corr.geo_aware_cv[0].reshape(B, self.cv_groups, H1, W1, W2).permute(0, 1, 4, 2, 3)
        dist_disparity = F.softmax(self.cv_squeezer(geo_aware_cv).squeeze(1), dim=1)
        init_disparity = self.regress_disparity(dist_disparity, fmap1.shape[-1])

        coords1 = self.initialize_coords(fmap1)
        coords1 = coords1 + init_disparity

        m_outputs = []
        for _ in range(self.iters):
            coords1 = coords1.detach()
            sampled_corr = corr(coords1)
            net, mask, delta_disp = self.update_block(net, inp, sampled_corr, coords1)
            coords1 = coords1 + delta_disp
            up_disp = self.convex_upsample(coords1, mask, rate=fnet_ds)
            m_outputs.append({"up_disp": up_disp})

        return m_outputs


class IGEVStereoMBNet(IGEVStereoBase):
    """IGEV Stereo with MobileNetV3 Large as backbone"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fnet_proj = nn.Sequential(
            nn.Conv2d(24, self.hidden_dim * 2, 3, 1, 1),
            nn.ReLU(False),
        )
        self.cnet_proj = nn.Sequential(
            nn.Conv2d(24, self.context_dim * 2, 3, 1, 1),
            nn.ReLU(False),
        )
        if self.weights is not None:
            load_weights(self, weights=self.weights, strict_load=self.strict_load)

    def _init_fnet(self):
        return MobilenetV3LargeEncoder(pretrained=True, feature_hooks=[1, 2, 3, 4, 5])

    def _init_cost_volume_filter(self):
        return CostVolumeFilterNetwork(self.cv_groups, [40, 80, 160])

    def forward_fnet(self, frame1: torch.Tensor, frame2: torch.Tensor):
        B = frame1.shape[0]
        frames = torch.cat([frame1, frame2], dim=0)
        features = self.fnet(frames)
        fmaps = features[0]
        cnet1 = self.cnet_proj(torch.split(fmaps.clone(), B, dim=0)[0])
        fmaps = self.fnet_proj(fmaps)
        fmap1, fmap2 = torch.split(fmaps, B, dim=0)

        guide_features = [features[1], features[2], features[4]]
        guide_features = [torch.split(f, B, dim=0)[0] for f in guide_features]
        return fmap1, fmap2, cnet1, guide_features
