import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List

from nndepth.blocks.conv import MobileOneBlock, FeatureFusionBlock
from nndepth.extractors.rep_vit import RepViT
from nndepth.extractors.basic_encoder import BasicEncoder
from nndepth.blocks.update_block import BasicUpdateBlock
from nndepth.models.raft_stereo.cost_volume import (
    CorrBlock1D,
    GroupCorrBlock1D,
)
from nndepth.utils import load_weights, BaseModel


class RAFTStereo(BaseModel):
    """RAFT-Stereo: https://arxiv.org/pdf/2109.07547.pdf"""

    def __init__(
        self,
        iters: int = 12,
        fnet_dim: int = 256,
        hidden_dim: int = 128,
        context_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        tracing: bool = False,
        include_preprocessing: bool = False,
        weights: str = None,
        strict_load: bool = True,
        **kwargs
    ):
        """Initialize the RAFTStereo model.

        Args:
            hidden_dim (int): The hidden dimension. Default is 128.
            context_dim (int): The context dimension. Default is 128.
            corr_levels (int): The number of correlation levels. Default is 4.
            corr_radius (int): The correlation radius. Default is 4.
            tracing (bool): Whether to enable tracing for ONNX exportation. Default is False.
            include_preprocessing (bool): Whether to include preprocessing steps. Default is False.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.iters = iters
        self.fnet_dim = fnet_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.fnet = self._init_fnet(**kwargs)
        self.cnet_proj = nn.Sequential(
            nn.Conv2d(fnet_dim, self.context_dim + self.hidden_dim, kernel_size=3, padding=1), nn.ReLU(False)
        )

        self.update_block = self._init_update_block()
        self.corr_fn = CorrBlock1D

        # onnx exportation argument
        self.tracing = tracing
        self.include_preprocessing = include_preprocessing
        # load weights
        self.weights = weights
        self.strict_load = strict_load
        if weights is not None:
            load_weights(self, weights=weights, strict_load=strict_load)

    def _init_fnet(self):
        raise NotImplementedError("Must be implemented in child class")

    def _init_update_block(self):
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

    def initialize_coords(self, fmap1):
        B, C, H, W = fmap1.shape
        coords = torch.arange(W, device=fmap1.device).float()
        coords = coords[None, None, None, :].repeat(B, 1, H, 1)
        return coords

    def convex_upsample(self, flow, mask, rate=8):
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
        # forward backbone. This method must return fmap1, fmap2, cnet
        fmap1, fmap2, cnet1 = self.forward_fnet(frame1, frame2)
        fnet_ds = frame1.shape[-1] // fmap1.shape[-1]

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # C = cnet1.shape[1]
        net, inp = torch.split(cnet1, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = F.relu(inp)

        corr = self.corr_fn(fmap1, fmap2, self.corr_levels, self.corr_radius)

        coords1 = self.initialize_coords(fmap1)
        org_coords = self.initialize_coords(fmap1)

        m_outputs = []
        for _ in range(self.iters):
            coords1 = coords1.detach()
            sampled_corr = corr(coords1)
            net, mask, delta_disp = self.update_block(net, inp, sampled_corr, coords1 - org_coords)
            coords1 = coords1 + delta_disp
            disp = coords1 - org_coords
            up_disp = self.convex_upsample(disp, mask, rate=fnet_ds)
            m_outputs.append({"up_disp": up_disp})

        return m_outputs


class BaseRAFTStereo(RAFTStereo):
    """Original RAFT Stereo presented in the paper"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_fnet(self):
        return BasicEncoder(output_dim=self.fnet_dim)

    def _init_update_block(self):
        return BasicUpdateBlock(
            hidden_dim=self.hidden_dim,
            cor_planes=self.corr_levels * (self.corr_radius * 2 + 1),
            flow_channel=1,
            context_dim=self.context_dim,
            spatial_scale=8,
        )

    def forward_fnet(self, frame1: torch.Tensor, frame2: torch.Tensor):
        fmap1, fmap2 = self.fnet([frame1, frame2])
        cnet = self.cnet_proj(fmap1)
        return fmap1, fmap2, cnet


class Coarse2FineGroupRepViTRAFTStereo(RAFTStereo):
    def __init__(
        self,
        num_groups: int = 4,
        downsample_ratios: List[Tuple[int, int]] = [[2, 2], [2, 2], [2, 2], [2, 2]],
        ffn_exp_ratios: List[float] = [1.0, 3.0, 3.0, 4.0],
        num_blocks_per_stage: List[int] = [4, 4, 6, 2],
        patch_size: int = 7,
        stem_strides: List[int] = [[2, 2], [2, 2], [1, 1]],
        token_mixer_types: List[str] = ["repmixer", "repmixer", "repmixer", "attention"],
        use_ffn_per_stage: List[bool] = [False, True, True, True],
        width_multipliers: List[float] = [1.0, 1.0, 1.0, 1.0],
        weights: str = None,
        strict_load: bool = True,
        **kwargs
    ):
        """
        Coarse2FineGroupRepViTRAFTStereo

        Args:
            num_groups (int): Number of groups. Default is 4.
            downsample_ratios (List[Tuple[int, int]]): Downsample ratios. Default is [[2, 2], [2, 2], [2, 2], [2, 2]].
            ffn_exp_ratios (List[float]): Feed-forward expansion ratios. Default is [1.0, 3.0, 3.0, 4.0].
            num_blocks_per_stage (List[int]): Number of blocks per stage. Default is [4, 4, 6, 2].
            patch_size (int): Patch size. Default is 7.
            stem_strides (List[int]): Stem strides. Default is [[2, 2], [2, 2], [1, 1]].
            token_mixer_types (List[str]): Token mixer types.
                Default is ["repmixer", "repmixer", "repmixer", "attention"].
            use_ffn_per_stage (List[bool]): Use feed-forward network per stage. Default is [False, True, True, True].
            width_multipliers (List[float]): Width multipliers. Default is [1.0, 1.0, 1.0, 1.0].
            weights (str): Weights. Default is None.
            strict_load (bool): Strict load. Default is True.
            **kwargs: Additional keyword arguments.

            For `num_groups`, `downsample_ratios`, `ffn_exp_ratios`, `num_blocks_per_stage`, `stem_strides`,
                `token_mixer_types`, `use_ffn_per_stage`, `width_multipliers`,
                Please refer to `nndepth.extractors.rep_vit.RepViT` for detail implementation.
        """
        self.num_groups = num_groups
        self.downsample_ratios = downsample_ratios
        self.ffn_exp_ratios = ffn_exp_ratios
        self.num_blocks_per_stage = num_blocks_per_stage
        self.patch_size = patch_size
        self.stem_strides = stem_strides
        self.token_mixer_types = token_mixer_types
        self.use_ffn_per_stage = use_ffn_per_stage
        self.width_multipliers = width_multipliers

        super().__init__(**kwargs)
        assert self.corr_levels == 1, "Corr level must be 1 in Coarse2FineGroupRepViTRaftStereo"
        self.corr_fn = GroupCorrBlock1D
        self.cnet_proj = nn.ModuleList(
            [
                MobileOneBlock(256, self.context_dim * 2, kernel_size=1, stride=1, padding=0),
                MobileOneBlock(64, self.context_dim * 2, kernel_size=1, stride=1, padding=0),
                MobileOneBlock(64, self.context_dim * 2, kernel_size=1, stride=1, padding=0),
            ]
        )
        self.fusion_blocks = nn.ModuleList(
            [
                FeatureFusionBlock(256, 64, 64, 1, 0),
                FeatureFusionBlock(64, 16, 64, 1, 0),
            ]
        )
        self.weights = weights
        self.strict_load = strict_load
        if self.weights is not None:
            load_weights(self, weights=self.weights, strict_load=self.strict_load)

    def _init_fnet(self, **kwargs):
        return RepViT(
            downsample_ratios=self.downsample_ratios,
            ffn_exp_ratios=self.ffn_exp_ratios,
            num_blocks_per_stage=self.num_blocks_per_stage,
            patch_size=self.patch_size,
            stem_strides=self.stem_strides,
            token_mixer_types=self.token_mixer_types,
            use_ffn_per_stage=self.use_ffn_per_stage,
            width_multipliers=self.width_multipliers,
        )

    def _init_update_block(self):
        return BasicUpdateBlock(
            hidden_dim=self.hidden_dim,
            cor_planes=self.num_groups * self.corr_levels * (self.corr_radius * 2 + 1),
            flow_channel=1,
            context_dim=self.context_dim,
            gru="conv_gru",
            spatial_scale=(4, 4),
        )

    def convex_upsample(self, flow, mask, rate: Union[Tuple[int, int], int] = (4, 4)):
        """Upsample flow field [H/N, W/M, 1] -> [H, W, 1] using convex combination"""
        if isinstance(rate, int):
            rate = (rate, rate)
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate[0], rate[1], H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(rate[1] * flow, [3, 3], padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, rate[0] * H, rate[1] * W)

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor, **kwargs):
        B = frame1.shape[0]

        features = self.fnet(torch.cat([frame1, frame2], axis=0))[::2]
        features = features[::-1]
        init_coords = self.initialize_coords(torch.split(features[0], [B, B], dim=0)[0])
        org_coords = self.initialize_coords(torch.split(features[0], [B, B], dim=0)[0])

        previous_feat = None
        m_outputs = []
        for idx, feat in enumerate(features):
            if previous_feat is not None:
                feat = self.fusion_blocks[idx - 1]([previous_feat, feat])
            fmap1, fmap2 = torch.split(feat, [B, B], dim=0)
            cnet = fmap1.clone()
            cnet = self.cnet_proj[idx](cnet)

            fmap1 = fmap1.float()
            fmap2 = fmap2.float()

            C = cnet.shape[1]
            net, inp = torch.split(cnet, C // 2, dim=1)
            net = torch.tanh(net)
            inp = F.relu(inp)

            corr = self.corr_fn(fmap1, fmap2, self.corr_levels, self.corr_radius, self.num_groups)

            coords1 = init_coords.detach()

            for _ in range(self.iters):
                coords1 = coords1.detach()
                sampled_corr = corr(coords1)
                net, mask, delta_disp = self.update_block(net, inp, sampled_corr, coords1 - org_coords)
                coords1 = coords1 + delta_disp
                disp = coords1 - org_coords
                up_disp = self.convex_upsample(disp, mask, rate=(4, 4))

                rate = frame1.shape[-1] / up_disp.shape[-1]
                if rate == 1:
                    disp_at_org_size = up_disp
                else:
                    disp_at_org_size = nn.functional.interpolate(up_disp, size=frame1.shape[-2:]) * rate
                m_outputs.append({"up_disp": disp_at_org_size})
            if idx < len(features) - 1:
                org_coords = self.initialize_coords(torch.split(features[idx + 1], [B, B], dim=0)[0])
                init_coords = org_coords + up_disp.detach()
                previous_feat = feat

        return m_outputs
