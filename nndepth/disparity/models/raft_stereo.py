import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Tuple

import aloscene

from nndepth.blocks.conv import MobileOneBlock, FeatureFusionBlock
from nndepth.extractors.rep_vit import RepViT
from nndepth.extractors.basic_encoder import BasicEncoder
from nndepth.blocks.update_block import BasicUpdateBlock
from nndepth.disparity.models.cost_volume.raft_stereo import CorrBlock1D, GroupCorrBlock1D


class RAFTStereo(nn.Module):
    """RAFT-Stereo: https://arxiv.org/pdf/2109.07547.pdf
    """

    def __init__(
            self,
            hidden_dim=128,
            context_dim=128,
            corr_levels=4,
            corr_radius=4,
            tracing=False,
            include_preprocessing=False,
            **kwargs
    ):
        super().__init__()
        self.fnet = self._init_fnet(**kwargs)
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius

        self.update_block = self._init_update_block()
        self.corr_fn = CorrBlock1D

        # onnx exportation argument
        self.tracing = tracing
        self.include_preprocessing = include_preprocessing

    def _init_fnet(self):
        raise NotImplementedError("Must be implemented in child class")

    def _init_update_block(self):
        raise NotImplementedError("Must be implemented in child class")

    def _preprocess_input(self, frame1: aloscene.Frame, frame2: aloscene.Frame):
        if self.tracing:
            assert isinstance(frame1, torch.Tensor)
            assert isinstance(frame2, torch.Tensor)
            if self.include_preprocessing:
                assert (frame1.ndim == 3) and (frame2.ndim == 3)
                frame1 = frame1.permute(2, 0, 1)
                frame2 = frame2.permute(2, 0, 1)

                frame1 = frame1.unsqueeze(0)
                frame2 = frame2.unsqueeze(0)

                frame1 = frame1 / 255 * 2 - 1
                frame2 = frame2 / 255 * 2 - 1

            assert (frame1.ndim == 4) and (frame2.ndim == 4)
        else:
            for frame in [frame1, frame2]:
                assert frame.normalization == "minmax_sym"
                assert frame.names == ("B", "C", "H", "W")
            frame1 = frame1.as_tensor()
            frame2 = frame2.as_tensor()
        return frame1, frame2

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

    @torch.no_grad()
    def inference(self, m_outputs: Dict[str, torch.Tensor], only_last=False):
        def generate_frame(out_dict):
            return aloscene.Disparity(
                out_dict["up_flow"], names=("B", "C", "H", "W"), camera_side="left", disp_format="signed"
            )

        if only_last:
            return generate_frame(m_outputs[-1])
        else:
            return [generate_frame(out_dict) for out_dict in m_outputs]

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

        up_flow = F.unfold(rate * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, rate * H, rate * W)

    def forward_fnet(self, frame1: torch.Tensor, frame2: torch.Tensor):
        """Forward in backbone. This method must return fmap1, fmap2, cnet1 and guide_features
        """
        raise NotImplementedError("Must be implemented in child class")

    def forward(
        self,
        frame1: aloscene.Frame,
        frame2: aloscene.Frame,
        iters=12,
        **kwargs
    ):
        frame1, frame2 = self._preprocess_input(frame1, frame2)

        # forward backbone. This method must return fmap1, fmap2, cnet
        fmap1, fmap2, cnet1 = self.forward_fnet(frame1, frame2)
        fnet_ds = frame1.shape[-1] // fmap1.shape[-1]

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        C = cnet1.shape[1]
        net, inp = torch.split(cnet1, C // 2, dim=1)
        net = torch.tanh(net)
        inp = F.relu(inp)

        corr = self.corr_fn(fmap1, fmap2, self.corr_levels, self.corr_radius)

        coords1 = self.initialize_coords(fmap1)
        org_coords = self.initialize_coords(fmap1)

        m_outputs = []
        for _ in range(iters):
            coords1 = coords1.detach()
            sampled_corr = corr(coords1)
            net, mask, delta_disp = self.update_block(net, inp, sampled_corr, coords1 - org_coords)
            coords1 = coords1 + delta_disp
            disp = coords1 - org_coords
            up_disp = self.convex_upsample(disp, mask, rate=fnet_ds)
            m_outputs.append({"up_flow": up_disp})

        return m_outputs


class BaseRAFTStereo(RAFTStereo):
    """Original RAFT Stereo presented in the paper
    """
    def _init_fnet(self):
        return BasicEncoder()

    def _init_update_block(self):
        return BasicUpdateBlock(
            hidden_dim=self.hidden_dim,
            cor_planes=self.corr_levels * (self.corr_radius * 2 + 1),
            flow_channel=1,
            context_dim=self.context_dim,
            spatial_scale=8
        )

    def forward_fnet(self, frame1: torch.Tensor, frame2: torch.Tensor):
        fmap1, fmap2 = self.fnet([frame1, frame2])
        cnet = fmap1.clone()
        return fmap1, fmap2, cnet


class Corse2FineGroupRepViTRAFTStereo(RAFTStereo):
    def __init__(self, num_groups: int = 4, **kwargs):
        self.num_groups = num_groups
        super().__init__(**kwargs)
        assert self.corr_levels == 1, "Corr level must be 1 in Corse2FineGroupRepViTRaftStereo"
        self.corr_fn = GroupCorrBlock1D
        self.cnet_proj = nn.ModuleList([
            MobileOneBlock(256, self.context_dim * 2, kernel_size=1, stride=1, padding=0),
            MobileOneBlock(64, self.context_dim * 2, kernel_size=1, stride=1, padding=0),
            MobileOneBlock(64, self.context_dim * 2, kernel_size=1, stride=1, padding=0),
        ])
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(256, 64, 64, 1, 0),
            FeatureFusionBlock(64, 16, 64, 1, 0),
        ])

    def _init_fnet(self, **kwargs):
        return RepViT(**kwargs)

    def _init_update_block(self):
        return BasicUpdateBlock(
            hidden_dim=self.hidden_dim,
            cor_planes=self.num_groups * self.corr_levels * (self.corr_radius * 2 + 1),
            flow_channel=1,
            context_dim=self.context_dim,
            gru="conv_gru",
            spatial_scale=(4, 4)
        )

    def convex_upsample(self, flow, mask, rate: Union[Tuple[int, int], int] = (4, 4)):
        """Upsample flow field [H/N, W/M, 1] -> [H, W, 1] using convex combination"""
        if isinstance(rate, int):
            rate = (rate, rate)
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate[0], rate[1], H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate[1] * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, rate[0] * H, rate[1] * W)

    def forward(
        self,
        frame1: aloscene.Frame,
        frame2: aloscene.Frame,
        iters=12,
        **kwargs
    ):
        B = frame1.shape[0]
        frame1, frame2 = self._preprocess_input(frame1, frame2)

        features = self.fnet(torch.cat([frame1, frame2], axis=0))[::2]
        features = features[::-1]
        init_coords = self.initialize_coords(torch.split(features[0], [B, B], dim=0)[0])
        org_coords = self.initialize_coords(torch.split(features[0], [B, B], dim=0)[0])

        previous_feat = None
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

            m_outputs = []
            for _ in range(iters):
                coords1 = coords1.detach()
                sampled_corr = corr(coords1)
                net, mask, delta_disp = self.update_block(net, inp, sampled_corr, coords1 - org_coords)
                coords1 = coords1 + delta_disp
                disp = coords1 - org_coords
                up_disp = self.convex_upsample(disp, mask, rate=(4, 4))
                m_outputs.append({"up_flow": up_disp})
            if idx < len(features) - 1:
                org_coords = org_coords = self.initialize_coords(torch.split(features[idx + 1], [B, B], dim=0)[0])
                init_coords = org_coords + up_disp.detach()
                previous_feat = feat

        return m_outputs
