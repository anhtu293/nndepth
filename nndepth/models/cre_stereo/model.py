import torch
import torch.nn as nn
import torch.nn.functional as F

from nndepth.blocks.update_block import BasicUpdateBlock
from nndepth.extractors.basic_encoder import BasicEncoder
from nndepth.models.cre_stereo.cost_volume import AGCL

from nndepth.blocks.pos_enc import PositionEncodingSine
from nndepth.blocks.transformer import LocalFeatureTransformer

from nndepth.utils import load_weights


class CREStereoBase(nn.Module):
    """CreStereo: https://arxiv.org/abs/2203.11483"""

    SUPPORTED_FNET_CLS = {"basic_encoder": {"cls": BasicEncoder, "downsample": 8}}
    SUPPORTED_UPDATE_CLS = {"basic_update_block": BasicUpdateBlock}

    def __init__(
        self,
        fnet_cls: str = "basic_encoder",
        update_cls: str = "basic_update_block",
        iters: int = 12,
        max_disp: int = 192,
        num_fnet_channels: int = 256,
        hidden_dim: int = 128,
        context_dim: int = 128,
        search_num: int = 9,
        mixed_precision: bool = False,
        test_mode: bool = False,
        tracing: bool = False,
        include_preprocessing: bool = False,
        weights: str = None,
        strict_load: bool = True,
        **kwargs,
    ):
        """
        Initialize the CREStereoBase model.

        Args:
            fnet_cls (str): The class name of the feature extraction network. Default is "basic_encoder".
            update_cls (str): The class name of the update block. Default is "basic_update_block".
            max_disp (int): The maximum disparity value. Default is 192.
            num_fnet_channels (int): The number of channels in the feature extraction network. Default is 256.
            hidden_dim (int): The hidden dimension size. Default is 128.
            context_dim (int): The context dimension size. Default is 128.
            search_num (int): The number of search iterations. Default is 9.
            mixed_precision (bool): Whether to use mixed precision training. Default is False.
            test_mode (bool): Whether to run the model in test mode. Default is False.
            tracing (bool): Whether to enable tracing. Default is False.
            include_preprocessing (bool): Whether to include preprocessing steps. Default is False.
            **kwargs: Additional keyword arguments.
        """
        super(CREStereoBase, self).__init__()
        assert context_dim == hidden_dim, "Context dim must be equal to hidden_dim in this model"

        self.max_flow = max_disp
        self.mixed_precision = mixed_precision
        self.test_mode = test_mode
        self.iters = iters

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.dropout = 0
        self.tracing = tracing
        self.include_preprocessing = include_preprocessing

        self.fnet = self.SUPPORTED_FNET_CLS[fnet_cls]["cls"](
            output_dim=num_fnet_channels, norm_fn="instance", dropout=self.dropout
        )
        self.fnet_ds = self.SUPPORTED_FNET_CLS[fnet_cls]["downsample"]
        self.update_block = self.SUPPORTED_UPDATE_CLS[update_cls](
            hidden_dim=self.hidden_dim, cor_planes=4 * 9, spatial_scale=self.fnet_ds
        )

        # loftr
        self.self_att_fn = LocalFeatureTransformer(
            d_model=num_fnet_channels,
            nhead=8,
            layer_names=["self"] * 1,
            attention="linear",
        )
        self.cross_att_fn = LocalFeatureTransformer(
            d_model=num_fnet_channels,
            nhead=8,
            layer_names=["cross"] * 1,
            attention="linear",
        )

        # adaptive search
        self.search_num = search_num
        self.conv_offset_16 = nn.Conv2d(num_fnet_channels, self.search_num * 2, kernel_size=3, stride=1, padding=1)
        self.conv_offset_8 = nn.Conv2d(num_fnet_channels, self.search_num * 2, kernel_size=3, stride=1, padding=1)
        self.range_16 = 1
        self.range_8 = 1

        # load weights
        self.weights = weights
        self.strict_load = strict_load
        if weights is not None:
            load_weights(self, weights=weights, strict_load=strict_load)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def convex_upsample(self, flow, mask, rate=4):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        # print(flow.shape, mask.shape, rate)
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(rate * flow, [3, 3], padding=1)
        up_disp = up_disp.view(N, 2, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 2, rate * H, rate * W)

    def zero_init(self, fmap):
        N, C, H, W = fmap.shape
        _x = torch.zeros([N, 1, H, W], dtype=torch.float32)
        _y = torch.zeros([N, 1, H, W], dtype=torch.float32)
        zero_flow = torch.cat((_x, _y), dim=1).to(fmap.device)
        return zero_flow

    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        flow_init=None,
        upsample=True,
        test_mode=False,
        **kwargs,
    ):
        """Estimate optical flow between pair of frames"""
        frame1 = frame1.contiguous()
        frame2 = frame2.contiguous()

        hdim = self.hidden_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([frame1, frame2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # 1/fnet_ds -> 1/(fnet_ds * 2)
        # feature
        fmap1_dw8 = F.avg_pool2d(fmap1, 2, stride=2)
        fmap2_dw8 = F.avg_pool2d(fmap2, 2, stride=2)

        # offset
        offset_dw8 = self.conv_offset_8(fmap1_dw8)
        offset_dw8 = self.range_8 * (torch.sigmoid(offset_dw8) - 0.5) * 2.0

        # context
        net, inp = torch.split(fmap1, [hdim, hdim], dim=1)
        net = torch.tanh(net)
        inp = F.relu(inp)
        net_dw8 = F.avg_pool2d(net, 2, stride=2)
        inp_dw8 = F.avg_pool2d(inp, 2, stride=2)

        # 1/fnet_ds -> 1/(fnet_ds * 4)
        # feature
        fmap1_dw16 = F.avg_pool2d(fmap1, 4, stride=4)
        fmap2_dw16 = F.avg_pool2d(fmap2, 4, stride=4)
        offset_dw16 = self.conv_offset_16(fmap1_dw16)
        offset_dw16 = self.range_16 * (torch.sigmoid(offset_dw16) - 0.5) * 2.0

        # context
        net_dw16 = F.avg_pool2d(net, 4, stride=4)
        inp_dw16 = F.avg_pool2d(inp, 4, stride=4)

        # positional encoding and self-attention
        pos_encoding_fn_small = PositionEncodingSine(
            d_model=256, max_shape=(frame1.shape[2] // (self.fnet_ds * 4), frame1.shape[3] // (self.fnet_ds * 4))
        )

        # 'n c h w -> n (h w) c'
        x_tmp = pos_encoding_fn_small(fmap1_dw16)
        fmap1_dw16 = x_tmp.permute(0, 2, 3, 1).reshape(x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3], x_tmp.shape[1])

        # 'n c h w -> n (h w) c'
        x_tmp = pos_encoding_fn_small(fmap2_dw16)
        fmap2_dw16 = x_tmp.permute(0, 2, 3, 1).reshape(x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3], x_tmp.shape[1])

        fmap1_dw16, fmap2_dw16 = self.self_att_fn(fmap1_dw16, fmap2_dw16)
        fmap1_dw16, fmap2_dw16 = [
            x.reshape(x.shape[0], frame1.shape[2] // (self.fnet_ds * 4), -1, x.shape[2]).permute(0, 3, 1, 2)
            for x in [fmap1_dw16, fmap2_dw16]
        ]

        corr_fn = AGCL(fmap1, fmap2)
        corr_fn_dw8 = AGCL(fmap1_dw8, fmap2_dw8)
        corr_fn_att_dw16 = AGCL(fmap1_dw16, fmap2_dw16, att=self.cross_att_fn)

        # Cascaded refinement (1/(fnet_ds*4) + 1/(fnet_ds*2) + 1/fnet_ds)
        m_outputs = []
        flow = None
        if flow_init is not None:
            scale = fmap1.shape[2] / flow_init.shape[2]
            flow = -scale * F.interpolate(
                flow_init,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        else:
            # zero initialization
            flow_dw16 = self.zero_init(fmap1_dw16)

            # Recurrent Update Module
            # RUM: 1/(fnet_ds * 4)
            for itr in range(self.iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                flow_dw16 = flow_dw16.detach()
                out_corrs = corr_fn_att_dw16(flow_dw16, offset_dw16, small_patch=small_patch)

                net_dw16, up_mask, delta_flow = self.update_block(net_dw16, inp_dw16, out_corrs, flow_dw16)

                flow_dw16 = flow_dw16 + delta_flow
                flow = self.convex_upsample(flow_dw16, up_mask, rate=self.fnet_ds)

                m_outputs.append({"up_disp": flow})

            scale = fmap1_dw8.shape[2] / flow.shape[2]
            flow_dw8 = scale * F.interpolate(
                flow,
                size=(fmap1_dw8.shape[2], fmap1_dw8.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

            # RUM: 1/(fnet_ds * 2)
            for itr in range(self.iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                flow_dw8 = flow_dw8.detach()
                out_corrs = corr_fn_dw8(flow_dw8, offset_dw8, small_patch=small_patch)

                net_dw8, up_mask, delta_flow = self.update_block(net_dw8, inp_dw8, out_corrs, flow_dw8)

                flow_dw8 = flow_dw8 + delta_flow
                flow = self.convex_upsample(flow_dw8, up_mask, rate=self.fnet_ds)
                m_outputs.append({"up_disp": flow})

            scale = fmap1.shape[2] / flow.shape[2]
            flow = scale * F.interpolate(
                flow,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

        # RUM: 1/self.fnet_ds
        for itr in range(self.iters):
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True

            flow = flow.detach()
            out_corrs = corr_fn(flow, None, small_patch=small_patch, iter_mode=True)

            net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = self.convex_upsample(flow, up_mask, rate=self.fnet_ds)

            # predictions.append(flow_up)
            m_outputs.append({"up_disp": flow_up})

        if self.test_mode:
            return flow_up

        return m_outputs
