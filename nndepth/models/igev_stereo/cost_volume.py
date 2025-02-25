import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from nndepth.models.igev_stereo.utils import linear_sampler


class GeometryAwareCostVolume(nn.Module):
    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        features: List[torch.Tensor],
        regularizer_3d: nn.Module,
        num_levels: int = 4,
        radius: int = 4,
        num_groups: int = 8,
    ):
        """
        Initializes the GeometryAwareCostVolume module.

        Args:
            fmap1 (torch.Tensor): The feature map from the first image. Shape: (B, C, H, W).
            fmap2 (torch.Tensor): The feature map from the second image. Shape: (B, C, H, W).
            features (List[torch.Tensor]): List of additional feature maps. Each feature map has shape (B, C, H, W).
            regularizer_3d (nn.Module): The 3D regularizer module.
            num_levels (int): The number of levels in the cost volume pyramid. Default: 4.
            radius (int): The radius of the correlation window. Default: 4.
            num_groups (int): The number of groups in the cost volume. Default: 8.
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.radius = radius
        feat_cost_volume = self.build_cost_volume(fmap1, fmap2)  # B, num_groups, H, W, W(~max_disp)
        geo_aware_cost_volume = regularizer_3d(feat_cost_volume.clone().permute(0, 1, 4, 2, 3), features)

        # pooling
        B, N, H1, W1, W2 = feat_cost_volume.shape
        assert N == self.num_groups, "N must be equal to num_groups"
        feat_cost_volume = feat_cost_volume.reshape(B * N * H1 * W1, 1, W2)
        B, N, W2, H1, W1 = geo_aware_cost_volume.shape
        assert N == self.num_groups, "N must be equal to num_groups"
        geo_aware_cost_volume = geo_aware_cost_volume.permute(0, 1, 3, 4, 2).reshape(B * N * H1 * W1, 1, W2)
        self.feat_corr_cv = [feat_cost_volume]
        self.geo_aware_cv = [geo_aware_cost_volume]
        for _ in range(self.num_levels):
            feat_cost_volume = F.avg_pool1d(feat_cost_volume, 2)
            self.feat_corr_cv.append(feat_cost_volume)
            geo_aware_cost_volume = F.avg_pool1d(geo_aware_cost_volume, 2)
            self.geo_aware_cv.append(geo_aware_cost_volume)

    def forward(self, coords: torch.Tensor):
        r = self.radius
        batch, _, h, w = coords.shape
        out_corr = []
        for i in range(self.num_levels):
            feat_corr = self.feat_corr_cv[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            coords1 = coords.permute((0, 2, 3, 1)).unsqueeze(1)
            coords1 = coords1.repeat((1, self.num_groups, 1, 1, 1))
            center = coords1.reshape(batch * self.num_groups * h * w, 1) / 2**i
            dx = dx.reshape(1, 2 * r + 1).to(coords1.device)
            coords_lvl = center + dx
            feat_corr = feat_corr.reshape(batch * self.num_groups * h * w, -1)
            sampled_feat_corr = linear_sampler(feat_corr, coords_lvl)
            sampled_feat_corr = sampled_feat_corr.reshape(batch, self.num_groups, h, w, -1)
            sampled_feat_corr = sampled_feat_corr.permute(0, 2, 3, 1, 4).reshape(batch, h, w, -1)

            geo_aware = self.geo_aware_cv[i]
            geo_aware = geo_aware.reshape(batch * self.num_groups * h * w, -1)
            sampled_geo_aware = linear_sampler(geo_aware, coords_lvl)
            sampled_geo_aware = sampled_geo_aware.reshape(batch, self.num_groups, h, w, -1)
            sampled_geo_aware = sampled_geo_aware.permute(0, 2, 3, 1, 4).reshape(batch, h, w, -1)

            out_corr.extend([sampled_feat_corr, sampled_geo_aware])
        out_corr = torch.cat(out_corr, dim=-1)
        return out_corr.permute(0, 3, 1, 2).contiguous().float()

    def build_cost_volume(self, fmap1: torch.Tensor, fmap2: torch.Tensor):
        assert (
            fmap1.shape[1] % self.num_groups == 0 and fmap2.shape[1] % self.num_groups == 0
        ), "Number of channels of fmap1 and fmap2 must be the factor of num_groups"
        group_fmap1 = torch.split(fmap1, self.num_groups, dim=1)
        group_fmap2 = torch.split(fmap2, self.num_groups, dim=1)

        cost_volumes = []
        # build feature correlation cost volume
        for i in range(self.num_groups):
            feat1, feat2 = group_fmap1[i], group_fmap2[i]
            num_channels = feat1.shape[1]
            feat1 = torch.permute(feat1, (0, 2, 3, 1))
            feat2 = torch.permute(feat2, (0, 2, 1, 3))
            cost_volumes.append(torch.matmul(feat1, feat2) / num_channels**0.5)
        cost_volumes = torch.stack(cost_volumes, dim=1)

        return cost_volumes


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, bias=False, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Upsampler3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, bias=False, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="trilinear", align_corners=True)
        return self.relu(self.bn(self.conv(x)))


class FeatureGuidedBlock(nn.Module):
    def __init__(self, cv_channel: int, feat_channel: int):
        super(FeatureGuidedBlock, self).__init__()

        self.feat_att = nn.Sequential(
            nn.Conv2d(feat_channel, feat_channel // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feat_channel // 2),
            nn.ReLU(False),
            nn.Conv2d(feat_channel // 2, cv_channel, 1),
        )

    def forward(self, cv: torch.Tensor, feat: torch.Tensor):
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att) * cv
        return cv


class CostVolumeFilterNetwork(nn.Module):
    def __init__(self, in_channels: int, feat_channels: List[int]):
        """
        Initializes the CostVolumeFilterNetwork module.

        Args:
            in_channels (int): The number of input channels.
            feat_channels (List[int]): A list of integers representing the number of feature channels at each level.
        """
        super(CostVolumeFilterNetwork, self).__init__()

        self.conv1 = nn.Sequential(
            ConvBn3D(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1),
            ConvBn3D(in_channels * 2, in_channels * 2, kernel_size=3, stride=1, padding=1),
        )
        self.conv1_feat_guided = FeatureGuidedBlock(in_channels * 2, feat_channels[0])

        self.conv2 = nn.Sequential(
            ConvBn3D(in_channels * 2, in_channels * 4, kernel_size=3, stride=2, padding=1),
            ConvBn3D(in_channels * 4, in_channels * 4, kernel_size=3, stride=1, padding=1),
        )
        self.conv2_feat_guided = FeatureGuidedBlock(in_channels * 4, feat_channels[1])

        self.conv3 = nn.Sequential(
            ConvBn3D(in_channels * 4, in_channels * 8, kernel_size=3, stride=2, padding=1),
            ConvBn3D(in_channels * 8, in_channels * 8, kernel_size=3, stride=1, padding=1),
        )
        self.conv3_feat_guided = FeatureGuidedBlock(in_channels * 8, feat_channels[2])

        self.conv3_up = Upsampler3D(in_channels * 8, in_channels * 4, kernel_size=3, stride=1, padding=1)
        self.proj_3 = ConvBn3D(in_channels * 8, in_channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_up_feat_guided = FeatureGuidedBlock(in_channels * 4, feat_channels[1])

        self.conv2_up = Upsampler3D(in_channels * 4, in_channels * 2, kernel_size=3, stride=1, padding=1)
        self.proj_2 = ConvBn3D(in_channels * 4, in_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_up_feat_guided = FeatureGuidedBlock(in_channels * 2, feat_channels[0])

        self.conv1_up = Upsampler3D(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)
        self.final_conv = ConvBn3D(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
        corr1 = self.conv1(x)
        corr1 = self.conv1_feat_guided(corr1, features[0])

        corr2 = self.conv2(corr1)
        corr2 = self.conv2_feat_guided(corr2, features[1])

        corr3 = self.conv3(corr2)
        corr3 = self.conv3_feat_guided(corr3, features[2])

        corr3_up = self.conv3_up(corr3)
        corr2 = torch.cat((corr3_up, corr2), dim=1)
        corr2 = self.proj_3(corr2)
        corr2 = self.conv3_up_feat_guided(corr2, features[1])

        corr2_up = self.conv2_up(corr2)
        corr1 = torch.cat((corr2_up, corr1), dim=1)
        corr1 = self.proj_2(corr1)
        corr1 = self.conv2_up_feat_guided(corr1, features[0])

        final_corr = self.final_conv(self.conv1_up(corr1))

        return final_corr
