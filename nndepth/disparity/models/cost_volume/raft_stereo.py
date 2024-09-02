import torch
import torch.nn.functional as F

from nndepth.disparity.models.utils import linear_sampler


class CorrBlock1D:
    """Correlation Block of Raft Stereo.
    Inspired from https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/utils/utils.py
    """

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, num_levels: int = 4, radius: int = 4):
        """Initialize the CorrBlock1D.

        Args:
            fmap1 (torch.Tensor): The feature map 1.
            fmap2 (torch.Tensor): The feature map 2.
            num_levels (int, optional): The number of levels in the correlation pyramid. Defaults to 4.
            radius (int, optional): The radius of the correlation window. Defaults to 4.
        """
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock1D.corr(fmap1, fmap2)

        batch, h1, w1, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, 1, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool1d(corr, 2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords: torch.Tensor):
        r = self.radius
        batch, _, h1, w1 = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            corr = corr.reshape(batch * h1 * w1, -1)
            dx = torch.linspace(-r, r, 2 * r + 1)
            dx = dx.view(1, 2 * r + 1).to(coords.device)
            coords_lvl = dx + coords.reshape(batch * h1 * w1, 1) / 2**i

            corr = linear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1: torch.Tensor, fmap2: torch.Tensor):
        C = fmap1.shape[1]
        fmap1 = torch.permute(fmap1, (0, 2, 3, 1))
        fmap2 = torch.permute(fmap2, (0, 2, 1, 3))
        corr = torch.matmul(fmap1, fmap2) / C**0.5
        return corr


class GroupCorrBlock1D:
    """Group Correlation Block of Raft Stereo."""

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, num_levels: int = 4, radius: int = 4, num_groups=4):
        """
        Initializes the GroupCorrBlock1D module.

        Args:
            fmap1 (torch.Tensor): The feature map 1.
            fmap2 (torch.Tensor): The feature map 2.
            num_levels (int, optional): The number of levels in the correlation pyramid. Defaults to 4.
            radius (int, optional): The radius of the correlation window. Defaults to 4.
            num_groups (int, optional): The number of groups for group correlation. Defaults to 4.
        """
        self.num_levels = num_levels
        self.radius = radius
        self.num_groups = num_groups
        self.corr_pyramid = []

        # all pairs correlation
        corr = self.corr(fmap1, fmap2)

        batch, N, h1, w1, w2 = corr.shape
        corr = corr.reshape(batch * N * h1 * w1, 1, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool1d(corr, 2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords: torch.Tensor):
        r = self.radius
        batch, _, h1, w1 = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            corr = corr.reshape(batch * self.num_groups * h1 * w1, -1)
            dx = torch.linspace(-r, r, 2 * r + 1)
            dx = dx.view(1, 2 * r + 1).to(coords.device)
            coords_1 = torch.permute(coords, (0, 2, 3, 1)).unsqueeze(1).repeat((1, self.num_groups, 1, 1, 1))
            coords_lvl = dx + coords_1.reshape(batch * self.num_groups * h1 * w1, 1) / 2**i

            corr = linear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    def corr(self, fmap1, fmap2) -> torch.Tensor:
        C = fmap1.shape[1]
        group_fmap1 = torch.split(fmap1, self.num_groups, dim=1)
        group_fmap2 = torch.split(fmap2, self.num_groups, dim=1)

        cost_volumes = []
        # build feature correlation cost volume
        for i in range(self.num_groups):
            feat1, feat2 = group_fmap1[i], group_fmap2[i]
            feat1 = torch.permute(feat1, (0, 2, 3, 1))
            feat2 = torch.permute(feat2, (0, 2, 1, 3))
            cost_volumes.append(torch.matmul(feat1, feat2) / C**0.5)
        cost_volumes = torch.stack(cost_volumes, dim=1)

        return cost_volumes
