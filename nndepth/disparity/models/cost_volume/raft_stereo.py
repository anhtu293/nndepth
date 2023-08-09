import torch
import torch.nn.functional as F

from nndepth.disparity.utils import linear_sampler


class CorrBlock1D:
    """Correlation Block of Raft Stereo.
    Inspired from https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/utils/utils.py
    """
    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, num_levels: int = 4, radius: int = 4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock1D.corr(fmap1, fmap2)

        batch, h1, w1, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, 1, 1, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
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
            coords_lvl = dx + coords.reshape(batch * h1 * w1, 1, ) / 2 ** i

            corr = linear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        C = fmap1.shape[1]
        fmap1 = torch.permute(fmap1, (0, 2, 3, 1))
        fmap2 = torch.permute(fmap2, (0, 2, 1, 3))
        corr = torch.matmul(fmap1, fmap2) / C ** 0.5
        return corr
