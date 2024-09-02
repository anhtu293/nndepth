import torch
import torch.nn.functional as F
from typing import Dict


class EvalCriterion:
    def __init__(self, d_threshold: dict = None, max_flow: int = 1000):
        """
        Evaluation Metric Computation class

        Args:
            d_threshold (Dict): threshold to compute dX metric in stereo depth estimation.
                Example: {"kitti-d1": 3.0}
        """
        self.d_threshold = d_threshold
        self.max_flow = max_flow

    def __call__(
            self, disp_gt: torch.Tensor, disp_pred: torch.Tensor, valid_mask: torch.Tensor = None
    ) -> Dict[str, float]:
        """
        Compute metrics
        Args:
            disp_gt (torch.Tensor): GT of disparity.
            disp_pred (torch.Tensor): Prediction of disparity.

        Return:
            {
                "epe": epe,
                ...
            }
        """
        if disp_pred.shape[-2:] != disp_gt.shape[-2:]:
            scale = disp_gt.shape[-1] // disp_pred.shape[-1]
            gt = -F.max_pool2d(-disp_gt, kernel_size=scale) / scale
            gt = F.interpolate(gt, size=disp_pred.shape[-2:])
        else:
            gt = disp_gt

        epe = torch.sum((disp_pred - gt) ** 2, dim=1).sqrt()
        mag = torch.sum(gt**2, dim=1, keepdim=True).sqrt()
        valid = mag < self.max_flow
        if valid_mask is not None:
            valid = valid & valid_mask
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            "epe": epe.mean().item(),
        }
        if self.d_threshold is not None:
            for key, thres in self.d_threshold.items():
                metrics[key] = (epe > thres).float().mean().item()
        return metrics
