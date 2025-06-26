import torch
import torch.nn.functional as F
from typing import List, Dict


class CREStereoLoss:
    def __init__(
        self,
        gamma: float = 0.8,
        max_flow: int = 1000,
    ):
        self.gamma = gamma
        self.max_flow = max_flow

    def __call__(self, disp_gt: torch.Tensor, m_outputs: List[Dict[str, torch.Tensor]]):
        """Loss function defined over sequence of flow predictions"""
        n_predictions = len(m_outputs)
        disp_loss = 0.0

        for i in range(n_predictions):
            m_dict = m_outputs[i]
            i_weight = self.gamma ** (n_predictions - i - 1)
            if m_dict["up_disp"].shape[-2:] != disp_gt.shape[-2:]:
                scale = disp_gt.shape[-1] // m_dict["up_disp"].shape[-1]
                gt = -F.max_pool2d(-disp_gt, kernel_size=scale) / scale
                gt = F.interpolate(gt, size=m_dict["up_disp"].shape[-2:])
            else:
                gt = disp_gt

            mag = torch.sum(gt**2, dim=1, keepdim=True).sqrt()
            valid = mag < self.max_flow
            i_loss = (m_dict["up_disp"] - gt).abs()
            disp_loss += i_weight * (valid * i_loss).mean()

        # Compute metrics
        if m_outputs[-1]["up_disp"].shape[-2:] != disp_gt.shape[-2:]:
            scale = disp_gt.shape[-1] // m_outputs[-1]["up_disp"].shape[-1]
            gt = -F.max_pool2d(-disp_gt, kernel_size=scale) / scale
            gt = F.interpolate(gt, size=m_outputs[-1]["up_disp"].shape[-2:])
        epe = torch.sum((m_outputs[-1]["up_disp"] - gt) ** 2, dim=1).sqrt()
        mag = torch.sum(gt**2, dim=1, keepdim=True).sqrt()
        valid = mag < self.max_flow
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            "epe": epe.mean().item(),
            "0.5px": (epe < 0.5).float().mean().item(),
            "1px": (epe < 1).float().mean().item(),
            "3px": (epe < 3).float().mean().item(),
            "5px": (epe < 5).float().mean().item(),
        }
        return disp_loss, metrics
