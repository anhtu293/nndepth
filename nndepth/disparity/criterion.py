from torch import nn
import torch
import torch.nn.functional as F
from typing import List, Dict

import aloscene


class DisparityCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def sequence_loss(
        m_outputs: List[Dict[str, torch.Tensor]],
        flow_gt: torch.Tensor,
        gamma: float = 0.8,
        max_flow: int = 1000,
        compute_per_iter: bool = False,
    ):
        """Loss function defined over sequence of flow predictions"""
        n_predictions = len(m_outputs)
        flow_loss = 0.0

        for i in range(n_predictions):
            m_dict = m_outputs[i]
            i_weight = gamma ** (n_predictions - i - 1)

            if m_dict["up_disp"].shape[-2:] != flow_gt.shape[-2:]:
                scale = flow_gt.shape[-1] // m_dict["up_disp"].shape[-1]
                gt = -F.max_pool2d(-flow_gt, kernel_size=scale) / scale
                gt = F.interpolate(gt, size=m_dict["up_disp"].shape[-2:])
            else:
                gt = flow_gt

            mag = torch.sum(gt**2, dim=1, keepdim=True).sqrt()
            valid = mag < max_flow
            i_loss = (m_dict["up_disp"] - gt).abs()
            flow_loss += i_weight * (valid * i_loss).mean()

        if compute_per_iter:
            epe_per_iter = []
            for i in range(n_predictions):
                m_dict = m_outputs[i]
                epe = torch.sum((m_dict["up_disp"] - flow_gt) ** 2, dim=1).sqrt()
                epe = epe.view(-1)[valid.view(-1)]
                epe_per_iter.append(epe)
        else:
            epe_per_iter = None
        epe = torch.sum((m_outputs[-1]["up_disp"] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            "loss": flow_loss.item(),
            "epe": epe.mean().item(),
            "1px": (epe < 1).float().mean().item(),
            "3px": (epe < 3).float().mean().item(),
            "5px": (epe < 5).float().mean().item(),
        }
        return flow_loss, metrics, epe_per_iter

    def forward(
        self, m_outputs: List[Dict[str, torch.Tensor]], frame1: aloscene.Frame, compute_per_iter: bool = False
    ):
        assert isinstance(frame1, aloscene.Frame)
        disp_gt = frame1.disparity
        disp_mask = frame1.disparity.mask

        assert disp_gt.names in [tuple("BCHW"), tuple("BTCHW")]

        disp_gt = disp_gt.as_tensor()
        disp_mask = disp_mask.as_tensor() if disp_mask is not None else None
        flow_loss, metrics, epe_per_iter = self.sequence_loss(m_outputs, disp_gt, compute_per_iter=compute_per_iter)
        return flow_loss, metrics, epe_per_iter
