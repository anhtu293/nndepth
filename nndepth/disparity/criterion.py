from torch import nn
import torch

import aloscene


class DisparityCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def sequence_loss(m_outputs, flow_gt, valid=None, gamma=0.8, max_flow=400, compute_per_iter=False):
        """Loss function defined over sequence of flow predictions"""
        n_predictions = len(m_outputs)
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        if valid is None:
            valid = torch.ones_like(mag, dtype=torch.bool)
        else:
            valid = (valid >= 0.5) & (mag < max_flow)
        for i in range(n_predictions):
            m_dict = m_outputs[i]
            i_weight = gamma ** (n_predictions - i - 1)
            i_loss = (m_dict["up_flow"] - flow_gt).abs()
            flow_loss += i_weight * (valid * i_loss).mean()

        if compute_per_iter:
            epe_per_iter = []
            for i in range(n_predictions):
                m_dict = m_outputs[i]
                epe = torch.sum((m_dict["up_flow"] - flow_gt) ** 2, dim=1).sqrt()
                epe = epe.view(-1)[valid.view(-1)]
                epe_per_iter.append(epe)
        else:
            epe_per_iter = None
        epe = torch.sum((m_outputs[-1]["up_flow"] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            "loss": flow_loss.item(),
            "epe": epe.mean().item(),
            "1px": (epe < 1).float().mean().item(),
            "3px": (epe < 3).float().mean().item(),
            "5px": (epe < 5).float().mean().item(),
        }
        return flow_loss, metrics, epe_per_iter

    def forward(self, m_outputs, frame1, use_valid=True, compute_per_iter=False):
        assert isinstance(frame1, aloscene.Frame)
        disp_gt = frame1.disparity
        disp_mask = frame1.disparity.mask

        assert disp_gt.names in [tuple("BCHW"), tuple("BTCHW")]

        disp_gt = disp_gt.as_tensor()
        disp_mask = disp_mask.as_tensor() if disp_mask is not None else None
        valid = self.get_valid(disp_gt, disp_mask, use_valid)
        flow_loss, metrics, epe_per_iter = self.sequence_loss(
            m_outputs, disp_gt, valid, compute_per_iter=compute_per_iter
        )
        return flow_loss, metrics, epe_per_iter

    @staticmethod
    def get_valid(disp_gt, disp_mask, use_valid):
        if disp_mask is None and not use_valid:
            return None
        elif disp_mask is None and use_valid:
            return disp_gt.abs() < 1000
        elif disp_mask is not None and use_valid:
            return disp_mask.bool() & (disp_gt.abs() < 1000)

        else:
            return disp_mask.bool()
