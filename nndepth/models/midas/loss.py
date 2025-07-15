import torch
import torch.nn as nn
from typing import Tuple, List


def scale_shift_estimation(depth: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate the scale and shift of the depth

    Args:
        depth (torch.Tensor): depth tensor of shape (B, 1, H, W)
        valid_mask (torch.Tensor): valid mask tensor of shape (B, 1, H, W)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: scale and shift of shape (B, 1)
    """
    l_scale = []
    l_shift = []
    for i in range(depth.shape[0]):
        mask = valid_mask[i]
        valid_depth = depth[i][mask]
        shift = torch.median(valid_depth)
        scale = torch.mean(torch.abs(valid_depth - shift))
        l_scale.append(scale)
        l_shift.append(shift)
    return torch.stack(l_scale, dim=0)[:, None, None, None], torch.stack(l_shift, dim=0)[:, None, None, None]


def ssi_depth(depth: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """
    Scale and shift the depth

    Args:
        depth (torch.Tensor): depth tensor of shape (B, 1, H, W)
        scale (torch.Tensor): scale tensor of shape (B, 1)
        shift (torch.Tensor): shift tensor of shape (B, 1)

    Returns:
        torch.Tensor: scaled and shifted depth tensor of shape (B, 1, H, W)
    """
    scale[scale == 0] = 1
    ssi = (depth - shift) / scale
    return ssi


def normalize_01_depth(depth: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    Normalize the depth to [0, 1]
    Args:
        depth (torch.Tensor): depth tensor of shape (B, 1, H, W)
        valid_mask (torch.Tensor): valid mask tensor of shape (B, 1, H, W)

    Returns:
        torch.Tensor: normalized depth tensor of shape (B, 1, H, W)
    """
    min_depth = depth[valid_mask].min()
    max_depth = depth[valid_mask].max()
    normalized_depth = (depth - min_depth) / (max_depth - min_depth + 1e-6)
    return normalized_depth


def mae_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> List[torch.Tensor]:
    """
    Mean absolute error loss

    Args:
        pred (torch.Tensor): predicted depth tensor of shape (B, 1, H, W)
        target (torch.Tensor): target depth tensor of shape (B, 1, H, W)
        valid_mask (torch.Tensor): valid mask tensor of shape (B, 1, H, W)

    Returns:
        List[torch.Tensor]: absolute error loss. List of B elements, each of shape (N) with N being the
            number of valid pixels
    """
    loss = []
    for i in range(len(pred)):
        valid_pred = pred[i][valid_mask[i]]
        valid_target = target[i][valid_mask[i]]
        loss.append(torch.abs(valid_pred - valid_target))
    return loss


def trim_loss(losses: List[torch.Tensor], trim_ratio: float = 0.1) -> List[torch.Tensor]:
    """
    Trimed mean absolute error loss

    Args:
        loss (List[torch.Tensor]): loss tensor. List of B elements, each of shape (N) with N being the
            number of valid pixels
        trim_ratio (float): Trim the highest % of the loss

    Returns:
        List[torch.Tensor]: trimed mean absolute error loss. List of B elements, each of shape (N) with N being the
            number of valid pixels
    """
    trimmed_loss = []
    for i in range(len(losses)):
        loss = losses[i].flatten(-1)
        trim_num = int(loss.shape[-1] * trim_ratio)
        loss = torch.sort(loss, stable=True)[0]
        loss = loss[:-trim_num]
        trimmed_loss.append(loss)
    return trimmed_loss


def average_loss(losses: List[torch.Tensor]) -> torch.Tensor:
    """
    Average the loss

    Args:
        losses (List[torch.Tensor]): loss tensor. List of B elements, each of shape (N) with N being the
            number of valid pixels
    Returns:
        torch.Tensor: average loss
    """
    total_loss_per_image = []
    total_pixels = 0
    for i in range(len(losses)):
        total_pixels += losses[i].shape[-1]
        total_loss_per_image.append(torch.sum(losses[i]))
    total_loss = torch.sum(torch.stack(total_loss_per_image, dim=0))
    return total_loss / total_pixels


def gradient_regulizer(
    normalized_target: torch.Tensor,
    normalized_pred: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Gradient regulizer.
    Resize ground truth maxpooling and pred by average pooling, getting the multi-scale approach.

    Args:
        normalized_target (torch.Tensor): normalized target depth tensor.
        normalized_pred (torch.Tensor): normalized predicted depth tensor.
        valid_mask (torch.Tensor): valid mask tensor.

    Returns:
        torch.Tensor: gradient regulizer
    """
    # Average pooling & Max pooling
    avgpool = torch.nn.AvgPool2d((2, 2), stride=2)
    maxpool = torch.nn.MaxPool2d((2, 2), return_indices=True, stride=2)

    # Gradient regulizer
    gradient_reg = 0
    num_valid_pixels = valid_mask.sum()

    normalized_target[~valid_mask] = 0

    for _ in range(4):
        R = normalized_pred - normalized_target
        Rx = R[:, :, :-1, :-1] - R[:, :, :-1, 1:]
        Ry = R[:, :, :-1, :-1] - R[:, :, 1:, :-1]
        mask = valid_mask[:, :, 1:, 1:]

        R = torch.abs(Rx) + torch.abs(Ry)
        gradient_reg += torch.sum(R[mask])

        # Reduce input size to half
        normalized_target, indices = maxpool(normalized_target)
        valid_mask = valid_mask.flatten(2).gather(dim=2, index=indices.flatten(2)).view_as(indices)
        normalized_pred = avgpool(normalized_pred)
    return gradient_reg / num_valid_pixels


class MiDasLoss(nn.Module):
    def __init__(self, trimmed_loss_coef: float = 1.0, gradient_reg_coef: float = 0.1):
        """
        MiDasLoss

        Args:
            trimmed_loss_coef (float): trimmed loss coefficient
            gradient_reg_coef (float): gradient regularization coefficient
        """

        super(MiDasLoss, self).__init__()
        self.trimmed_loss_coef = trimmed_loss_coef
        self.gradient_reg_coef = gradient_reg_coef

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred (torch.Tensor): predicted depth tensor of shape (B, 1, H, W)
            target (torch.Tensor): target depth tensor of shape (B, 1, H, W)
            valid_mask (torch.Tensor): valid mask tensor of shape (B, 1, H, W)

        Returns:
            Tuple[torch.Tensor, dict]: loss and metrics
        """
        # Compute SSI trimmed loss
        normalized_target = normalize_01_depth(target, valid_mask)
        gt_scale, gt_shift = scale_shift_estimation(normalized_target, valid_mask)
        ssi_target = ssi_depth(normalized_target, gt_scale, gt_shift)

        pred_scale, pred_shift = scale_shift_estimation(pred, valid_mask)
        ssi_pred = ssi_depth(pred, pred_scale, pred_shift)

        # Compute SSI trimmed loss
        mae = mae_loss(ssi_pred, ssi_target, valid_mask)
        trimed_mae = trim_loss(mae)
        trimed_mae = average_loss(trimed_mae)

        # Compute SSI gradient matching loss
        gradient_reg = gradient_regulizer(normalized_target, ssi_pred, valid_mask)

        loss = trimed_mae * self.trimmed_loss_coef + gradient_reg * self.gradient_reg_coef

        # Absolute deviation
        abs_dev = torch.abs(ssi_pred - ssi_target)
        abs_dev = abs_dev[valid_mask]
        abs_dev = abs_dev.mean()

        metrics = {
            "loss": loss.detach().cpu().item(),
            "trimmed_mae": trimed_mae.detach().cpu().item(),
            "gradient_reg": gradient_reg.detach().cpu().item(),
            "abs_dev": abs_dev.detach().cpu().item(),
        }
        return loss, metrics
