import torch
import numpy as np
import argparse
import sys
import yaml
from tqdm import tqdm
from loguru import logger
from tabulate import tabulate
from typing import Dict

from nndepth.utils.common import load_weights
from nndepth.models.midas.config import MBNetV3DepthModelConfig
from nndepth.models.midas.models import MobileNetV3DepthModel
from nndepth.models.midas.loss import scale_shift_estimation, ssi_depth, normalize_01_depth
from nndepth.data.dataloaders.depth import (
    TartanairDepthDataLoader,
    DIMLDepthDataLoader,
    HypersimDepthDataLoader,
    HRWSIDepthDataLoader
)


NAME_TO_MODEL_CONFIG = {
    "mbnet_v3": {
        "model_config": MBNetV3DepthModelConfig,
        "model": MobileNetV3DepthModel,
    },
}

DATA_LOADERS = {
    "tartanair": TartanairDepthDataLoader,
    "diml": DIMLDepthDataLoader,
    "hypersim": HypersimDepthDataLoader,
    "hrwsi": HRWSIDepthDataLoader,
}


class DepthEvalCriterion:
    def __init__(self, max_depth: float = 80.0):
        """
        Depth Evaluation Metrics for MiDaS

        Args:
            max_depth (float): Maximum depth value to consider for evaluation
        """
        self.max_depth = max_depth

    def __call__(
        self,
        depth_pred: torch.Tensor,
        depth_gt: torch.Tensor,
        valid_mask: torch.Tensor = None
    ) -> Dict[str, float]:
        """
        Compute depth evaluation metrics

        Args:
            depth_pred (torch.Tensor): Predicted depth of shape (B, 1, H, W)
            depth_gt (torch.Tensor): Ground truth depth of shape (B, 1, H, W)
            valid_mask (torch.Tensor): Valid mask of shape (B, 1, H, W)

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        # Apply scale and shift alignment for relative depth prediction
        depth_pred_aligned = self._scale_shift_align(depth_pred, depth_gt, valid_mask)

        # Create validity mask
        if valid_mask is None:
            valid_mask = torch.ones_like(depth_gt, dtype=torch.bool)

        # Apply depth range limits
        range_mask = (depth_gt > 0.1) & (depth_gt < self.max_depth)
        valid_mask = valid_mask & range_mask

        # Get valid pixels
        pred_valid = depth_pred_aligned[valid_mask]
        gt_valid = depth_gt[valid_mask]

        if len(pred_valid) == 0:
            return self._empty_metrics()

        # Compute standard depth metrics
        abs_rel = torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid).item()
        sq_rel = torch.mean(((pred_valid - gt_valid) ** 2) / gt_valid).item()
        rmse = torch.sqrt(torch.mean((pred_valid - gt_valid) ** 2)).item()
        rmse_log = torch.sqrt(torch.mean((torch.log(pred_valid) - torch.log(gt_valid)) ** 2)).item()

        # Accuracy metrics (δ < threshold)
        ratio = torch.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
        delta1 = torch.mean((ratio < 1.25).float()).item()
        delta2 = torch.mean((ratio < 1.25 ** 2).float()).item()
        delta3 = torch.mean((ratio < 1.25 ** 3).float()).item()

        # Scale invariant depth metrics using SSI
        ssi_error = self._compute_ssi_metrics(depth_pred_aligned, depth_gt, valid_mask)

        return {
            "abs_rel": abs_rel,
            "sq_rel": sq_rel,
            "rmse": rmse,
            "rmse_log": rmse_log,
            "delta1": delta1,
            "delta2": delta2,
            "delta3": delta3,
            "ssi_mae": ssi_error["ssi_mae"],
            "ssi_rmse": ssi_error["ssi_rmse"],
        }

    def _scale_shift_align(
        self,
        depth_pred: torch.Tensor,
        depth_gt: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply scale and shift alignment to predicted depth

        Args:
            depth_pred (torch.Tensor): Predicted depth of shape (B, 1, H, W)
            depth_gt (torch.Tensor): Ground truth depth of shape (B, 1, H, W)
            valid_mask (torch.Tensor): Valid mask of shape (B, 1, H, W)

        Returns:
            torch.Tensor: Aligned depth of shape (B, 1, H, W)
        """
        depth_pred_aligned = depth_pred.clone()

        for i in range(depth_pred.shape[0]):
            if valid_mask is not None:
                mask = valid_mask[i]  # Keep the channel dimension
            else:
                mask = torch.ones_like(depth_gt[i], dtype=torch.bool)

            if torch.sum(mask) > 100:  # Minimum number of valid pixels
                # Properly index using the mask
                pred_valid = depth_pred[i].view(-1)[mask.view(-1)]
                gt_valid = depth_gt[i].view(-1)[mask.view(-1)]

                # Least squares solution for scale and shift
                A = torch.stack([pred_valid, torch.ones_like(pred_valid)], dim=1)
                solution = torch.linalg.lstsq(A, gt_valid)[0]
                scale, shift = solution[0], solution[1]

                depth_pred_aligned[i] = depth_pred[i] * scale + shift

        return depth_pred_aligned

    def _compute_ssi_metrics(
        self,
        depth_pred: torch.Tensor,
        depth_gt: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute Scale and Shift Invariant (SSI) metrics

        Args:
            depth_pred (torch.Tensor): Predicted depth of shape (B, 1, H, W)
            depth_gt (torch.Tensor): Ground truth depth of shape (B, 1, H, W)
            valid_mask (torch.Tensor): Valid mask of shape (B, 1, H, W)

        Returns:
            Dict[str, float]: Dictionary of SSI metrics
        """
        try:
            # Normalize depth to [0, 1] range
            gt_normalized = normalize_01_depth(depth_gt, valid_mask)
            pred_normalized = normalize_01_depth(depth_pred, valid_mask)

            # Scale and shift estimation
            gt_scale, gt_shift = scale_shift_estimation(gt_normalized, valid_mask)
            pred_scale, pred_shift = scale_shift_estimation(pred_normalized, valid_mask)

            # Apply SSI transformation
            ssi_gt = ssi_depth(gt_normalized, gt_scale, gt_shift)
            ssi_pred = ssi_depth(pred_normalized, pred_scale, pred_shift)

            # Compute metrics on SSI depths
            valid_pixels = valid_mask.bool()
            ssi_error = torch.abs(ssi_pred - ssi_gt)[valid_pixels]
            ssi_sq_error = (ssi_pred - ssi_gt)[valid_pixels] ** 2

            ssi_mae = torch.mean(ssi_error).item()
            ssi_rmse = torch.sqrt(torch.mean(ssi_sq_error)).item()

            return {"ssi_mae": ssi_mae, "ssi_rmse": ssi_rmse}

        except Exception as e:
            logger.warning(f"Failed to compute SSI metrics: {e}")
            return {"ssi_mae": float('inf'), "ssi_rmse": float('inf')}

    def _empty_metrics(self) -> Dict[str, float]:
        """
        Return empty metrics when no valid pixels are available

        Returns:
            Dict[str, float]: Dictionary of empty metrics
        """
        return {
            "abs_rel": float('inf'),
            "sq_rel": float('inf'),
            "rmse": float('inf'),
            "rmse_log": float('inf'),
            "delta1": 0.0,
            "delta2": 0.0,
            "delta3": 0.0,
            "ssi_mae": float('inf'),
            "ssi_rmse": float('inf'),
        }


def parse_args(model_name: str):
    parser = argparse.ArgumentParser(description="Evaluate MiDaS depth estimation model")
    NAME_TO_MODEL_CONFIG[model_name]["model_config"].add_args(parser)

    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
        choices=list(DATA_LOADERS.keys()),
        help="Dataset name for evaluation"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        required=True,
        help="Path to dataset configuration YAML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=80.0,
        help="Maximum depth value for evaluation"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all samples)"
    )

    args = parser.parse_args(sys.argv[2:])
    return args


@torch.no_grad()
def main(model_name: str, args):
    # Instantiate the model
    model_config_cls = NAME_TO_MODEL_CONFIG[model_name]["model_config"]
    model_cls = NAME_TO_MODEL_CONFIG[model_name]["model"]
    model_config = model_config_cls.from_args(args)
    model = model_cls(**model_config.to_dict())
    model = load_weights(model, model_config.weights, strict_load=model_config.strict_load).cuda()
    model.eval()
    logger.info("Model loaded successfully!")

    # Initialize dataloader
    with open(args.data_config, "r") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)

    dataloader = DATA_LOADERS[args.data_name](**data_config)
    dataloader.setup(stage="val")
    logger.info(f"Loaded {args.data_name} dataset with {len(dataloader.val_dataloader)} batches")

    # Initialize evaluation criterion
    criterion = DepthEvalCriterion(max_depth=args.max_depth)

    # Storage for metrics
    all_metrics = {
        "abs_rel": [],
        "sq_rel": [],
        "rmse": [],
        "rmse_log": [],
        "delta1": [],
        "delta2": [],
        "delta3": [],
        "ssi_mae": [],
        "ssi_rmse": [],
    }

    # Evaluation loop
    total_samples = 0
    for i_batch, batch in enumerate(tqdm(dataloader.val_dataloader, desc="Evaluating")):
        if args.num_samples is not None and total_samples >= args.num_samples:
            break

        input_tensor = batch.data.cuda()
        gt_depth = batch.depth.data.cuda()
        valid_mask = batch.depth.valid_mask.cuda()
        pred_depth = model(input_tensor)
        metrics = criterion(pred_depth, gt_depth, valid_mask)

        for key, value in metrics.items():
            if not (np.isnan(value) or np.isinf(value)):
                all_metrics[key].append(value)

        total_samples += input_tensor.shape[0]

    # Compute final statistics
    final_metrics = {}
    for key, values in all_metrics.items():
        if len(values) > 0:
            final_metrics[key] = np.mean(values)
        else:
            final_metrics[key] = float('inf') if key not in ["delta1", "delta2", "delta3"] else 0.0

    # Create results table
    table = []
    metric_names = {
        "abs_rel": "Abs Rel Error",
        "sq_rel": "Sq Rel Error",
        "rmse": "RMSE",
        "rmse_log": "RMSE (log)",
        "delta1": "δ < 1.25",
        "delta2": "δ < 1.25²",
        "delta3": "δ < 1.25³",
        "ssi_mae": "SSI MAE",
        "ssi_rmse": "SSI RMSE",
    }

    for key, name in metric_names.items():
        value = final_metrics[key]
        if key in ["delta1", "delta2", "delta3"]:
            table.append([name, f"{value:.3f}"])
        else:
            table.append([name, f"{value:.4f}"])

    # Print and save results
    tab = tabulate(table, headers=["Metric", "Value"], tablefmt="grid")
    print("\nEvaluation Results:")
    print(tab)

    # Save to file
    with open(args.output, "w") as f:
        f.write(f"MiDaS {model_name} Evaluation Results\n")
        f.write(f"Dataset: {args.data_name}\n")
        f.write(f"Max Depth: {args.max_depth}m\n")
        f.write(f"Samples Evaluated: {total_samples}\n\n")
        f.write(tab)

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    model_name = sys.argv[1]
    assert (
        model_name in NAME_TO_MODEL_CONFIG
    ), f"Model {model_name} not found. Available models: {list(NAME_TO_MODEL_CONFIG.keys())}"

    args = parse_args(model_name)
    main(model_name, args)
