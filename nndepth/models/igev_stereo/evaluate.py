import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from loguru import logger
from tabulate import tabulate
from typing import Dict
from nndepth.utils.common import load_weights
from nndepth.data.dataloaders.utils import Padder
from nndepth.models.igev_stereo import STEREO_MODELS
from nndepth.data.dataloaders import TartanairDisparityDataLoader, Kitti2015DisparityDataLoader


DATA_LOADERS = {
    "tartanair": TartanairDisparityDataLoader,
    "kitti": Kitti2015DisparityDataLoader,
}


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, choices=STEREO_MODELS.keys(), help="Name of the model to evaluate"
    )
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config file")
    parser.add_argument(
        "--data_name", type=str, required=True, choices=DATA_LOADERS.keys(), help="Name of the dataset to evaluate"
    )
    parser.add_argument("--data_config", type=str, required=True, help="Path to data config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weight")
    parser.add_argument("--metric_name", nargs="+", default=[], help="Name of metric. Example: kitti-d1")
    parser.add_argument(
        "--metric_threshold",
        nargs="+",
        type=float,
        default=[],
        help="Threshold to compute metrics: percentage of points whose error is larger than `metric_threshold`")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output. Directory in case of save_format == image, mp4 file in case of video",
    )
    parser.add_argument(
        "--divisible_by",
        type=int,
        default=32,
        help="The input resolution of image will be padded so that its height and width are divisible by this number\
             which is highest downsample of backbone. Default: 32 for RAFTStereo")
    args = parser.parse_args()
    return args


@torch.no_grad()
def main(args):
    assert len(args.metric_name) == len(args.metric_threshold), (
        "length of `metric_name` and `metric_threshold` must be equal."
    )

    # Instantiate the model
    model, _ = STEREO_MODELS[args.model_name].init_from_config(args.model_config)
    model = load_weights(model, args.weights, strict_load=True).cuda()
    model.eval()
    logger.info("Model is loaded successfully !")

    # init dataloader
    dataloader, data_config = DATA_LOADERS[args.data_name].init_from_config(args.data_config)
    dataloader.setup()

    # init criterion
    metric_info = {}
    for i, metric_name in enumerate((args.metric_name)):
        metric_info[metric_name] = args.metric_threshold[i]
    criterion = EvalCriterion(metric_info)

    global_metrics = {"epe": []}
    for k in metric_info.keys():
        global_metrics[k] = []

    # padder
    padder = Padder(data_config["HW"], divis_by=args.divisible_by)

    for batch in tqdm(dataloader.val_dataloader, desc="Evaluating"):
        left_frame, right_frame = batch["left"][0], batch["right"][0]
        left_tensor, right_tensor = left_frame.data[None].cuda(), right_frame.data[None].cuda()
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

        # Forward
        m_outputs = model(left_tensor, right_tensor)
        disp_pred = m_outputs[-1]["up_disp"]
        disp_pred = padder.unpad(disp_pred)
        disp_gt = left_frame.disparity.data[None].cuda()
        if left_frame.disparity.occlusion is not None:
            occ_mask = left_frame.disparity.occlusion[None].cuda() > 0
            metrics = criterion(disp_gt, disp_pred, ~occ_mask)
        else:
            metrics = criterion(disp_gt, disp_pred)

        for k, v in metrics.items():
            global_metrics[k].append(v)

    table = []
    for key, metrics in global_metrics.items():
        table.append([key, np.mean(metrics)])

    tab = tabulate(table, headers=["Metric", "Value"], tablefmt="grid")
    with open(args.output, "w") as f:
        f.write(tab)


if __name__ == "__main__":
    args = parse_args()
    main(args)
