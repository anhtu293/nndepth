import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from loguru import logger
from tabulate import tabulate
from typing import Tuple

import aloscene

from nndepth.utils.common import instantiate_with_config_file, load_weights
from nndepth.disparity.criterions import EvalCriterion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--data_config", type=str, required=True, help="Path to data config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weight")
    parser.add_argument("--metric_name", nargs="+", default=[], help="Name of metric. Example: kitti-d1")
    parser.add_argument(
        "--metric_threshold",
        nargs="+",
        default=[],
        help="Threshold to compute metrics: percentage of points whose error is larger than `metric_threshold`")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output. Directory in case of save_format == image, mp4 file in case of video",
    )
    args = parser.parse_args()
    return args


def preprocess_frame(frame: aloscene.Frame, HW: Tuple[int, int]) -> aloscene.Frame:
    return frame.resize(HW).norm_minmax_sym().batch()


@torch.no_grad()
def main(args):
    assert len(args.metric_name) == len(args.metric_threshold), "length of `metric_name` and `metric_threshold` must be equal."

    # Instantiate the model
    model, _ = instantiate_with_config_file(args.model_config, "nndepth.disparity.models")
    model = load_weights(model, args.weights, strict_load=True).cuda()
    model.eval()
    logger.info("Model is loaded successfully !")

    # init dataloader
    dataloader, _ = instantiate_with_config_file(args.data_config, "nndepth.disparity.models")
    dataloader.setup()

    # init criterion
    metric_info = {}
    for i, metric_name in enumerate((args.metric_name)):
        metric_info[metric_name] = args.metric_threshold[i]
    criterion = EvalCriterion(metric_info)

    global_metrics = {"epe": []}
    for k in metric_info.keys():
        global_metrics[k] = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        left_frame, right_frame = batch["left"], batch["right"]
        left_frame, right_frame = left_frame.as_tensor().cuda(), right_frame.as_tensor().cuda()

        # Forward
        m_outputs = model(left_frame, right_frame)
        disp_pred = m_outputs[-1][""]["up_disp"]
        disp_gt = left_frame.disparity.as_tensor().cuda()

        metrics = criterion(disp_gt, disp_pred)
        for k, v in metrics:
            global_metrics[k].append(v)

    table = []
    for key, metrics in global_metrics.items():
        table.append([key, np.mean(metrics)])

    tab = tabulate.tabulate(table, headers=["Metric", "Value"], tablefmt="grid")
    with open(args.output, "w") as f:
        f.write(tab)


if __name__ == "__main__":
    args = parse_args()
    main(args)
