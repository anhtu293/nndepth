import os
import torch
import torch.nn.functional as F
import argparse
import cv2
import sys
import numpy as np
from tqdm import tqdm
from loguru import logger
from typing import Tuple

from nndepth.scene import Frame, Disparity
from nndepth.utils.common import load_weights

from nndepth.models.raft_stereo.configs import BaseRAFTStereoModelConfig, RepViTRAFTStereoModelConfig
from nndepth.models.raft_stereo.model import BaseRAFTStereo, Coarse2FineGroupRepViTRAFTStereo


NAME_TO_MODEL_CONFIG = {
    "base": {
        "model_config": BaseRAFTStereoModelConfig,
        "model": BaseRAFTStereo,
    },
    "repvit": {
        "model_config": RepViTRAFTStereoModelConfig,
        "model": Coarse2FineGroupRepViTRAFTStereo,
    },
}


def parse_args(model_name: str):
    parser = argparse.ArgumentParser()
    NAME_TO_MODEL_CONFIG[model_name]["model_config"].add_args(parser)
    parser.add_argument("--left_path", type=str, required=True, help="Path to directory of left images")
    parser.add_argument("--right_path", type=str, required=True, help="Path to directory of right images")
    parser.add_argument("--HW", type=int, nargs="+", default=(480, 640), help="Model input size")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output. Directory in case of save_format == image, mp4 file in case of video",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["image", "video"],
        default="video",
        help="Which format to save output. image or video are supported. Default: %(default)s",
    )
    parser.add_argument("--viz_hw", type=int, nargs="+", default=(480, 640), help="Resolution of output image/video")
    args = parser.parse_args(sys.argv[2:])
    return args


def preprocess_frame(frame: torch.Tensor, HW: Tuple[int, int]) -> Frame:
    frame = frame.unsqueeze(0)
    # frame = frame.resize(HW)
    frame = F.interpolate(frame, HW, mode="bilinear")
    frame = (frame - 127.5) / 127.5
    return frame


def load_image(image_path: str) -> torch.Tensor:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@torch.no_grad()
def main(model_name: str, args):
    # Instantiate the model
    model_config_cls = NAME_TO_MODEL_CONFIG[model_name]["model_config"]
    model_cls = NAME_TO_MODEL_CONFIG[model_name]["model"]
    model_config = model_config_cls.from_args(args)
    model = model_cls(**model_config.to_dict())
    model = load_weights(model, model_config.weights, strict_load=model_config.strict_load).cuda()
    model.eval()
    logger.info("Model is loaded successfully !")

    if args.save_format == "image":
        os.makedirs(args.output, exist_ok=True)

    # load files
    left_files = sorted(os.listdir(args.left_path))
    left_files = [os.path.join(args.left_path, x) for x in left_files]

    right_files = sorted(os.listdir(args.right_path))
    right_files = [os.path.join(args.right_path, x) for x in right_files]

    if len(left_files) != len(right_files):
        ValueError(
            f"Left and Right dont have the same number of frames.\
                   Found {len(left_files)} left frames and {len(right_files)} right frames !"
        )

    logger.info(f"Found {len(left_files)} frames !")

    if args.save_format == "video":
        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"MJPG"),
            15,
            (args.viz_hw[1] * 2, args.viz_hw[0])
        )

    for idx, (left, right) in tqdm(enumerate(zip(left_files, right_files))):
        left_image, right_image = load_image(left), load_image(right)
        left_frame = preprocess_frame(torch.Tensor(left_image.transpose((2, 0, 1))), args.HW)
        right_frame = preprocess_frame(torch.Tensor(right_image.transpose((2, 0, 1))), args.HW)

        left_frame = left_frame.cuda()
        right_frame = right_frame.cuda()

        # get model output
        output = model(left_frame, right_frame)

        # format output into Disparity object
        disp_pred = Disparity(data=output[-1]["up_disp"][0].cpu(), disp_sign="negative")
        disp_view = disp_pred.get_view(cmap="RdYlGn")
        disp_view = cv2.resize(disp_view, (args.viz_hw[1], args.viz_hw[0]))
        left_image = cv2.resize(left_image, (args.viz_hw[1], args.viz_hw[0]))

        frame_view = np.concatenate([left_image, disp_view], axis=1)
        frame_view = cv2.cvtColor(frame_view, cv2.COLOR_BGR2RGB)

        if args.save_format == "image":
            cv2.imwrite(os.path.join(args.output, str(idx) + ".png"), frame_view)

        if args.save_format == "video":
            writer.write(frame_view)


if __name__ == "__main__":
    model_name = sys.argv[1]
    assert (
        model_name in NAME_TO_MODEL_CONFIG
    ), f"Model {model_name} not found. Available models: {NAME_TO_MODEL_CONFIG.keys()}"

    args = parse_args(model_name)
    main(model_name, args)
