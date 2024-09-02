import os
import torch
import argparse
from tqdm import tqdm
import matplotlib
from loguru import logger
from typing import Tuple

matplotlib.use("TkAgg")

import aloscene

from nndepth.utils.common import instantiate_with_config_file, load_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weight")
    parser.add_argument("--left_path", type=str, required=True, help="Path to directory of left images")
    parser.add_argument("--right_path", type=str, required=True, help="Path to directory of right images")
    parser.add_argument("--HW", type=int, nargs="+", default=(480, 640), help="Model input size")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output. Directory in case of save_format == image, mp4 file in case of video",
    )
    parser.add_argument("--render", action="store_true", help="Render results")
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["image", "video"],
        default="video",
        help="Which format to save output. image or video are supported. Default: %(default)s",
    )
    args = parser.parse_args()
    return args


def preprocess_frame(frame: aloscene.Frame, HW: Tuple[int, int]) -> aloscene.Frame:
    return frame.resize(HW).norm_minmax_sym().batch()


@torch.no_grad()
def main(args):
    # Instantiate the model
    model, model_config = instantiate_with_config_file(args.model_config, "nndepth.disparity.models")
    model = load_weights(model, args.weights, strict_load=True).cuda()
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

    for idx, (left, right) in tqdm(enumerate(zip(left_files, right_files))):
        left_frame = preprocess_frame(aloscene.Frame(left), args.HW)
        right_frame = preprocess_frame(aloscene.Frame(right), args.HW)

        left_tensor = left_frame.as_tensor().cuda()
        right_tensor = right_frame.as_tensor().cuda()

        # get model output
        output = model(left_tensor, right_tensor)

        # format output into aloscene.Disparity object
        disp_pred = aloscene.Disparity(
                output[-1]["up_disp"].cpu(),
                names=("B", "C", "H", "W"),
                camera_side="left",
                disp_format="signed",
            )
        disp_view = disp_pred.get_view(min_disp=None, max_disp=None, cmap="RdYlGn")

        # get frame visualization
        frame_view = left_frame.get_view()

        if args.save_format == "image":
            frame_view = frame_view.add(disp_view)
            frame_view.save(os.path.join(args.output, str(idx) + ".png"))

        if args.save_format == "video" or args.render:
            aloscene.render([frame_view, disp_view], record_file=args.output, skip_views=not args.render)


if __name__ == "__main__":
    args = parse_args()
    main(args)
