import os
import torch
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')

import aloscene

from nndepth.disparity.train import LitDisparityModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser = LitDisparityModel.add_argparse_args(parser)
    parser.add_argument("--weight", type=str, required=True, help="Path to model weight")
    parser.add_argument("--left_path", type=str, required=True, help="Path to directory of left images")
    parser.add_argument("--right_path", type=str, required=True, help="Path to directory of right images")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output. Directory in case of save_format == image, mp4 file in case of video"
    )
    parser.add_argument("--render", action="store_true", help="Render results")
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["image", "video"],
        default="video",
        help="Which format to save output. image or video are supported. Default: %(default)s"
    )
    args = parser.parse_args()
    return args


@torch.no_grad()
def main(args):
    # load model
    lit = LitDisparityModel.load_from_checkpoint(args.weight, strict=True, args=args)
    lit.model.eval()
    print("Model is loaded successfully !")

    if args.save_format == "image":
        os.makedirs(args.output, exist_ok=True)

    # load files
    left_files = sorted(os.listdir(args.left_path))
    left_files = [os.path.join(args.left_path, x) for x in left_files]

    right_files = sorted(os.listdir(args.right_path))
    right_files = [os.path.join(args.right_path, x) for x in right_files]

    if len(left_files) != len(right_files):
        ValueError(f"Left and Right dont have the same number of frames.\
                   Found {len(left_files)} left frames and {len(right_files)} right frames !")

    print(f"Found {len(left_files)} frames !")

    left_files = left_files[:150]
    right_files = right_files[:150]

    for idx, (left, right) in tqdm(enumerate(zip(left_files, right_files))):
        left_frame = aloscene.Frame(left).norm_minmax_sym()
        right_frame = aloscene.Frame(right).norm_minmax_sym()
        inputs = [{"left": left_frame, "right": right_frame}]

        # get model output
        output = lit(inputs)

        # format output into aloscene.Disparity object
        output = lit.inference(output, only_last=True)
        disp_view = output.get_view(min_disp=None, max_disp=None, cmap="RdYlGn")

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
