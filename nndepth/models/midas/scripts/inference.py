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

from nndepth.scene import Depth
from nndepth.utils.common import load_weights

from nndepth.models.midas.config import MBNetV3DepthModelConfig
from nndepth.models.midas.models import MobileNetV3DepthModel


NAME_TO_MODEL_CONFIG = {
    "mbnet_v3": {
        "model_config": MBNetV3DepthModelConfig,
        "model": MobileNetV3DepthModel,
    },
}


def parse_args(model_name: str):
    parser = argparse.ArgumentParser()
    NAME_TO_MODEL_CONFIG[model_name]["model_config"].add_args(parser)
    parser.add_argument("--input_path", type=str, required=True, help="Path to single input image")
    parser.add_argument("--HW", type=int, nargs="+", default=(384, 384), help="Model input size")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output",
    )
    parser.add_argument("--viz_hw", type=int, nargs="+", default=(480, 640), help="Resolution of output image")
    parser.add_argument("--cmap", type=str, default="magma", help="Colormap for depth visualization")
    args = parser.parse_args(sys.argv[2:])
    return args


def preprocess_frame(frame: torch.Tensor, HW: Tuple[int, int]) -> torch.Tensor:
    """
    Preprocess input frame for MiDaS model

    Args:
        frame (torch.Tensor): Input frame tensor of shape (C, H, W)
        HW (Tuple[int, int]): Target height and width

    Returns:
        torch.Tensor: Preprocessed frame tensor of shape (1, C, H, W)
    """
    frame = frame.unsqueeze(0)
    frame = F.interpolate(frame, HW, mode="bilinear", align_corners=False)
    frame = (frame - 127.5) / 127.5  # Normalize to [-1, 1]
    return frame


def load_image(image_path: str) -> torch.Tensor:
    """
    Load image from path and convert to RGB tensor

    Args:
        image_path (str): Path to the image

    Returns:
        torch.Tensor: RGB image tensor of shape (C, H, W)
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32)


@torch.no_grad()
def main(model_name: str, args):
    # Instantiate the model
    model_config_cls = NAME_TO_MODEL_CONFIG[model_name]["model_config"]
    model_cls = NAME_TO_MODEL_CONFIG[model_name]["model"]
    model_config = model_config_cls.from_args(args)
    model = model_cls(**model_config.to_dict())
    model = load_weights(model, model_config.weights, strict_load=model_config.strict_load).cuda()
    model.eval()
    logger.info("Model is loaded successfully!")

    # Load files
    input_files = [args.input_path]

    for image_path in tqdm(input_files):
        # Load and preprocess image
        image = load_image(image_path)
        original_image = image.clone()

        # Preprocess for model input
        input_frame = preprocess_frame(image, args.HW).cuda()

        # Get model output
        depth_output = model(input_frame)

        # Format output into Depth object
        # Create a dummy valid mask (all pixels valid for inference)
        valid_mask = torch.ones_like(depth_output, dtype=torch.bool)

        # Convert to CPU and ensure proper dtype for visualization
        depth_pred = depth_output[0].cpu().to(torch.float32)
        valid_mask = valid_mask[0].cpu()

        depth_obj = Depth(depth_pred, valid_mask)
        depth_view = depth_obj.get_view(cmap=args.cmap)

        # Resize for visualization
        depth_view = cv2.resize(depth_view, (args.viz_hw[1], args.viz_hw[0]))

        # Prepare original image for visualization
        original_image = original_image.permute(1, 2, 0).numpy().astype(np.uint8)
        original_image = cv2.resize(original_image, (args.viz_hw[1], args.viz_hw[0]))

        # Concatenate original image and depth prediction
        frame_view = np.concatenate([original_image, depth_view], axis=1)
        frame_view = cv2.cvtColor(frame_view, cv2.COLOR_RGB2BGR)

        output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_depth.png"
        cv2.imwrite(os.path.join(args.output, output_filename), frame_view)

    logger.info(f"Inference completed! Results saved to {args.output}")


if __name__ == "__main__":
    model_name = sys.argv[1]
    assert (
        model_name in NAME_TO_MODEL_CONFIG
    ), f"Model {model_name} not found. Available models: {NAME_TO_MODEL_CONFIG.keys()}"

    args = parse_args(model_name)
    main(model_name, args)
