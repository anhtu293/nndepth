from argparse import ArgumentParser
import torch
from loguru import logger


def add_common_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--resume_from_checkpoint", default=None, help="Path to checkpoint to resume.")
    parser.add_argument("--compile", action="store_true", help="Compile the model.")
    return parser


def load_weights(model: torch.nn.Module, weights: str, strict_load: bool = True, device=torch.device("cpu")):
    state_dict = torch.load(weights, map_location=device)
    model.load_state_dict(state_dict, strict=strict_load)
    logger.info(f"Loaded weights from {weights}")
    return model
