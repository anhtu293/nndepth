import torch
from safetensors.torch import load_file
from loguru import logger

from nndepth.utils.distributed_training import is_dist_initialized


def load_weights(model: torch.nn.Module, weights: str, strict_load: bool = True, device=torch.device("cpu")):
    if weights.endswith(".safetensors"):
        state_dict = load_file(weights, device=device)
    else:
        state_dict = torch.load(weights, map_location=device)
    model.load_state_dict(state_dict, strict=strict_load)
    logger.info(f"Loaded weights from {weights}")
    return model


def get_model_state_dict(model: torch.nn.Module):
    if is_dist_initialized():
        model_state_dict = getattr(model.module, '_orig_mod', model.module).state_dict()
    else:
        model_state_dict = getattr(model, '_orig_mod', model).state_dict()
    return model_state_dict
