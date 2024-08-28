from argparse import ArgumentParser
import torch
import yaml
import importlib
from typing import Tuple, Union
from loguru import logger

from nndepth.utils.base_trainer import BaseTrainer
from nndepth.utils.base_dataloader import BaseDataLoader


def add_common_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--model_config", required=True, help="Path to model config file")
    parser.add_argument("--data_config", required=True, help="Path to data config file")
    parser.add_argument("--training_config", required=True, help="Path to training config file")
    parser.add_argument("--resume_from_checkpoint", default=None, help="Path to checkpoint to resume.")
    return parser


def load_weights(model: torch.nn.Module, weights: str, strict_load: bool = True, device=torch.device("cpu")):
    state_dict = torch.load(weights, map_location=device)
    checkpoints = {}
    for k, v in state_dict.items():
        if k[:6] == "model.":
            key = k[6:]
        else:
            key = k
        checkpoints[key] = v
    model.load_state_dict(checkpoints, strict=strict_load)
    logger.info(f"Loaded weights from {weights}")
    return model


def instantiate_with_config_file(
    config_file: str, module_path: str, cls_name: str = None
) -> Tuple[Union[torch.nn.Module, BaseTrainer, BaseDataLoader], dict]:
    """
    Instantiate an object from the config file.

    Parameters
        config_file (str): Path to the config file.
        module_path (str): Path to the module containing the class.

    Returns:
            Tuple[Type, dict]: The instantiated object and the config dictionary.
    """
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if "name" not in config and cls_name is None:
        raise RuntimeError("cls_name must be set or config file must contain a 'name' field which is the class name.")
    cls_name = cls_name if cls_name is not None else config.pop("name")

    # load the class from the module
    if "/" in module_path:
        module_path = module_path.replace("/", ".")
    module = importlib.import_module(module_path)
    cls_type = getattr(module, cls_name, None)
    if cls_type is None:
        raise RuntimeError(f"Class {cls_name} not found in module {module_path}.")

    return cls_type(**config), config
