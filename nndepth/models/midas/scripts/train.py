import argparse
from argparse import Namespace
import sys
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger

from nndepth.models.midas.config import MBNetV3MidasTrainingConfig

from nndepth.models.midas.models import MobileNetV3DepthModel
from nndepth.models.midas.trainer import MiDasTrainer
from nndepth.data.dataloaders.depth import MultiDatasetsDepthDataLoader


MODEL_CONFIG_CLS = {
    "mbnet_v3": MBNetV3MidasTrainingConfig,
}
MODEL_CLS = {
    "mbnet_v3": MobileNetV3DepthModel,
}


def parse_args(model_name: str):
    parser = argparse.ArgumentParser()
    parser = MODEL_CONFIG_CLS[model_name].add_args(parser)
    parser.add_argument("--compile", action="store_true", help="Whether to compile the model")
    parser.add_argument("--overfit", action="store_true", help="Whether to overfit the model")
    parser.add_argument("--save_config", default=None, help="Path to save the config")

    args = parser.parse_args(sys.argv[2:])
    return args


def main(args: Namespace, model_name: str):
    training_config = MODEL_CONFIG_CLS[model_name].from_args(args)

    if args.save_config is not None:
        training_config.save(args.save_config)
        logger.info(f"Config saved to {args.save_config}")
        exit()

    with_ddp = torch.cuda.device_count() > 2
    if with_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(os.environ["LOCAL_RANK"])
        logger.info(f"Using DDP with {torch.cuda.device_count()} GPUs")

    # Setup data
    data_module = MultiDatasetsDepthDataLoader(**training_config.data.to_dict())
    data_module.setup("train")

    # Setup model
    model = MODEL_CLS[model_name](**training_config.model.to_dict())
    model.cuda()
    if with_ddp:
        model = DDP(model, device_ids=[os.environ["LOCAL_RANK"]])

    if args.compile:
        model = torch.compile(model)

    # Setup trainer
    trainer = MiDasTrainer(model, **training_config.trainer.to_dict())

    # Train
    if args.overfit:
        trainer.train(data_module.val_dataloader, data_module.val_dataloader)
    else:
        trainer.train(data_module.train_dataloader, data_module.val_dataloader)


if __name__ == "__main__":
    model_name = sys.argv[1]
    assert model_name in MODEL_CONFIG_CLS, f"Model {model_name} not found"
    args = parse_args(model_name=model_name)
    main(args, model_name=model_name)
