import os
import argparse
import sys
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from loguru import logger
from typing import Union

from nndepth.data.dataloaders import TartanairDisparityDataLoader
from nndepth.models.raft_stereo import RAFTTrainer
from nndepth.models.raft_stereo.configs import BaseRAFTTrainingConfig, RepViTRAFTStereoTrainingConfig
from nndepth.models.raft_stereo.model import BaseRAFTStereo, Coarse2FineGroupRepViTRAFTStereo


def main():
    NAME_TO_CONFIGS_MODEL = {
        "base": {
            "training_config": BaseRAFTTrainingConfig,
            "model": BaseRAFTStereo,
        },
        "repvit": {
            "training_config": RepViTRAFTStereoTrainingConfig,
            "model": Coarse2FineGroupRepViTRAFTStereo,
        },
    }
    model_name = sys.argv[1]
    assert (
        model_name in NAME_TO_CONFIGS_MODEL
    ), f"Model {model_name} not found. Available models: {NAME_TO_CONFIGS_MODEL.keys()}"

    training_config_cls = NAME_TO_CONFIGS_MODEL[model_name]["training_config"]
    model_cls = NAME_TO_CONFIGS_MODEL[model_name]["model"]

    parser = argparse.ArgumentParser()
    training_config_cls.add_args(parser)
    parser.add_argument("--compile", action="store_true", help="Compile the model")
    parser.add_argument("--save_config", default=None, help="Save the training config to a file")
    args = parser.parse_args(sys.argv[2:])

    # Parse training config
    training_config = training_config_cls.from_args(args)

    if args.save_config:
        training_config.save(args.save_config)
        logger.info(f"Training config saved to {args.save_config}")
        return

    # Init data loader
    dataloader = TartanairDisparityDataLoader(**training_config.data.to_dict())
    dataloader.setup()

    # Init distributed training
    model: nn.Module = model_cls(**training_config.model.to_dict())
    model.cuda()

    with_ddp = torch.cuda.device_count() > 1
    if with_ddp:
        init_process_group(backend="nccl")
        ddp_local_rank = os.environ.get("LOCAL_RANK", 0)
        torch.cuda.set_device(int(ddp_local_rank))
        model = DDP(model, device_ids=[int(ddp_local_rank)])
        if args.compile:
            logger.info("Compiling model ...")
            model.compile(model)
            logger.info("Model compiled !")

        logger.info(f"Model DDP initialized on device {ddp_local_rank}")
    else:
        if args.compile:
            logger.info("Compiling model ...")
            model.compile(model)
            logger.info("Model compiled !")

    # Init trainer
    trainer = RAFTTrainer(model, **training_config.trainer.to_dict(), training_config=training_config.to_dict())

    # Train the model
    trainer.train(model, dataloader.train_dataloader, dataloader.val_dataloader)

    if with_ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
