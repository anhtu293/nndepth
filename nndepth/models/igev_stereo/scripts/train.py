import os
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from loguru import logger

from nndepth.data.dataloaders import TartanairDisparityDataLoader
from nndepth.models.igev_stereo.configs import MBNetIGEVStereoTrainingConfig
from nndepth.models.igev_stereo.model import IGEVStereoMBNet
from nndepth.models.igev_stereo.igev_trainer import IGEVStereoTrainer


def main():
    parser = argparse.ArgumentParser()
    MBNetIGEVStereoTrainingConfig.add_args(parser)
    parser.add_argument("--compile", action="store_true", help="Compile the model")
    parser.add_argument("--save_config", default=None, help="Save the training config to a file")
    args = parser.parse_args()

    # Parse training config
    training_config: MBNetIGEVStereoTrainingConfig = MBNetIGEVStereoTrainingConfig.from_args(args)

    if args.save_config:
        training_config.save(args.save_config)
        logger.info(f"Training config saved to {args.save_config}")
        return

    # Init data loader
    dataloader = TartanairDisparityDataLoader(**training_config.data.to_dict())
    dataloader.setup(stage="train")

    # Init model
    model = IGEVStereoMBNet(**training_config.model.to_dict())
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
    trainer = IGEVStereoTrainer(model, **training_config.trainer.to_dict(), training_config=training_config.to_dict())

    # Train the model
    trainer.train(model, dataloader.train_dataloader, dataloader.val_dataloader)

    if with_ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
