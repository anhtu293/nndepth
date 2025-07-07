import os
import argparse
import sys
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributed import init_process_group, destroy_process_group, get_world_size
from loguru import logger
from typing import Union

from nndepth.data.dataloaders import TartanairDisparityDataLoader
from nndepth.utils.trackers.wandb import WandbTracker
from nndepth.utils.distributed_training import is_distributed_training
from nndepth.models.raft_stereo import RAFTLoss, RAFTTrainer
from nndepth.models.raft_stereo.configs import BaseRAFTTrainingConfig, RepViTRAFTStereoTrainingConfig
from nndepth.models.raft_stereo.model import BaseRAFTStereo, Coarse2FineGroupRepViTRAFTStereo


def main():

    TRAINING_CONFIGS = {
        "base_raft_stereo": BaseRAFTTrainingConfig,
        "coarse2fine_group_repvit_raft_stereo": RepViTRAFTStereoTrainingConfig,
    }
    MODEL_CONFIGS = {
        "base_raft_stereo": BaseRAFTStereo,
        "coarse2fine_group_repvit_raft_stereo": Coarse2FineGroupRepViTRAFTStereo,
    }
    model_name = sys.argv[1]
    assert model_name in TRAINING_CONFIGS, f"Model {model_name} not found"
    training_config = TRAINING_CONFIGS[model_name]

    parser = argparse.ArgumentParser()
    training_config.add_args(parser)
    args = parser.parse_args()

    training_config: Union[BaseRAFTTrainingConfig, RepViTRAFTStereoTrainingConfig] = training_config.from_args(args)

    dataloader = TartanairDisparityDataLoader(**training_config.data.to_dict())
    trainer = RAFTTrainer(**training_config.trainer.to_dict())
    model = MODEL_CONFIGS[model_name](**training_config.model.to_dict())


    # Init data loader
    dataloader.setup()

    # Init loss, optimizer, scheduler
    criterion = RAFTLoss(gamma=0.8, max_flow=1000)
    optimizer = optim.AdamW(model.parameters(), lr=trainer.lr, weight_decay=trainer.weight_decay, eps=trainer.epsilon)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=trainer.lr,
        total_steps=trainer.max_steps + 100,
        anneal_strategy="linear",
        pct_start=0.05,
        cycle_momentum=False,
    )

    # Init tracker
    grouped_configs = {"model": training_config.model.to_dict(), "data": training_config.data.to_dict(), "training": training_config.trainer.to_dict()}
    wandb_tracker = WandbTracker(
        project_name=training_config.trainer.project_name,
        run_name=training_config.trainer.experiment_name,
        root_log_dir=training_config.trainer.workdir,
        config=grouped_configs,
    )

    # Resume from checkpoint if required
    if args.resume_from_checkpoint is not None:
        trainer.resume_from_checkpoint(args.resume_from_checkpoint, model, optimizer, scheduler)

    if is_distributed_training():
        init_process_group(backend="nccl")
        ddp_local_rank = os.environ.get("LOCAL_RANK", 0)
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(int(ddp_local_rank))
        model = model.to(device)
        if args.compile:
            logger.info("Compiling model ...")
            model = torch.compile(model)
            logger.info("Model compiled !")

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(ddp_local_rank)])
        logger.info(f"Model DDP initialized on device {ddp_local_rank}")

        if training_config.trainer.max_steps is not None:
            assert (
                training_config.trainer.max_steps % get_world_size() == 0
            ), "max_steps must be divisible by the number of GPUs"
            training_config.trainer.max_steps = training_config.trainer.max_steps // get_world_size()
    else:
        model = model.to(torch.device("cuda"))
        device = "cuda:0"
        if args.compile:
            logger.info("Compiling model ...")
            model = torch.compile(model)
            logger.info("Model compiled !")

    # Gradient scaler
    scaler = torch.amp.GradScaler(device=device, enabled=(trainer.dtype == torch.bfloat16))

    # Train the model
    trainer.train(
        model,
        criterion,
        dataloader.train_dataloader,
        dataloader.val_dataloader,
        optimizer,
        scheduler,
        scaler,
        wandb_tracker,
        device,
    )

    if is_distributed_training():
        destroy_process_group()


if __name__ == "__main__":
    main()
