import os
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributed import init_process_group, destroy_process_group, get_world_size
from loguru import logger

from nndepth.data.dataloaders import TartanairDisparityDataLoader
from nndepth.utils import add_common_args
from nndepth.utils.trackers.wandb import WandbTracker
from nndepth.utils.distributed_training import is_dist_initialized
from nndepth.models.igev_stereo import STEREO_MODELS, IGEVStereoLoss, IGEVStereoTrainer


def main():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        choices=STEREO_MODELS.keys(),
        help="Name of the model to train",
    )
    args = parser.parse_args()

    model, model_config = STEREO_MODELS[args.model_name].init_from_config(args.model_config)
    dataloader, data_config = TartanairDisparityDataLoader.init_from_config(args.data_config)
    trainer, training_config = IGEVStereoTrainer.init_from_config(args.training_config)

    # Init data loader
    dataloader.setup()

    # Init loss, optimizer, scheduler
    criterion = IGEVStereoLoss(gamma=0.8, max_flow=1000)
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
    grouped_configs = {"model": model_config, "data": data_config, "training": training_config}
    wandb_tracker = WandbTracker(
        project_name=trainer.project_name,
        run_name=trainer.experiment_name,
        root_log_dir=trainer.artifact_dir,
        config=grouped_configs,
    )

    # Resume from checkpoint if required
    if args.resume_from_checkpoint is not None:
        trainer.resume_from_checkpoint(args.resume_from_checkpoint, model, optimizer, scheduler)

    if is_dist_initialized():
        init_process_group(backend="nccl")
        ddp_local_rank = os.environ.get("LOCAL_RANK", 0)
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(int(ddp_local_rank))
        model = model.to(device)
        if args.compile:
            logger.info("Compiling model ...")
            model = torch.compile(model)
            logger.info("Model compiled !")

        model = torch.nn.parallel.DistributedDataParallel(model, devices_ids=[ddp_local_rank])
        logger.info(f"Model DDP initialized on device {ddp_local_rank}")

        if training_config["max_steps"] is not None:
            assert (
                training_config["max_steps"] % get_world_size() == 0
            ), "max_steps must be divisible by the number of GPUs"
            training_config["max_steps"] = training_config["max_steps"] // get_world_size()
    else:
        model = model.to(torch.device("cuda"))
        device = "cuda:0"
        if args.compile:
            logger.info("Compiling model ...")
            model = torch.compile(model)
            logger.info("Model compiled !")

    # Gradient scaler
    scaler = torch.amp.GradScaler(device="cuda", enabled=(trainer.dtype == torch.bfloat16))

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

    if is_dist_initialized():
        destroy_process_group()


if __name__ == "__main__":
    main()
