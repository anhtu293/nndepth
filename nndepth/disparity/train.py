import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from nndepth.utils.common import add_common_args, instantiate_with_config_file
from nndepth.utils.trackers.wandb import WandbTracker

from nndepth.disparity.criterions import RAFTCriterion


def main():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args()

    # Instantiate the model, dataloader and trainer
    model, model_config = instantiate_with_config_file(args.model_config, "nndepth.disparity.models")
    dataloader, data_config = instantiate_with_config_file(args.data_config, "nndepth.disparity.data_loaders")
    trainer, training_config = instantiate_with_config_file(args.training_config, "nndepth.disparity.trainers")

    # Init the criterion, optimizer, scheduler
    criterion = RAFTCriterion(gamma=0.8, max_flow=1000)
    optimizer = optim.AdamW(model.parameters(), lr=trainer.lr, weight_decay=trainer.weight_decay, eps=trainer.epsilon)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=trainer.lr,
        total_steps=trainer.total_steps + 100,
        anneal_strategy="linear",
        pct_start=0.05,
        cycle_momentum=False,
    )

    # Setup the tracker
    grouped_configs = {"model": model_config, "data": data_config, "training": training_config}
    wandb_tracker = WandbTracker(
        project_name=trainer.project_name,
        run_name=trainer.experiment_name,
        root_log_dir=trainer.artifact_dir,
        config=grouped_configs,
        resume=args.resume_from_checkpoint is not None,
    )

    # Prepare the trainer
    trainer.prepare(model, dataloader.train_dataloader, dataloader.val_dataloader, optimizer, scheduler)

    # Resume from checkpoint if required
    if args.resume_from_checkpoint is not None:
        trainer.resume_from_checkpoint(args.resume_from_checkpoint)

    # Train the model
    trainer.train(
        model=model,
        criterion=criterion,
        train_dataloader=dataloader.train_dataloader,
        val_dataloader=dataloader.val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        tracker=wandb_tracker,
    )


if __name__ == "__main__":
    main()
