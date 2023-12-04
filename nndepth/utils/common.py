import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import os
import shutil


def load_weights(model: torch.nn.Module, weights: str, strict_load: bool = True, device=torch.device("cpu")):
    state_dict = torch.load(weights, map_location=device)["state_dict"]
    checkpoints = {}
    for k, v in state_dict.items():
        if k[:6] == "model.":
            key = k[6:]
        else:
            key = k
        checkpoints[key] = v
    model.load_state_dict(checkpoints, strict=strict_load)
    print(f"[INFO]: Loaded weights from {weights}")
    return model


class ConfigRegisterCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        log_dir = trainer.logger.save_dir

        # Copy config file to log dir
        model_file_name = pl_module.args.model_config.split("/")[-1]
        if pl_module.args.model_config != os.path.join(log_dir, model_file_name):
            shutil.copyfile(pl_module.args.model_config, os.path.join(log_dir, model_file_name))
        if pl_module.args.data_config != os.path.join(log_dir, "data_config.yaml"):
            shutil.copyfile(pl_module.args.data_config, os.path.join(log_dir, "data_config.yaml"))

        print(f"Copied config files to {log_dir} !")
