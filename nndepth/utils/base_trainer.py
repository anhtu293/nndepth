import os
import datetime
from typing import Union
from loguru import logger


class BaseTrainer(object):
    def __init__(
        self,
        workdir: str,
        project_name: str,
        experiment_name: str,
        val_interval: Union[float, int] = 1,
        log_interval: int = 100,
        save_best_k_cp: int = 3,
    ):
        """
        Base class for all trainers

        Args:
            workdir (str): path to save the experiment
            project_name (str): name of the project
            experiment_name (str): name of the experiment
            val_interval (Union[float, int]): interval to validate
            log_interval (int): interval to log
            save_best_k_cp (int): number of best checkpoints to save
        """
        assert isinstance(val_interval, int) or (
            isinstance(val_interval, float) and val_interval <= 1
        ), "val_interval must be either int or float <= 1"
        assert log_interval > 0, "log_interval must be greater than 0"
        assert save_best_k_cp > 0, "save_best_k_cp must be greater than 0"

        self.workdir = workdir
        self.project_name = project_name
        self.experiment_name = "{}_{:%B-%d-%Y-%Hh-%M}".format(experiment_name, datetime.datetime.now())
        self.val_interval = val_interval
        self.log_interval = log_interval
        self.save_best_k_cp = save_best_k_cp
        self.checkpoint_infos = []
        self.total_steps = 0

        self.setup_workdir()

    def setup_workdir(self):
        """
        Setup directories for the experiment
        """
        os.makedirs(self.workdir, exist_ok=True)
        os.makedirs(os.path.join(self.workdir, self.project_name), exist_ok=True)
        os.makedirs(os.path.join(self.workdir, self.experiment_name), exist_ok=True)

    def get_checkpoint_name(self, epoch: int, steps: int, metric: float, metric_name: str) -> str:
        """
        Get the name of the checkpoint

        Args:
            epoch (int): epoch number
            steps (int): step number
            metric (float): metric value

        Returns:
            str: checkpoint name
        """
        return f"epoch-{epoch}_steps-{steps}_{metric_name}-{metric:.4f}.pth"

    @staticmethod
    def get_last_checkpoint_from_dir(dir_path: str):
        """
        Get the last checkpoint from the directory

        Args:
            dir_path (str): path to the directory

        Returns:
            str: path to the last checkpoint
        """
        checkpoints = [f for f in os.listdir(dir_path) if f.endswith(".pth")]
        if not checkpoints:
            return None
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[1].split("-")[1]))
        return os.path.join(dir_path, checkpoints[-1])

    def load_state(self, dir_path: str):
        """
        Load state of Trainer

        Args:
            dir_path (str): path to the directory
        """
        if dir_path.endswith(".pth"):
            dir_path = dir_path.replace(os.path.basename(dir_path), "")
        checkpoint_infos = []
        latest = 0
        for f in os.listdir(dir_path):
            if f.endswith(".pth"):
                epoch = int(f.split("_")[0].split("-")[1])
                steps = int(f.split("_")[1].split("-")[1])
                if steps > latest:
                    latest = steps
                metric = float(f.split("_")[2].split("-")[1].split(".")[0])
                checkpoint_infos.append({"epoch": epoch, "step": steps, "metric": metric})
        checkpoint_infos = sorted(checkpoint_infos, key=lambda x: x["metric"], reverse=True)
        self.checkpoint_infos = checkpoint_infos
        self.total_steps = latest
        logger.info("Trainer's state loaded!")

    def train(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in child class.")

    def validate(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in child class.")
