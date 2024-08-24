import os
from typing import Union, Optional, Tuple
from loguru import logger


class BaseTrainer(object):
    def __init__(
        self,
        workdir: str,
        project_name: str,
        experiment_name: str,
        val_interval: Union[float, int] = 1.0,
        log_interval: int = 100,
        num_val_samples: int = -1,
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
            num_val_samples (int): number of samples during evaluation.
                Useful to limit the number of samples during evaluation. Defaults to -1 (all samples)
            save_best_k_cp (int): number of best checkpoints to save
        """
        assert isinstance(val_interval, int) or (
            isinstance(val_interval, float) and val_interval <= 1
        ), "val_interval must be either int or float <= 1"
        assert log_interval > 0, "log_interval must be greater than 0"
        assert (
            save_best_k_cp == -1 or save_best_k_cp > 0
        ), "save_best_k_cp must be -1 (save all checkpoints) or greater than 0"

        self.workdir = workdir
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.artifact_dir = os.path.join(self.workdir, self.project_name, self.experiment_name)
        self.val_interval = val_interval
        self.log_interval = log_interval
        self.num_val_samples = num_val_samples
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
        if os.path.exists(os.path.join(self.workdir, self.project_name, self.experiment_name)):
            logger.warning(
                f"Experiment {os.path.join(self.workdir, self.project_name, self.experiment_name)} already exists!."
            )
            logger.warning(
                "You may overwrite the existing experiment. Ignore this message if you are resuming the run."
            )
        os.makedirs(os.path.join(self.workdir, self.project_name, self.experiment_name), exist_ok=True)

    def get_topk_checkpoint_name(self, epoch: int, steps: int, metric: float, metric_name: str) -> str:
        """
        Get the name of the checkpoint

        Args:
            epoch (int): epoch number
            steps (int): step number
            metric (float): metric value

        Returns:
            str: checkpoint name
        """
        return f"epoch={epoch}_steps={steps}_{metric_name}={metric:.4f}.pth"

    def get_latest_checkpoint_name(self, steps: int) -> str:
        """
        Get the name of the latest checkpoint

        Args:
            steps (int): step number

        Returns:
            str: checkpoint name
        """
        return f"latest_steps={steps}.pth"

    @staticmethod
    def get_latest_checkpoint_from_dir(dir_path: str):
        """
        Get the last checkpoint from the directory

        Args:
            dir_path (str): path to the directory

        Returns:
            str: path to the last checkpoint
        """
        assert os.path.isdir(dir_path), f"{dir_path} is not a directory"

        checkpoints = [f for f in os.listdir(dir_path) if f.endswith(".pth")]
        for cp in checkpoints:
            if "latest" in cp:
                return cp
        logger.info("No latest checkpoint found. Latest checkpoint must have the format `latest_steps-<steps>.pth`")
        return None

    @staticmethod
    def get_best_checkpoint_from_dir(dir_path: str, condition: str = "max"):
        """
        Get the best checkpoint from the directory

        Args:
            dir_path (str): path to the directory
            condition (str): condition to select the best checkpoint. `max` or `min`. Defaults to `max`

        Returns:
            str: path to the best checkpoint
        """
        assert condition in ["max", "min"], "condition must be either `max` or `min`"
        assert os.path.isdir(dir_path), f"{dir_path} is not a directory"

        checkpoints = [f for f in os.listdir(dir_path) if f.endswith(".pth") and "latest" not in f]
        if not checkpoints:
            return None
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))
        if condition == "max":
            return checkpoints[-1]
        else:
            return checkpoints[0]

    def load_state(self, cp_path: str):
        """
        Load state of Trainer

        Args:
            dir_path (str): path to the directory
        """
        assert os.path.exists(cp_path) and cp_path.endswith(".pth"), f"{cp_path} must be a `.pth` file"

        checkpoint_infos = []
        file_name = os.path.basename(cp_path)
        dir_path = os.path.dirname(cp_path)
        if "latest" in file_name:
            # Load from latest checkpoint
            self.total_steps = int(file_name.replace(".pth", "").split("_")[1].split("=")[1])
            for f in os.listdir(dir_path):
                if f.endswith(".pth") and "latest" not in f:
                    epoch = int(f.split("_")[0].split("=")[1])
                    steps = int(f.split("_")[1].split("=")[1])
                    metric = float(f.split("_")[2].split("=")[1].split(".")[0])
                    checkpoint_infos.append({"epoch": epoch, "step": steps, "metric": metric})
        else:
            logger.info("Not loading from latest checkpoint but from a specific checkpoint.")
            # Load from a specific checkpoint
            self.total_steps = int(file_name.split("_")[1].split("=")[1])
            for f in os.listdir(dir_path):
                if f.endswith(".pth") and "latest" not in f:
                    steps = int(f.split("_")[1].split("=")[1])
                    if steps > self.total_steps:
                        continue
                    epoch = int(f.split("_")[0].split("=")[1])
                    metric = float(f.split("_")[2].split("=")[1].split(".")[0])
                    checkpoint_infos.append({"epoch": epoch, "step": steps, "metric": metric})

        checkpoint_infos = sorted(checkpoint_infos, key=lambda x: x["metric"])
        self.checkpoint_infos = checkpoint_infos
        logger.info(f"State of Trainer loaded from : {cp_path}")
        logger.info(f"Total steps: {self.total_steps}")
        logger.info(f"Checkpoint infos: {self.checkpoint_infos}")

    def is_topk_checkpoint(
        self, epoch: int, steps: int, metric: float, metric_name: str, condition: str = "min"
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Save checkpoint

        Args:
            epoch (int): epoch number
            steps (int): step number
            metric (float): metric value
            metric_name (str): name of the metric
            condition (str): condition to select the best checkpoint. `max` or `min`. Defaults to `min`

        Returns:
            Tuple[Optional[str], Optional[str]]: new checkpoint path, replaced checkpoint path
        """
        assert condition in ["max", "min"], "condition must be either `max` or `min`"

        new_cp_dir = None  # Path to save the new checkpoint
        replaced_cp_dir = None  # This will be useful to remove the old checkpoint

        if self.save_best_k_cp == -1 or len(self.checkpoint_infos) < self.save_best_k_cp:
            self.checkpoint_infos.append(
                {"epoch": epoch, "steps": steps, "metric": metric, "metric_name": metric_name}
            )
            self.checkpoint_infos = sorted(self.checkpoint_infos, key=lambda x: x["metric"])
            new_cp_dir = self.get_topk_checkpoint_name(epoch, steps, metric, metric_name)

        else:
            replaced_cp = None
            if condition == "min" and metric < self.checkpoint_infos[-1]["metric"]:
                replaced_cp = self.checkpoint_infos.pop(-1)
            elif condition == "max" and metric > self.checkpoint_infos[0]["metric"]:
                replaced_cp = self.checkpoint_infos.pop(0)
            if replaced_cp is not None:
                self.checkpoint_infos.append(
                    {
                        "epoch": epoch,
                        "steps": self.total_steps,
                        "metric": metric,
                        "metric_name": metric_name,
                    }
                )
                self.checkpoint_infos = sorted(self.checkpoint_infos, key=lambda x: x["metric"])
                new_cp_dir = self.get_topk_checkpoint_name(epoch, self.total_steps, metric, metric_name)
                replaced_cp_dir = self.get_topk_checkpoint_name(
                    replaced_cp["epoch"], replaced_cp["steps"], replaced_cp["metric"], replaced_cp["metric_name"]
                )

        return new_cp_dir, replaced_cp_dir

    def train(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in child class.")

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in child class.")
