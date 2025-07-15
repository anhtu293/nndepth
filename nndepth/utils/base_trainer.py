import os
import json
from datetime import datetime
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, OrderedDict
from loguru import logger
import numpy as np

from nndepth.utils.distributed_training import run_on_main_process
from nndepth.utils.trackers import WandbTracker


class BaseTrainer(ABC):
    def __init__(
        self,
        workdir: str,
        project_name: str,
        experiment_name: str,
        num_epochs: Optional[int] = 100,
        max_steps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        val_interval: Union[float, int] = 1.0,
        log_interval: int = 100,
        num_val_samples: Optional[int] = None,
        save_best_k_cp: Optional[int] = None,
        tracker: Optional[str] = None,
        checkpoint: Optional[str] = None,
        resume: bool = False,
        training_config: Optional[dict] = None,
    ):
        """
        Base class for all trainers

        Args:
            workdir (str): path to save the experiment
            project_name (str): name of the project
            experiment_name (str): name of the experiment
            num_epochs (Optional[int]): number of epochs to train
            max_steps (Optional[int]): number of steps to train
            gradient_accumulation_steps (int): number of steps to accumulate gradients
            val_interval (Union[float, int]): interval to validate
            log_interval (int): interval to log
            num_val_samples (Optional[int]): number of samples during evaluation.
                Useful to limit the number of samples during evaluation. Defaults to None (all samples)
            save_best_k_cp (Optional[int]): number of best checkpoints to save
            tracker (Optional[str]): name of the tracker to use
            checkpoint (str): path to the checkpoint to resume from
            resume (bool): whether to resume from the checkpoint
            training_config (Optional[dict]): training config to log with the tracker
        """
        assert isinstance(val_interval, int) or (
            isinstance(val_interval, float) and val_interval <= 1
        ), "val_interval must be either int or float <= 1"
        assert log_interval > 0, "log_interval must be greater than 0"
        assert (
            save_best_k_cp is None or save_best_k_cp > 0
        ), "save_best_k_cp must be None (save all checkpoints) or greater than 0"

        self._workdir = workdir
        self._project_name = project_name
        self._experiment_name = experiment_name + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self._artifact_dir = os.path.join(self._workdir, self._project_name, self._experiment_name)
        self._num_epochs = num_epochs
        self._max_steps = max_steps
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._val_interval = val_interval
        self._log_interval = log_interval
        self._num_val_samples = num_val_samples
        self._save_best_k_cp = save_best_k_cp
        self._checkpoint_infos = []
        self._current_steps = 0
        self._checkpoint = checkpoint
        self._resume = resume
        self._training_config = training_config
        self._current_epoch = 0
        self._current_step = 0
        self.setup_workdir()
        if tracker is not None:
            self.setup_tracker(tracker)
        else:
            self._tracker = None

    @property
    def workdir(self) -> str:
        """
        Get the workdir path

        Returns:
            str: workdir
        """
        return self._workdir

    @workdir.setter
    def workdir(self, value: str):
        """
        Set the workdir path

        Args:
            value (str): workdir path
        """
        self._workdir = value

    @property
    def project_name(self) -> str:
        """
        Get the project name

        Returns:
            str: project name
        """
        return self._project_name

    @project_name.setter
    def project_name(self, value: str):
        """
        Set the project name

        Args:
            value (str): project name
        """
        self._project_name = value

    @property
    def experiment_name(self) -> str:
        """
        Get the experiment name

        Returns:
            str: experiment name
        """
        return self._experiment_name

    @experiment_name.setter
    def experiment_name(self, value: str):
        """
        Set the experiment name

        Args:
            value (str): experiment name
        """
        self._experiment_name = value

    @property
    def artifact_dir(self) -> str:
        """
        Get the artifact directory

        Returns:
            str: artifact directory
        """
        return self._artifact_dir

    @artifact_dir.setter
    def artifact_dir(self, value: str):
        """
        Set the artifact directory

        Args:
            value (str): artifact directory
        """
        self._artifact_dir = value

    @property
    def num_epochs(self) -> Optional[int]:
        """
        Get the number of epochs
        """
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, value: Optional[int]):
        """
        Set the number of epochs
        """
        assert value is None or value > 0, "num_epochs must be None or greater than 0"
        self._num_epochs = value

    @property
    def max_steps(self) -> Optional[int]:
        """
        Get the maximum number of steps
        """
        return self._max_steps

    @max_steps.setter
    def max_steps(self, value: Optional[int]):
        """
        Set the maximum number of steps
        """
        assert value is None or value > 0, "max_steps must be None or greater than 0"
        self._max_steps = value

    @property
    def gradient_accumulation_steps(self) -> int:
        """
        Get the gradient accumulation steps
        """
        return self._gradient_accumulation_steps

    @gradient_accumulation_steps.setter
    def gradient_accumulation_steps(self, value: int):
        """
        Set the gradient accumulation steps
        """
        assert value > 0, "gradient_accumulation_steps must be greater than 0"
        self._gradient_accumulation_steps = value

    @property
    def val_interval(self) -> Union[float, int]:
        """
        Get the validation interval

        Returns:
            Union[float, int]: validation interval
        """
        return self._val_interval

    @val_interval.setter
    def val_interval(self, value: Union[float, int]):
        """
        Set the validation interval

        Args:
            value (Union[float, int]): validation interval
        """
        self._val_interval = value

    @property
    def log_interval(self) -> int:
        """
        Get the log interval

        Returns:
            int: log interval
        """
        return self._log_interval

    @log_interval.setter
    def log_interval(self, value: int):
        """
        Set the log interval

        Args:
            value (int): log interval
        """
        self._log_interval = value

    @property
    def num_val_samples(self) -> Optional[int]:
        """
        Get the number of validation samples

        Returns:
            int: number of validation samples
        """
        return self._num_val_samples

    @num_val_samples.setter
    def num_val_samples(self, value: Optional[int]):
        """
        Set the number of validation samples

        Args:
            value (int): number of validation samples
        """
        assert value is None or value > 0, "num_val_samples must be None or greater than 0"
        self._num_val_samples = value

    @property
    def save_best_k_cp(self) -> Optional[int]:
        """
        Get the number of best checkpoints to save

        Returns:
            int: number of best checkpoints to save
        """
        return self._save_best_k_cp

    @save_best_k_cp.setter
    def save_best_k_cp(self, value: Optional[int]):
        """
        Set the number of best checkpoints to save

        Args:
            value (int): number of best checkpoints to save
        """
        assert value is None or value > 0, "save_best_k_cp must be None or greater than 0"
        self._save_best_k_cp = value

    @property
    def checkpoint(self) -> Optional[str]:
        """
        Get the checkpoint path

        Returns:
            Optional[str]: checkpoint path
        """
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, value: Optional[str]):
        """
        Set the checkpoint path

        Args:
            value (str): checkpoint path
        """
        self._checkpoint = value

    @property
    def tracker(self) -> Optional[WandbTracker]:
        """
        Get the tracker
        """
        return self._tracker

    @property
    def resume(self) -> bool:
        """
        Get the resume flag
        """
        return self._resume

    @property
    def current_epoch(self) -> int:
        """
        Get the current epoch
        """
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value: int):
        """
        Set the current epoch
        """
        self._current_epoch = value

    @property
    def current_step(self) -> int:
        """
        Get the current step
        """
        return self._current_step

    @current_step.setter
    def current_step(self, value: int):
        """
        Set the current step
        """
        self._current_step = value

    def setup_workdir(self):
        """
        Setup directories for the experiment
        """
        os.makedirs(self._workdir, exist_ok=True)
        os.makedirs(os.path.join(self._workdir, self._project_name), exist_ok=True)
        if os.path.exists(os.path.join(self._workdir, self._project_name, self._experiment_name)):
            logger.warning(
                f"Experiment {os.path.join(self._workdir, self._project_name, self._experiment_name)} already exists!."
            )
            logger.warning(
                "You may overwrite the existing experiment. Ignore this message if you are resuming the run."
            )
        os.makedirs(os.path.join(self._workdir, self._project_name, self._experiment_name), exist_ok=True)

    def setup_tracker(self, tracker: Optional[str]):
        """
        Setup the tracker
        """
        if tracker is None:
            return
        if tracker == "wandb":
            self._tracker = WandbTracker(
                project_name=self._project_name,
                run_name=self._experiment_name,
                root_log_dir=self._workdir,
                config=self._training_config if self._training_config is not None else None,
            )
        else:
            raise ValueError(f"Tracker {tracker} not supported")

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
        return f"epoch={epoch}_steps={steps}_{metric_name}={metric:.4f}"

    def get_latest_checkpoint_name(self, steps: int) -> str:
        """
        Get the name of the latest checkpoint

        Args:
            steps (int): step number

        Returns:
            str: checkpoint name
        """
        return f"latest_steps={steps}"

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

        checkpoints = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
        for cp in checkpoints:
            if "latest" in cp:
                return cp
        logger.info("No latest checkpoint found. Latest checkpoint must have the format `latest_steps-<steps>`")
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

        checkpoints = [
            f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f)) and "latest" not in f
        ]
        if not checkpoints:
            return None
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))
        if condition == "max":
            return checkpoints[-1]
        else:
            return checkpoints[0]

    def load_state(self, cp_path: str):
        """
        Load state of Trainer from the checkpoint

        Args:
            dir_path (str): path to the directory
        """
        cp_path = cp_path.rstrip("/")
        assert os.path.exists(cp_path), f"{cp_path} does not exist"

        dir_path = os.path.dirname(cp_path)
        state_file = os.path.join(dir_path, "state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
                self._current_epoch = state["current_epoch"]
                self._current_step = state["current_step"]
                self._checkpoint_infos = state["checkpoint_infos"]

        logger.info(f"State of Trainer loaded from : {cp_path}")
        logger.info(f"Current epoch: {self._current_epoch}")
        logger.info(f"Current step: {self._current_step}")
        logger.info(f"Checkpoint infos: {self._checkpoint_infos}")

    def is_topk_checkpoint(
        self, metric: float, metric_name: str, condition: str = "min"
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Save checkpoint

        Args:
            metric (float): metric value
            metric_name (str): name of the metric
            condition (str): condition to select the best checkpoint. `max` or `min`. Defaults to `min`

        Returns:
            Tuple[Optional[str], Optional[str]]: new checkpoint path, replaced checkpoint path
        """
        assert condition in ["max", "min"], "condition must be either `max` or `min`"

        new_cp_dir = None  # Path to save the new checkpoint
        replaced_cp_dir = None  # This will be useful to remove the old checkpoint

        if self._save_best_k_cp is None or len(self._checkpoint_infos) < self._save_best_k_cp:
            self._checkpoint_infos.append(
                {
                    "epoch": self._current_epoch,
                    "steps": self._current_step,
                    "metric": metric,
                    "metric_name": metric_name,
                }
            )
            self._checkpoint_infos = sorted(self._checkpoint_infos, key=lambda x: x["metric"])
            new_cp_dir = self.get_topk_checkpoint_name(self._current_epoch, self._current_step, metric, metric_name)

        else:
            replaced_cp = None
            if condition == "min" and metric < self._checkpoint_infos[-1]["metric"]:
                replaced_cp = self._checkpoint_infos.pop(-1)
            elif condition == "max" and metric > self._checkpoint_infos[0]["metric"]:
                replaced_cp = self._checkpoint_infos.pop(0)
            if replaced_cp is not None:
                self._checkpoint_infos.append(
                    {
                        "epoch": self._current_epoch,
                        "steps": self._current_step,
                        "metric": metric,
                        "metric_name": metric_name,
                    }
                )
                self._checkpoint_infos = sorted(self._checkpoint_infos, key=lambda x: x["metric"])
                new_cp_dir = self.get_topk_checkpoint_name(
                    self._current_epoch,
                    self._current_step,
                    metric,
                    metric_name,
                )
                replaced_cp_dir = self.get_topk_checkpoint_name(
                    replaced_cp["epoch"], replaced_cp["steps"], replaced_cp["metric"], replaced_cp["metric_name"]
                )

        return new_cp_dir, replaced_cp_dir

    @run_on_main_process
    def save_checkpoint(
        self,
        dir_path: str,
        state_dicts: dict[str, OrderedDict],
        use_safetensors: dict[str, bool],
    ) -> None:
        """
        Save checkpoint

        Args:
            dir_path (str): path to the directory
            state_dicts (dict[str, OrderedDict]): state dicts to save
            use_safetensors (dict[str, bool]): whether to use safetensors for the state dicts
        """
        os.makedirs(dir_path, exist_ok=True)

        if use_safetensors is None:
            use_safetensors = {key: False for key in state_dicts.keys()}
        else:
            for key in state_dicts.keys():
                if key not in use_safetensors.keys():
                    use_safetensors[key] = False

        for key, state_dict in state_dicts.items():
            if use_safetensors[key]:
                save_file(state_dict, os.path.join(dir_path, f"{key}.safetensors"))
            else:
                torch.save(state_dict, os.path.join(dir_path, f"{key}.pth"))

        # Save state of the trainer
        with open(os.path.join(dir_path, "state.json"), "w") as f:
            json.dump(
                {
                    "current_epoch": self._current_epoch,
                    "current_step": self._current_step,
                    "checkpoint_infos": self._checkpoint_infos,
                },
                f,
                indent=4,
            )

    def resume_from_checkpoint(
        self,
        module: dict[str, nn.Module],
        use_safetensors: dict[str, bool],
        load_args: dict[str, dict],
        device: torch.device = torch.device("cpu"),
    ):
        """
        Resume training from the checkpoint

        Args:
            module (dict[str, nn.Module]): modules to load
            use_safetensors (dict[str, bool]): whether to use safetensors for the modules
            load_args (dict[str, dict]): arguments to load the modules
            device (torch.device): device to load the modules
        """
        assert self._checkpoint is not None, "Checkpoint is not set"

        # Load state
        self.load_state(self._checkpoint)

        # Set default load_args and use_safetensors
        if load_args is None:
            load_args = {key: {} for key in module.keys()}
        else:
            for key in module.keys():
                if key not in load_args.keys():
                    load_args[key] = {}

        if use_safetensors is None:
            use_safetensors = {key: False for key in module.keys()}
        else:
            for key in module.keys():
                if key not in use_safetensors.keys():
                    use_safetensors[key] = False

        # Load the module
        for key in module.keys():
            if use_safetensors[key]:
                if device == torch.device("cpu"):
                    device_str = "cpu"
                else:
                    device_str = "cuda"
                module[key].load_state_dict(
                    load_file(os.path.join(self._checkpoint, f"{key}.safetensors"), device=device_str),
                    **load_args[key],
                )
            else:
                module[key].load_state_dict(
                    torch.load(os.path.join(self._checkpoint, f"{key}.pth"), map_location=device),
                    **load_args[key],
                )

        logger.info("Training is resumed from checkpoint !")

    @run_on_main_process
    def log_metrics(self, metrics: dict):
        """
        Log metrics to tracker

        Args:
            metrics (dict): metrics to log
        """
        if self.tracker is not None:
            for key, value in metrics.items():
                self.tracker.log_scalar(key, value, self.current_step)

    @run_on_main_process
    def log_visualization(self, data: dict[str, np.ndarray]):
        """
        Log visualization to tracker

        Args:
            data (dict[str, np.ndarray]): data to log
        """
        if self.tracker is not None:
            for key, value in data.items():
                self.tracker.log_image(key, value, self.current_step)

    def reach_eval_interval(self, dataloader_length: int) -> bool:
        """
        Check if the current step is a multiple of the interval
        """
        if isinstance(self.val_interval, int):
            val_interval_steps = self.val_interval
        elif isinstance(self.val_interval, float) and self.val_interval <= 1:
            val_interval_steps = int(self.val_interval * dataloader_length // self.gradient_accumulation_steps)
        else:
            raise ValueError("val_interval must be either int or float <= 1")

        return (self.current_step + 1) % val_interval_steps == 0

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass
