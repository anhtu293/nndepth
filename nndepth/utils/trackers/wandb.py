import os
import wandb
import json
from typing import Optional


class WandbTracker(object):
    def __init__(
        self,
        project_name: str,
        run_name: str,
        root_log_dir: Optional[str] = None,
        group_name: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        config: Optional[dict] = None,
        resume: bool = False,
    ):
        """
        WandbLogger class to log data to Weights and Biases platform

        Parameters
            project_name (str): Name of the project
            run_name (str): Name of the run
            root_log_dir (str, optional): Root directory to save logs. Defaults to None.
            group_name (str, optional): Group name for the run. Defaults to None.
            tags (list, optional): Tags for the run. Defaults to None.
            notes (str, optional): Notes for the run. Defaults to None.
            config (dict, optional): Configuration for the run. Defaults to None.
            resume (bool, optional): Whether to resume the run. Defaults to False.
        """
        self.project_name = project_name
        self.run_name = run_name
        self.group_name = group_name
        self.tags = tags
        self.notes = notes
        self.config = config
        self.resume = resume
        if root_log_dir is None:
            self.log_dir = os.path.join("wandb", self.project_name, self.run_name, "wandb")
        else:
            self.log_dir = os.path.join(root_log_dir, "wandb")
        os.makedirs(self.log_dir, exist_ok=True)

        run_unique_id = wandb.util.generate_id()

        if self.resume:
            # get unique id of the run from wandb-id.json
            wandb_resume_path = os.path.join(self.log_dir, "wandb", "wandb-id.json")
            if os.path.exists(wandb_resume_path):
                with open(wandb_resume_path, "r") as f:
                    id = json.load(f)["run_id"]
            self.runner = wandb.init(
                project=self.project_name, name=self.run_name, resume=True, dir=self.log_dir, id=id
            )
        else:
            self.run = wandb.init(
                project=self.project_name,
                name=self.run_name,
                group=self.group_name,
                tags=self.tags,
                notes=self.notes,
                config=self.config,
                resume="allow",
                id=run_unique_id,
                dir=self.log_dir,
            )

        # Set step
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="train/step")

    def log(self, data: dict):
        """
        Log data to wandb

        Parameters
            data (dict): Data to log
        """
        self.run.log(data)
