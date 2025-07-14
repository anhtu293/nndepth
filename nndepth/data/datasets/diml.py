import os
import numpy as np
import cv2
import torch
from loguru import logger
from typing import List, Dict, Optional

from .base_dataset import BaseDataset
from nndepth.scene import Frame, Depth


class DIMLDataset(BaseDataset):
    SCENES = ["indoor", "outdoor"]
    VAL_SUBSETS = ["16. Billiard Hall", "170721_C0"]

    def __init__(
        self,
        dataset_dir: str,
        subset: Optional[str] = None,
        scenes: Optional[List[str]] = None,
        outdoor_conf_threshold: float = 0.6,
        **kwargs,
    ):
        """
        Args:
            dataset_dir: The directory of the dataset.
            subset: The subset to load (`train`, `val`). If None, all subsets will be loaded. Default: None.
            scenes: The scenes to load (`indoor`, `outdoor`). If None, all scenes will be loaded. Default: None.
            outdoor_conf_threshold: The confidence threshold for the outdoor subset. Default: 0.6.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)

        self.scenes = [scene for scene in self.SCENES if scenes is None or scene in scenes]
        assert len(self.scenes) > 0, "No scenes to load"

        assert subset is None or subset in ["train", "val"], "Invalid subset"
        self.subset = subset
        self.dataset_dir = dataset_dir
        self.items = self.init_items()
        self.outdoor_conf_threshold = outdoor_conf_threshold

        logger.info(f"Loaded {len(self.items)} items from {self.dataset_dir}")

    def __len__(self):
        return len(self.items)

    def init_items(self) -> List[Dict[str, str]]:
        items = []
        for scene in self.scenes:
            logger.info(f"Loading {scene} subset")
            items.extend(self.load_indoor_data() if scene == "indoor" else self.load_outdoor_data())
            logger.info(f"Loaded {len(items)} items from {scene} subset")
        return items

    def load_indoor_data(self) -> List[Dict[str, str]]:
        """
        Load the indoor subset.

        Returns:
            A list of dictionaries, each containing the following keys:
            - `image_path`: The path to the image.
            - `depth_path`: The path to the depth map.
        """
        indoor_items = []
        indoor_dir = os.path.join(self.dataset_dir, "indoor")
        folders = os.listdir(indoor_dir)
        if self.subset == "train":
            folders = [folder for folder in folders if folder not in self.VAL_SUBSETS]
        elif self.subset == "val":
            folders = [folder for folder in folders if folder in self.VAL_SUBSETS]
        for folder in folders:
            dates = os.listdir(os.path.join(indoor_dir, folder))
            for date in dates:
                batches = os.listdir(os.path.join(indoor_dir, folder, date))
                for batch in batches:
                    image_dir = os.path.join(indoor_dir, folder, date, batch, "col")
                    depth_dir = os.path.join(indoor_dir, folder, date, batch, "up_png")

                    file_names = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
                    file_names = [f.split("_c.png")[0] for f in file_names]

                    depth_postfix = os.listdir(depth_dir)[0].split("_")[-1]
                    for file_name in file_names:
                        image_path = os.path.join(image_dir, f"{file_name}_c.png")
                        depth_path = os.path.join(depth_dir, f"{file_name}_{depth_postfix}")
                        indoor_items.append({
                            "image_path": image_path,
                            "depth_path": depth_path,
                            "scene": "indoor",
                        })

        return indoor_items

    def load_outdoor_data(self) -> List[Dict[str, str]]:
        """
        Load the outdoor subset.

        Returns:
            A list of dictionaries, each containing the following keys:
            - `image_path`: The path to the image.
            - `depth_path`: The path to the depth map.
            - `conf_path`: The path to the confidence map.
            - `scene": The scene name.
        """
        outdoor_items = []
        outdoor_dir = os.path.join(self.dataset_dir, "outdoor")
        folders = os.listdir(outdoor_dir)
        if self.subset == "train":
            folders = [folder for folder in folders if folder not in self.VAL_SUBSETS]
        elif self.subset == "val":
            folders = [folder for folder in folders if folder in self.VAL_SUBSETS]
        for folder in folders:
            batches = os.listdir(os.path.join(outdoor_dir, folder))
            batches = [b for b in batches if os.path.isdir(os.path.join(outdoor_dir, folder, b))]
            for batch in batches:
                image_dir = os.path.join(outdoor_dir, folder, batch, "left")
                depth_dir = os.path.join(outdoor_dir, folder, batch, "depth")
                conf_dir = os.path.join(outdoor_dir, folder, batch, "conf")

                file_names = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
                file_names = [f.split("_left.png")[0] for f in file_names]
                for file_name in file_names:
                    image_path = os.path.join(image_dir, f"{file_name}_left.png")
                    depth_path = os.path.join(depth_dir, f"{file_name}_depth.png")
                    conf_path = os.path.join(conf_dir, f"{file_name}_conf.png")
                    outdoor_items.append({
                        "image_path": image_path,
                        "depth_path": depth_path,
                        "conf_path": conf_path,
                        "scene": "outdoor",
                    })
        return outdoor_items

    def load_indoor_item(self, item: Dict[str, str]) -> Frame:
        image_path = item["image_path"]
        depth_path = item["depth_path"]

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Load depth
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0  # mm -> meters
        depth = torch.from_numpy(depth).unsqueeze(0)
        valid_mask = depth > 0
        depth = Depth(data=depth, valid_mask=valid_mask)

        # Create frame
        frame = Frame(data=image, depth=depth)

        return frame

    def load_outdoor_item(self, item: Dict[str, str]) -> Frame:
        image_path = item["image_path"]
        depth_path = item["depth_path"]
        conf_path = item["conf_path"]

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Load depth
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0  # mm -> meters
        depth = torch.from_numpy(depth).unsqueeze(0)

        # Load confidence
        # Based on the paper (https://dimlrgbd.github.io/downloads/technical_report.pdf),
        # we use a confidence threshold of 0.9
        conf = cv2.imread(conf_path, cv2.IMREAD_UNCHANGED)
        conf = conf.astype(np.float32) / 255.0
        conf = torch.from_numpy(conf).unsqueeze(0)
        valid_mask = conf > self.outdoor_conf_threshold
        depth = Depth(data=depth, valid_mask=valid_mask)

        # Create frame
        frame = Frame(data=image, depth=depth)

        return frame

    def get_item(self, idx: int) -> Optional[Frame]:
        item = self.items[idx]
        scene = item["scene"]
        if scene == "indoor":
            return self.load_indoor_item(item)
        elif scene == "outdoor":
            return self.load_outdoor_item(item)
