import h5py
import cv2
import os
import numpy as np
import torch
from typing import Dict, List, Optional
from tqdm import tqdm
from loguru import logger

from nndepth.scene import Frame, Depth
from .base_dataset import BaseDataset


class HypersimDataset(BaseDataset):
    LABELS = ["depth"]
    FOCAL_LENGTH = 886.81

    def __init__(self, dataset_dir: str, sequences: List[str] = None, labels: List[str] = ["depth"], **kwargs):
        """
        Hypersim dataset.

        Args:
            dataset_dir: Path to the dataset directory.
            sequences: List of sequences to load. If None, all sequences will be loaded. Default is None.
            labels: List of labels to load. Default is ["depth"].
        """
        super().__init__(**kwargs)

        self.labels = [label for label in self.LABELS if label in labels]
        assert len(self.labels) > 0, "No labels provided"

        self.sequences = sequences
        self.dataset_dir = dataset_dir

        self.items = self.init_items()

    def __len__(self) -> int:
        return len(self.items)

    def get_cameras(self, seq_dir: str) -> List[str]:
        folders = os.listdir(seq_dir)
        cameras = [folder for folder in folders if os.path.isdir(os.path.join(seq_dir, folder))]
        cameras = [camera for camera in cameras if camera.startswith("scene_cam_")]
        cameras = sorted([camera.split("_")[2] for camera in cameras])
        cameras = list(set(cameras))
        return cameras

    def init_items(self) -> List[Dict[str, str]]:
        logger.info(f"Loading items from {self.dataset_dir}")
        sequences = os.listdir(self.dataset_dir)
        if self.sequences is not None:
            sequences = [seq for seq in sequences if seq in self.sequences]

        if len(sequences) == 0:
            raise ValueError(f"No sequences found in {self.dataset_dir}")

        items = []
        sequences = sorted(sequences)
        for seq in tqdm(sequences, desc="Loading sequences"):
            seq_dir = os.path.join(self.dataset_dir, seq, "images")
            cameras = self.get_cameras(seq_dir)
            for camera in cameras:
                image_dir = os.path.join(seq_dir, f"scene_cam_{camera}_final_preview")
                geo_dir = os.path.join(seq_dir, f"scene_cam_{camera}_geometry_hdf5")

                image_files = [f for f in os.listdir(image_dir) if f.endswith(".color.jpg")]
                images = sorted([f.replace(".color.jpg", "") for f in image_files])

                for image in images:
                    image_path = os.path.join(image_dir, image + ".color.jpg")
                    geo_path = os.path.join(geo_dir, image + ".depth_meters.hdf5")
                    items.append({
                        "image": image_path,
                        "depth": geo_path,
                    })

        logger.info(f"Loaded {len(items)} items")

        return items

    def distance_to_depth(self, distance: torch.Tensor) -> torch.Tensor:
        """
        Convert distance to depth.
        """
        HW = distance.shape[1:]
        x_range = torch.arange(-(HW[1] // 2), HW[1] // 2, dtype=torch.float32)
        y_range = torch.arange(-(HW[0] // 2), HW[0] // 2, dtype=torch.float32)
        X, Y = torch.meshgrid(x_range, y_range, indexing="xy")
        Z = torch.full_like(X, self.FOCAL_LENGTH, dtype=torch.float32)
        coords = torch.stack([X, Y, Z], dim=-1)
        norm = torch.linalg.norm(coords, ord=2, dim=-1).unsqueeze(0)
        depth = distance / norm * self.FOCAL_LENGTH
        return depth

    def get_item(self, idx: int) -> Optional[Frame]:
        item = self.items[idx]
        image = cv2.imread(item["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1)

        distance = h5py.File(item["depth"], "r")["dataset"]
        distance = torch.from_numpy(np.asarray(distance).squeeze()).unsqueeze(0)
        depth = self.distance_to_depth(distance)

        valid_mask = torch.isnan(depth) == 0
        depth = Depth(depth, valid_mask)

        return Frame(data=image, depth=depth)
