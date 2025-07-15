import os
import numpy as np
import cv2
import torch
from loguru import logger
from typing import List, Dict, Optional

from nndepth.scene import Frame, Depth
from .base_dataset import BaseDataset


class HRWSIDataset(BaseDataset):
    SUBSETS = ["train", "val"]

    def __init__(self, dataset_dir: str, subset: str = None, **kwargs):
        """
        Args:
            dataset_dir: The directory of the dataset.
            splits: The splits to load (`train`, `val`). If None, all splits will be loaded. Default: None.
        """
        super().__init__(**kwargs)

        self.subset = [subset for subset in self.SUBSETS if subset is None or subset in subset]
        assert len(self.subset) > 0, "No subsets to load"

        self.dataset_dir = dataset_dir
        self.items = self.init_items()

        logger.info(f"Loaded {len(self.items)} items from {self.dataset_dir}")

    def __len__(self):
        return len(self.items)

    def init_items(self) -> List[Dict[str, str]]:
        items = []
        for subset in self.subset:
            logger.info(f"Loading {subset} subset")
            items.extend(self.load_subset_data(subset))
            logger.info(f"Loaded {len(items)} items from {subset} subset")
        return items

    def load_subset_data(self, subset: str) -> List[Dict[str, str]]:
        """
        Load data from a specific split.

        Args:
            subset: The subset to load (`train` or `val`).

        Returns:
            A list of dictionaries, each containing the following keys:
            - `image_path`: The path to the image.
            - `depth_path`: The path to the depth map.
            - `valid_mask_path`: The path to the valid mask.
            - `split`: The split name.
        """
        subset_items = []
        subset_dir = os.path.join(self.dataset_dir, subset)

        imgs_dir = os.path.join(subset_dir, "imgs")
        gts_dir = os.path.join(subset_dir, "gts")
        valid_masks_dir = os.path.join(subset_dir, "valid_masks")

        # Get all image files
        image_files = sorted([f for f in os.listdir(imgs_dir) if f.endswith(".jpg")])

        for image_file in image_files:
            # Extract base name without extension
            base_name = os.path.splitext(image_file)[0]

            # Build paths
            image_path = os.path.join(imgs_dir, image_file)
            depth_path = os.path.join(gts_dir, f"{base_name}.png")
            valid_mask_path = os.path.join(valid_masks_dir, f"{base_name}.png")

            # Check if all required files exist
            if not all(os.path.exists(path) for path in [image_path, depth_path, valid_mask_path]):
                logger.warning(f"Missing files for {base_name}, skipping")
                continue

            item = {
                "image_path": image_path,
                "depth_path": depth_path,
                "valid_mask_path": valid_mask_path,
                "subset": subset,
            }

            subset_items.append(item)

        return subset_items

    def load_item(self, item: Dict[str, str]) -> Frame:
        """
        Load a single item from the dataset.

        Args:
            item: Dictionary containing file paths.

        Returns:
            A Frame object containing the image and depth data.
        """
        image_path = item["image_path"]
        depth_path = item["depth_path"]
        valid_mask_path = item["valid_mask_path"]

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Load depth map
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)

        # Load valid mask
        valid_mask = cv2.imread(valid_mask_path, cv2.IMREAD_GRAYSCALE)
        valid_mask = torch.from_numpy(valid_mask).unsqueeze(0) > 0  # Convert to boolean

        # Create depth object with valid mask
        depth_obj = Depth(data=depth, valid_mask=valid_mask, is_inverse=True)
        depth_obj = depth_obj.inverse(clip_max=None, clip_min=None)

        # Create frame
        frame = Frame(data=image, depth=depth_obj)

        return frame

    def get_item(self, idx: int) -> Optional[Frame]:
        item = self.items[idx]
        return self.load_item(item)
