import os
import numpy as np
import cv2
import torch
from loguru import logger
from typing import List, Dict

from nndepth.scene import Frame, Depth


class HRWSIDataset:
    SUBSETS = ["train", "val"]
    LOADING_RETRY_LIMIT = 10

    def __init__(self, dataset_dir: str, subsets: List[str] = None):
        """
        Args:
            dataset_dir: The directory of the dataset.
            splits: The splits to load (`train`, `val`). If None, all splits will be loaded. Default: None.
        """
        self.subsets = [subset for subset in self.SUBSETS if subsets is None or subset in subsets]
        assert len(self.subsets) > 0, "No subsets to load"

        self.dataset_dir = dataset_dir
        self.items = self.get_items()

        logger.info(f"Loaded {len(self.items)} items from {self.dataset_dir}")

    def __len__(self):
        return len(self.items)

    def get_items(self) -> List[Dict[str, str]]:
        items = []
        for subset in self.subsets:
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

    def __getitem__(self, idx: int):
        nb_retry = 0
        while nb_retry < self.LOADING_RETRY_LIMIT:
            try:
                item = self.items[idx]
                return self.load_item(item)
            except Exception as e:
                nb_retry += 1
                idx = (idx + 1) % len(self.items)
                logger.warning(f"Error while loading item {idx}: {e}. Retry with item {idx}")

        logger.error("Error while loading data. Retry limit reached")
        return None
