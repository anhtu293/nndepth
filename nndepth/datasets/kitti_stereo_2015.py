import os
import cv2
import torch
import numpy as np
from typing import Dict

from aloscene import Frame, Disparity, Mask
from aloscene.camera_calib import CameraIntrinsic, CameraExtrinsic

from alodataset.utils.kitti import load_calib_cam_to_cam


class KittiStereo2015:
    SPLIT_FOLDERS = {"train": "training", "val": "testing"}
    LABELS = ["right", "disp_noc", "disp_occ"]
    CAMERAS = ["left", "right"]

    def __init__(
        self,
        dataset_dir: str = "/data/kitti/stereo_2015",
        subset: str = "train",
        sequence_start=0,
        sequence_end=11,
        cameras: list = ["left", "right"],
        labels: list = ["disp_occ"],
    ):
        """
        Stereo Tasks from Kitti 2015 dataset.
        Parameters
        ----------
        name : str
            Name of the dataset
        sequence_start : int
            20 images are available for each item. Only image 10 and 11 are annotated.
            sequence_start is the first image to load.
        sequence_end : int
            sequence_end is the last image to load.
        grayscale : bool
            If True, load images in grayscale.
        labels : List[str]
            List of data to load. Available data are:
            - right: right image
            - disp_noc: disparity map without occlusions
            - disp_occ: disparity map with occlusions
        split : Split
            Split of the dataset. Can be `Split.TRAIN` or `Split.TEST`.

        Examples
        --------
        >>> # Load dataset with only the 2 annotated images
        >>> dataset = KittiStereoFlowSFlow2015(sequence_start=10, sequence_end=11)
        >>> # Load dataset with 3 context images before the 2 annotated images
        >>> dataset = KittiStereoFlowSFlow2015(sequence_start=7, sequence_end=11)
        >>> # Load dataset with all the context images
        >>> dataset = KittiStereoFlowSFlow2015(sequence_start=0, sequence_end=20)
        """
        super().__init__()
        assert subset in ["train", "val"], "subset must be in [`train`, `val`]"
        assert not ("disp_noc" in labels and "disp_occ" in labels), (
            "only 1 disparity (`disp_occ` or `disp_noc`) can be passed to labels"
        )
        assert all([x in self.LABELS for x in labels]), f"labels must be in {self.LABELS}, found {labels}"
        assert all([c in self.CAMERAS for c in cameras]), f"cameras must be in {self.CAMERAS}, found {cameras}"
        assert "left" in cameras or "right" in cameras, f"`left` or `right` must be in cameras, found {cameras}"

        self.dataset_dir = dataset_dir
        self.subset = subset
        self.sequence_start = sequence_start
        self.sequence_end = sequence_end
        self.cameras = cameras
        self.labels = labels

        assert sequence_start <= sequence_end, "sequence_start should be less than sequence_end"

        if "disp_noc" in labels or "disp_occ" in labels:
            assert sequence_start <= 11 and sequence_end >= 10, "Disparity is not available for this frame range"

        # Load sequence length
        self.split_folder = os.path.join(self.dataset_dir, self.SPLIT_FOLDERS[subset])
        left_img_folder = os.path.join(self.split_folder, "image_2")
        self.len = len([f for f in os.listdir(left_img_folder) if f.endswith("_10.png")])

    def __len__(self):
        return self.len

    def load_disp(self, disp_path: str, camera_side: str):
        img = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        disp = img / 256.0
        disp = torch.as_tensor(disp[None, ...], dtype=torch.float32)
        mask = disp <= 0
        mask = Mask(mask, names=("C", "H", "W"))
        disp = Disparity(disp, names=("C", "H", "W"), mask=mask, camera_side=camera_side).signed()
        return disp

    def extrinsic(self, idx: int) -> CameraExtrinsic:
        """Load extrinsic from corresponding file"""
        with open(os.path.join(self.split_folder, f"calib_cam_to_cam/{idx:06d}.txt"), "r") as f:
            file = f.readlines()
            rotation = [float(x) for x in file[5].split(" ")[1:]]
            translation = [float(x) for x in file[6].split(" ")[1:]]
            rotation = np.array(rotation).reshape(3, 3)
            translation = np.array(translation).reshape(3, 1)
            extrinsic = np.append(rotation, translation, axis=1)
            return CameraExtrinsic(np.append(extrinsic, np.array([[0, 0, 0, 1]]), axis=0))

    def __getitem__(self, idx) -> Dict[str, Frame]:
        """
        Load a sequence of frames from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sequence.

        Returns
        -------
        Dict[int, Dict[str, Frame]]
            Dictionary of index beetween sequance_start and sequance_end.\n
            Each index is a dictionary of frames ("left" and maybe "right").
        """
        sequence: Dict[int, Dict[str, Frame]] = {}
        calib = self._load_calib(self.split_folder, idx)

        # We need to load the sequance from the last to the first frame because we need information from the previous
        # frame to compute the scene flow.
        for index in range(self.sequence_end, self.sequence_start - 1, -1):
            sequence[index] = {}
            if "left" in self.cameras:
                sequence[index]["left"] = Frame(os.path.join(self.split_folder, f"image_2/{idx:06d}_{index:02d}.png"))
                sequence[index]["left"].baseline = calib["baseline"]
                sequence[index]["left"].append_cam_intrinsic(calib["left_intrinsic"])
                sequence[index]["left"].append_cam_extrinsic(calib["left_extrinsic"])
            if "right" in self.cameras:
                sequence[index]["right"] = Frame(os.path.join(self.split_folder, f"image_3/{idx:06d}_{index:02d}.png"))
                sequence[index]["right"].baseline = ["baseline"]
                sequence[index]["right"].append_cam_intrinsic(calib["right_intrinsic"])
                sequence[index]["right"].append_cam_extrinsic(calib["right_extrinsic"])

            # Frames at index 10 and 11 are the only one who have ground truth in dataset.
            if index == 11:
                dummy_disp_size = (1, sequence[index]["left"].H, sequence[index]["left"].W)
                if "disp_noc" in self.labels:
                    sequence[11]["left"].append_disparity(
                        self.load_disp(os.path.join(self.split_folder, f"disp_noc_1/{idx:06d}_10.png"), "left"),
                    )
                if "disp_occ" in self.labels:
                    sequence[11]["left"].append_disparity(
                        self.load_disp(os.path.join(self.split_folder, f"disp_occ_1/{idx:06d}_10.png"), "left"),
                    )
            elif index == 10:
                if "disp_noc" in self.labels:
                    sequence[10]["left"].append_disparity(
                        self.load_disp(os.path.join(self.split_folder, f"disp_noc_0/{idx:06d}_10.png"), "left"),
                    )
                if "disp_occ" in self.labels:
                    sequence[10]["left"].append_disparity(
                        self.load_disp(os.path.join(self.split_folder, f"disp_occ_0/{idx:06d}_10.png"), "left"),
                    )
            else:
                dummy_disp_size = (1, sequence[index]["left"].H, sequence[index]["left"].W)
                if "disp_noc" in self.labels:
                    dummy_disp = Disparity.dummy(dummy_disp_size, ("C", "H", "W")).signed("left")
                    sequence[index]["left"].append_disparity(dummy_disp)
                if "disp_occ" in self.labels:
                    dummy_disp = Disparity.dummy(dummy_disp_size, ("C", "H", "W")).signed("left")
                    sequence[index]["left"].append_disparity(dummy_disp)

            sequence[index]["left"] = sequence[index]["left"].temporal()
            if "right" in self.cameras:
                sequence[index]["right"] = sequence[index]["right"].temporal()

        result = {}
        left = [sequence[frame]["left"] for frame in range(self.sequence_start, self.sequence_end + 1)]
        result["left"] = torch.cat(left, dim=0)
        if "right" in self.cameras:
            right = [sequence[frame]["right"] for frame in range(self.sequence_start, self.sequence_end + 1)]
            result["right"] = torch.cat(right, dim=0)

        return result

    def _load_calib(self, path, idx):
        data = load_calib_cam_to_cam(
            os.path.join(path, "calib_cam_to_cam", f"{idx:06d}.txt"),
            os.path.join(path, "calib_velo_to_cam", f"{idx:06d}.txt"),
        )

        # Return only the parameters we are interested in.
        result = {
            "baseline": data["b_rgb"],
            "left_intrinsic": CameraIntrinsic(np.c_[data["K_cam2"], [0, 0, 0]]),
            "right_intrinsic": CameraIntrinsic(np.c_[data["K_cam3"], [0, 0, 0]]),
            "left_extrinsic": CameraExtrinsic(data["T_cam2_rect"]),
            "right_extrinsic": CameraExtrinsic(data["T_cam3_rect"]),
        }
        return result


if __name__ == "__main__":
    from random import randint

    dataset = KittiStereo2015(sequence_start=10, sequence_end=10)
    obj = dataset[randint(0, len(dataset))]
    print(obj["left"].shape)
    print(obj["left"].disparity)
    obj["left"].get_view().render()
