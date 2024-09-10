import os
import cv2
import torch
import numpy as np
from typing import Dict, Union

from nndepth.scene import Frame, Disparity


# https://github.com/utiasSTARS/pykitti/tree/master
def read_calib_file(filepath: str):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, "r") as f:
        for line in f.readlines():
            if line == "\n":
                continue
            key, value = line.split(" ", 1)
            key = key.strip(":")
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def transform_from_rot_trans(R: np.ndarray, t: np.ndarray):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def load_calib_rigid(filepath: str):
    """Read a rigid transform calibration file as a numpy.array."""
    data = read_calib_file(filepath)
    return transform_from_rot_trans(data["R"], data["T"])


def load_calib_cam_to_cam(cam_to_cam_filepath, velo_to_cam_file: Union[str, None] = None) -> dict:
    # We'll return the camera calibration as a dictionary
    data = {}

    # Load and parse the cam-to-cam calibration data
    filedata = read_calib_file(cam_to_cam_filepath)

    names = ["P_rect_00", "P_rect_01", "P_rect_02", "P_rect_03"]
    if "P0" in filedata:
        names = ["P0", "P1", "P2", "P3"]

    # Create 3x4 projection matrices
    p_rect = [np.reshape(filedata[p], (3, 4)) for p in names]

    for i, p in enumerate(p_rect):
        data[f"P_rect_{i}0"] = p

    # Compute the rectified extrinsics from cam0 to camN
    rectified_extrinsics = [np.eye(4) for _ in range(4)]
    for i in range(4):
        rectified_extrinsics[i][0, 3] = p_rect[i][0, 3] / p_rect[i][0, 0]
        data[f"T_cam{i}_rect"] = rectified_extrinsics[i]

        # Compute the camera intrinsics
        data[f"K_cam{i}"] = p_rect[i][0:3, 0:3]

    # Create 4x4 matrices from the rectifying rotation matrices
    r_rect = None
    if "R_rect_00" in filedata:
        r_rect = [np.eye(4) for _ in range(4)]
        for i in range(4):
            r_rect[i][0:3, 0:3] = np.reshape(filedata["R_rect_0" + str(i)], (3, 3))
            data[f"R_rect_{i}0"] = r_rect[i]

    # Load the rigid transformation from velodyne coordinates
    # to unrectified cam0 coordinates
    t_cam0unrect_velo = None
    stereo_baseline = None
    if velo_to_cam_file is not None and r_rect is not None:
        t_cam0unrect_velo = load_calib_rigid(velo_to_cam_file)

        velo_to_cam = [rectified_extrinsics[i].dot(r_rect[i].dot(t_cam0unrect_velo)) for i in range(4)]
        p_cam = np.array([0, 0, 0, 1])
        stereo_baseline = [np.linalg.inv(velo_to_cam[i]).dot(p_cam) for i in range(4)]

        for i in range(4):
            data[f"T_cam{i}_velo"] = velo_to_cam[i]

    elif "Tr_velo_to_cam" in filedata or "Tr_velo_cam" in filedata or "Tr" in filedata:
        prop_name = (
            "Tr_velo_to_cam" if "Tr_velo_to_cam" in filedata else "Tr_velo_cam" if "Tr_velo_cam" in filedata else "Tr"
        )
        data["T_cam0_velo"] = np.reshape(filedata[prop_name], (3, 4))
        data["T_cam0_velo"] = np.vstack([data["T_cam0_velo"], [0, 0, 0, 1]])
        for i in range(1, 4):
            data[f"T_cam{i}_velo"] = rectified_extrinsics[i].dot(data["T_cam0_velo"])
        p_cam = np.array([0, 0, 0, 1])
        stereo_baseline = [np.linalg.inv(data[f"T_cam{i}_velo"]).dot(p_cam) for i in range(4)]

    if stereo_baseline is not None:
        data["b_gray"] = np.linalg.norm(stereo_baseline[1] - stereo_baseline[0])
        data["b_rgb"] = np.linalg.norm(stereo_baseline[3] - stereo_baseline[2])

    return data


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

    def load_image(self, image_path: str) -> torch.Tensor:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.Tensor(img.transpose((2, 0, 1)))

    def load_disp(self, disp_path: str, camera_side: str) -> torch.Tensor:
        """Load disparity from path"""
        img = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        disp = img / 256.0
        disp = torch.as_tensor(disp[None, ...], dtype=torch.float32)
        mask = (disp <= 0).float()
        if camera_side == "left":
            disp *= -1
        disp_obj = Disparity(
            data=disp,
            disp_sign="negative",
            occlusion=mask,
        )
        return disp_obj

    def extrinsic(self, idx: int) -> torch.Tensor:
        """Load extrinsic from corresponding file"""
        with open(os.path.join(self.split_folder, f"calib_cam_to_cam/{idx:06d}.txt"), "r") as f:
            file = f.readlines()
            rotation = [float(x) for x in file[5].split(" ")[1:]]
            translation = [float(x) for x in file[6].split(" ")[1:]]
            rotation = np.array(rotation).reshape(3, 3)
            translation = np.array(translation).reshape(3, 1)
            extrinsic = np.append(rotation, translation, axis=1)
            return torch.Tensor(np.append(extrinsic, np.array([[0, 0, 0, 1]]), axis=0))

    def __getitem__(self, idx) -> Dict[str, Frame]:
        """
        Load a sequence of frames from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sequence.

        Returns
        -------
        Dict[str, List[Frame]]
            Dictionary of index beetween sequance_start and sequance_end.\n
            Each index is a list of frames.
        """
        sequence: Dict[int, Dict[str, Frame]] = {}
        calib = self._load_calib(self.split_folder, idx)

        # We need to load the sequance from the last to the first frame because we need information from the previous
        # frame to compute the scene flow.
        for index in range(self.sequence_end, self.sequence_start - 1, -1):
            sequence[index] = {}
            if "left" in self.cameras:
                img = self.load_image(os.path.join(self.split_folder, f"image_2/{idx:06d}_{index:02d}.png"))
                baseline = calib["baseline"]
                cam_intrinsic = calib["left_intrinsic"]
                cam_extrinsic = calib["left_extrinsic"]
                sequence[index]["left"] = Frame(
                    image=img,
                    cam_intrinsic=cam_intrinsic,
                    cam_extrinsic=cam_extrinsic,
                    baseline=baseline,
                )
            if "right" in self.cameras:
                img = self.load_image(os.path.join(self.split_folder, f"image_3/{idx:06d}_{index:02d}.png"))
                baseline = calib["baseline"]
                cam_intrinsic = calib["left_intrinsic"]
                cam_extrinsic = calib["left_extrinsic"]
                sequence[index]["right"] = Frame(
                    image=img,
                    cam_intrinsic=cam_intrinsic,
                    cam_extrinsic=cam_extrinsic,
                    baseline=baseline,
                )

            # Frames at index 10 and 11 are the only one who have ground truth in dataset.
            if index == 11:
                if "disp_noc" in self.labels:
                    sequence[11]["left"].disparity = (
                        self.load_disp(os.path.join(self.split_folder, f"disp_noc_1/{idx:06d}_10.png"), "left")
                    )
                if "disp_occ" in self.labels:
                    sequence[11]["left"].disparity = (
                        self.load_disp(os.path.join(self.split_folder, f"disp_occ_1/{idx:06d}_10.png"), "left")
                    )
            elif index == 10:
                if "disp_noc" in self.labels:
                    sequence[10]["left"].disparity = (
                        self.load_disp(os.path.join(self.split_folder, f"disp_noc_0/{idx:06d}_10.png"), "left")
                    )
                if "disp_occ" in self.labels:
                    sequence[10]["left"].disparity = (
                        self.load_disp(os.path.join(self.split_folder, f"disp_occ_0/{idx:06d}_10.png"), "left")
                    )

        result = {}
        left = [sequence[frame]["left"] for frame in range(self.sequence_start, self.sequence_end + 1)]
        result["left"] = left
        if "right" in self.cameras:
            right = [sequence[frame]["right"] for frame in range(self.sequence_start, self.sequence_end + 1)]
            result["right"] = right

        return result

    def _load_calib(self, path, idx):
        data = load_calib_cam_to_cam(
            os.path.join(path, "calib_cam_to_cam", f"{idx:06d}.txt"),
            os.path.join(path, "calib_velo_to_cam", f"{idx:06d}.txt"),
        )

        # Return only the parameters we are interested in.
        result = {
            "baseline": data["b_rgb"],
            "left_intrinsic": torch.Tensor(np.c_[data["K_cam2"], [0, 0, 0]]),
            "right_intrinsic": torch.Tensor(np.c_[data["K_cam3"], [0, 0, 0]]),
            "left_extrinsic": torch.Tensor(data["T_cam2_rect"]),
            "right_extrinsic": torch.Tensor(data["T_cam3_rect"]),
        }
        return result


if __name__ == "__main__":
    from random import randint

    dataset = KittiStereo2015(sequence_start=10, sequence_end=10)
    obj = dataset[randint(0, len(dataset))]
    frame = obj["left"][0]

    disps = torch.stack([frame.disparity, frame.disparity], dim=0)
    disp_viz = disps.get_view()

    cv2.imshow("disp", disp_viz[0])
    cv2.waitKey(0)
