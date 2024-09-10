import numpy as np
import torch
import os
from loguru import logger
from typing import List, Tuple, Dict

from nndepth.scene import Frame, Disparity, Depth
from nndepth.datasets.utils.geometry_trans import pos_quat2SE


def sequence_index(start, seq_size):
    end = start + (seq_size - 1)
    return np.linspace(start, end, seq_size).astype(int).tolist()


def sequence_indices(n_samples, seq_size, seq_skip):
    for start in range(0, n_samples - (seq_size - 1), seq_skip + 1):
        yield sequence_index(start, seq_size)


class TartanairDataset(object):

    CAMERAS = ["left", "right"]
    LABELS = ["disparity", "depth", "segmentation"]

    LOADING_RETRY_LIMIT = 10
    FX = 320
    FY = 320
    BASELINE = 0.25

    def __init__(
        self,
        dataset_dir: str = "/data/tartanair",
        envs: List[str] = None,
        sequences: Dict[str, List[str]] = None,
        cameras: List[str] = None,
        labels: List[str] = None,
        sequence_size: int = 1,
        sequence_skip: int = 0,
        start_from: int = 0,
        pose_format: str = "NED",
    ):
        """
        Dataset for Tartanair dataset

        Args:
            dataset_dir (str): path to the dataset
            envs (List[str]): list of environments
            sequences (Dict[str, List[str])]: list of sequences per evironment
            cameras (List[str]): list of cameras
            labels (List[str]): list of labels
            sequence_size (int): size of the sequence
            sequence_skip (int): skip between sequences
            start_from (int): start from
            pose_format (str): pose format
            get_cameras_fn (Callable): function to get cameras

        Returns:
            None
        """
        super().__init__()
        assert os.path.exists(dataset_dir), f"Dataset directory {dataset_dir} does not exist"
        assert pose_format in ["NED", "camera"], "pose_format should be in {'NED', 'camera'}"
        assert cameras is None or all([cam in self.CAMERAS for cam in cameras]), f"cameras should be in {self.CAMERAS}"
        assert labels is None or all([label in self.LABELS for label in labels]), f"labels should be in {self.LABELS}"

        self.dataset_dir = dataset_dir
        self.envs = envs
        self.sequences = sequences
        self.cameras = cameras if cameras is not None else self.CAMERAS
        self.labels = labels if labels is not None else self.LABELS
        self.sequence_size = sequence_size
        self.sequence_skip = sequence_skip
        self.start_from = start_from
        self.images_format = "png"
        self.items = {}

        if pose_format not in ["NED", "camera"]:
            raise ValueError("pose_format should be in {'NED', 'camera'}")
        self.pose_format = pose_format

        self.sequence_to_camera_pos = {}

        env_seq_levels = self._get_items(envs)
        for env, seq, level in env_seq_levels:
            data_dir = os.path.join(self.dataset_dir, env, env, level, seq)

            pose_left, pose_right = self._load_camera_poses(data_dir)

            sequence_name = f"{env}-{level}-{seq}"
            self.sequence_to_camera_pos[sequence_name] = {
                "pose_left": pose_left,
                "pose_right": pose_right,
            }

            # Count the number of element in the sequence
            sequence_size = len([el for el in os.listdir(os.path.join(data_dir, "image_left")) if ".png" in el])
            temporal_sequences = sequence_indices(sequence_size, self.sequence_size, self.sequence_skip)

            self.items.update(
                {
                    len(self.items)
                    + idx: {
                        "env": env,
                        "sequence": seq,
                        "temporal_sequence": temporal_seq,
                        "level": level,
                    }
                    for idx, temporal_seq in enumerate(temporal_sequences)
                }
            )

    def _load_camera_poses(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        # Open the cameras positions
        with open(os.path.join(data_dir, "pose_left.txt"), "r") as f:
            pose_left = np.array(
                [
                    [float(el) for el in line.split(" ") if len(el) > 0]
                    for line in f.read().split("\n")
                    if len(line) > 0
                ]
            )
        with open(os.path.join(data_dir, "pose_right.txt"), "r") as f:
            pose_right = np.array(
                [
                    [float(el) for el in line.split(" ") if len(el) > 0]
                    for line in f.read().split("\n")
                    if len(line) > 0
                ]
            )
        return pose_left, pose_right

    def _get_envs(self) -> List[str]:
        environments = []
        for env in sorted(os.listdir(self.dataset_dir)):
            if not env.endswith(".zip") and env != "tartanair_tools":
                environments.append(env)
        return environments

    def _get_sequences(self, env: str) -> List[str]:
        diff_dir = os.path.join(self.dataset_dir, env, env, "Easy")
        all_sequences = os.listdir(diff_dir)
        all_sequences = [
            seq for seq in all_sequences if os.path.isdir(os.path.join(self.dataset_dir, env, env, "Easy", seq))
        ]
        if self.sequences is not None and env in self.sequences:
            sequences = self.sequences[env]
            sequences = [seq for seq in sequences if seq in all_sequences]
        else:
            sequences = all_sequences
        return sorted(sequences)

    def _get_items(self, envs: List[str]) -> List[Tuple[str, str, str]]:
        envs = self._get_envs() if envs is None else envs

        env_seq_level = []
        for env in envs:
            seqs = self._get_sequences(env)
            env_seq_level.extend([[env, seq, "Easy"] for seq in seqs])
        return env_seq_level

    def load_image(self, image_path: str) -> torch.Tensor:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.Tensor(image.transpose((2, 0, 1)))

    def create_cam_intrinsic(self, fx: float, fy: float, cx: float, cy: float, skew: float) -> torch.Tensor:
        intrinsic = torch.eye(3)
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[0, 1] = skew
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy
        intrinsic = torch.cat([intrinsic, torch.zeros((3, 1))], dim=1)
        return intrinsic

    def get_frame(self, side: str, sub_sequence_folder: str, frame_id: int) -> Frame:
        # Frame + Camera Intrinsic
        frame_left_path = os.path.join(sub_sequence_folder, f"image_{side}", f"{frame_id:06d}_{side}.png")
        image = self.load_image(frame_left_path)

        cam_extrinsic = np.eye(4)
        cam_extrinsic[1, -1] = -self.BASELINE / 2 if side == "left" else self.BASELINE / 2
        cam_intrinsic = self.create_cam_intrinsic(
            fx=self.FX, fy=self.FY, cx=image.shape[-1] / 2, cy=image.shape[-2] / 2, skew=0
        )

        frame = Frame(
            image=image,
            cam_intrinsic=cam_intrinsic,
            cam_extrinsic=cam_extrinsic,
            camera=side,
            baseline=self.BASELINE
        )

        # Depth
        # right depth is not available for annotation at the moment
        if "depth" in self.labels and side == "left":
            depth_path = os.path.join(
                sub_sequence_folder,
                f"depth_{side}",
                f"{frame_id:06d}_{side}_depth.npy",
            )
            depth = torch.Tensor(np.expand_dims(np.load(depth_path), axis=0))
            depth = Depth(data=depth)
            frame.planar_depth = depth

        # load disparity from depth
        if "disparity" in self.labels and side == "left":
            depth_path = os.path.join(
                sub_sequence_folder,
                f"depth_{side}",
                f"{frame_id:06d}_{side}_depth.npy",
            )
            depth = np.expand_dims(np.load(depth_path), axis=0)
            disparity = -self.BASELINE * self.FX / (depth + 1e-8)
            disparity = torch.Tensor(disparity)
            disparity = Disparity(data=disparity, disp_sign="negative")
            frame.disparity = disparity

        return frame

    def NED2camera(self, P: np.ndarray) -> np.ndarray:
        T = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        T_inv = np.linalg.inv(T)
        return T @ P @ T_inv

    def __getitem__(self, idx: int) -> Dict[str, list[torch.Tensor]]:
        nb_retry = 0
        while nb_retry < self.LOADING_RETRY_LIMIT:
            try:
                sequence_data = self.items[idx]

                env = sequence_data["env"]
                seq = sequence_data["sequence"]
                difficulty = sequence_data["level"]

                sub_sequence_folder = os.path.join(
                    self.dataset_dir,
                    sequence_data["env"],
                    sequence_data["env"],
                    sequence_data["level"],
                    sequence_data["sequence"],
                )

                frames = {side: [] for side in self.cameras}

                sequence_name = f"{env}-{difficulty}-{seq}"

                temporal_sequence = sequence_data["temporal_sequence"]
                for el in temporal_sequence:
                    left_pose = self.sequence_to_camera_pos[sequence_name]["pose_left"][el]
                    left_pose = pos_quat2SE(left_pose)
                    if self.pose_format == "camera":
                        left_pose = self.NED2camera(left_pose)

                    right_pose = self.sequence_to_camera_pos[sequence_name]["pose_right"][el]
                    right_pose = pos_quat2SE(right_pose)
                    if self.pose_format == "camera":
                        right_pose = self.NED2camera(right_pose)

                    poses = {"left": left_pose, "right": right_pose}
                    for side in self.cameras:
                        frame = self.get_frame(side, sub_sequence_folder, el)
                        P = torch.as_tensor(poses[side], dtype=torch.float32)
                        frame.pose = P
                        frames[side].append(frame)

                return frames
            except Exception as e:
                nb_retry += 1
                idx = (idx + 1) % len(self.items)
                logger.info(
                    f"Error while loading sequence {sequence_name} - Idx {idx}: {e}. Retry with sequence {idx}"
                )
        logger.error(Exception("Error while loading data. Retry limit reached"))
        return None

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    import cv2

    dataset = TartanairDataset(
        dataset_dir="/data/tartanair",
        sequence_size=1,
        sequence_skip=2,
        labels=["depth", "disparity"],
        envs=["abandonedfactory"],
        sequences=[["P001"]],
    )

    frame = dataset[100]

    left_view = frame["left"][0]
    right_view = frame["right"][0]
    left_view = left_view.resize((384, 384))
    print(left_view)
    depth = left_view.planar_depth
    disparity = left_view.disparity

    print(depth.data)

    viz_depth = depth.get_view(max=20)
    viz_disparity = disparity.get_view()

    cv2.imshow("depth", viz_depth)
    cv2.imshow("disparity", viz_disparity)
    cv2.waitKey(0)
