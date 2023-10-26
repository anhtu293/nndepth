import numpy as np
import torch
import os
from typing import List, Union, Callable

from alodataset import BaseDataset, SequenceMixin
import aloscene

from nndepth.datasets.utils.geometry_trans import pos_quat2SE


def sequence_index(start, seq_size):
    end = start + (seq_size - 1)
    return np.linspace(start, end, seq_size).astype(int).tolist()


def sequence_indices(n_samples, seq_size, seq_skip):
    for start in range(0, n_samples - (seq_size - 1), seq_skip + 1):
        yield sequence_index(start, seq_size)


class TartanairDataset(BaseDataset, SequenceMixin):
    """
    TartanAirDataset

    Parameters
    ----------

    sequences : list
        list of tuple of the following format [environment]
    cameras : list of string
        One frame is returned for each camera.
        Possible cameras : {'left'}.
    labels : list
        Labels that will be attached to the frame
        Possible labels : {'disp', 'disp_occ', 'flow', 'flow_occ'}
    pose_format : {"NED", "camera"}
        "NED" (default) : original NED format from tartanair dataset
        "camera" : our camera coordinate system (x=right, y=down, z=infront)
    **kwargs :
        aloscene.BaseDataset parameters
    """

    CAMERAS = ["left", "right"]
    LABELS = ["disp", "depth", "segmentation"]

    def __init__(
        self,
        envs: List[str] = None,
        sequences: List[str] = None,
        cameras: List[str] = None,
        labels: List[str] = None,
        start_from: int = 0,
        pose_format: str = "NED",
        get_cameras_fn: Callable = None,
        **kwargs,
    ):
        super().__init__(name="TartanAir", **kwargs)
        self.envs = envs
        self.sequences = sequences
        self.cameras = cameras if cameras is not None else self.CAMERAS
        self.labels = labels if labels is not None else self.LABELS
        self.start_from = start_from
        self.images_format = "png"
        self.items = {}
        self.get_cameras_fn = get_cameras_fn

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

            # remove last item
            if "flow" in self.labels or "flow_occ" in self.labels:
                if self.skip != 0:
                    print("[WARNING][tartandataset] Flow label is loaded, but skip != 0.")
                del self.items[len(self.items) - 1]

    def _load_camera_poses(self, data_dir):
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

    def _get_envs(self):
        environments = []
        for env in sorted(os.listdir(self.dataset_dir)):
            if not env.endswith(".zip") and env != "tartanair_tools":
                environments.append(env)
        return environments

    def _get_sequences(self, env):
        diff_dir = os.path.join(self.dataset_dir, env, env, "Easy")
        all_sequences = os.listdir(diff_dir)
        all_sequences = [
            seq for seq in all_sequences if os.path.isdir(os.path.join(self.dataset_dir, env, env, "Easy", seq))
        ]
        if self.sequences is not None:
            idx = self.envs.index(env)
            sequences = self.sequences[idx]
            sequences = [seq for seq in sequences if seq in all_sequences]
        else:
            sequences = all_sequences
        return sorted(sequences)

    def _get_items(self, envs):
        if envs is None:
            envs = self._get_envs()

        env_seq_level = []
        for env in envs:
            seqs = self._get_sequences(env)
            env_seq_level.extend([[env, seq, "Easy"] for seq in seqs])
        return env_seq_level

    def get_frame(self, side, sub_sequence_folder, frame_id):
        # Frame + Camera Intrinsic
        frame_left_path = os.path.join(sub_sequence_folder, f"image_{side}", f"{frame_id:06d}_{side}.png")
        frame = aloscene.Frame(frame_left_path, camera_side=side, baseline=0.25)
        intrinsic = aloscene.CameraIntrinsic(focal_length=320, plane_size=frame.HW)
        frame.append_cam_intrinsic(intrinsic)

        cam_ext = np.eye(4)
        cam_ext[1, -1] = -0.125 if side == "left" else 0.125
        cam_ext = aloscene.CameraExtrinsic(cam_ext)
        frame.append_cam_extrinsic(cam_ext)

        # Depth
        # right depth is not available for annotation at the moment
        if "depth" in self.labels and side == "left":
            depth_path = os.path.join(
                sub_sequence_folder,
                f"depth_{side}",
                f"{frame_id:06d}_{side}_depth.npy",
            )
            depth = aloscene.Depth(
                np.expand_dims(np.load(depth_path), axis=0),
                baseline=0.25,
                camera_side=side,
            )
            depth.append_cam_intrinsic(intrinsic)
            frame.append_depth(depth)

        # load disp from depth
        if "disp" in self.labels and side == "left":
            depth_path = os.path.join(
                sub_sequence_folder,
                f"depth_{side}",
                f"{frame_id:06d}_{side}_depth.npy",
            )
            depth = aloscene.Depth(
                np.expand_dims(np.load(depth_path), axis=0),
                baseline=0.25,
                camera_side=side,
            )
            depth.append_cam_intrinsic(intrinsic)
            frame.append_disparity(depth.as_disp(camera_side=side).signed())

        # Assume only left is visible. Maybe it is right. we'll check later.
        if side == "left" and ("flow" in self.labels or "flow_occ" in self.labels):
            # Optical flow
            flow_path = os.path.join(
                sub_sequence_folder,
                "flow",
                f"{frame_id:06d}_{frame_id+self.skip+1:06d}_flow.npy",
            )
            if os.path.exists(flow_path):
                flow = np.transpose(np.load(flow_path), [2, 0, 1])
                flow = aloscene.Flow(flow)
                if "flow_occ" in self.labels:
                    mask_path = os.path.join(
                        sub_sequence_folder,
                        "flow",
                        f"{frame_id:06d}_{frame_id+self.skip+1:06d}_mask.npy",
                    )
                    mask = np.expand_dims(np.load(mask_path), 0) > 0
                    occ = aloscene.Mask(mask, names=("N", "H", "W"))
                    flow.append_occlusion(occ)
                frame.append_flow(flow, "flow_forward")

        return frame

    def NED2camera(self, P):
        T = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        T_inv = np.linalg.inv(T)
        return T @ P @ T_inv

    def getitem(self, idx):
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

        target_cameras = self.cameras if self.get_cameras_fn is None else self.get_cameras_fn()
        frames = {side: [] for side in target_cameras}

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
            for side in target_cameras:
                frame = self.get_frame(side, sub_sequence_folder, el)
                P = aloscene.Pose(torch.as_tensor(poses[side], dtype=torch.float32))
                frame.append_pose(P)
                frames[side].append(frame.temporal())

        for side in frames:
            frames[side] = torch.cat(frames[side], dim=0)

        return frames

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    from aloscene.renderer import Renderer

    dataset = TartanairDataset(
        sequence_size=1,
        sequence_skip=2,
        labels=["depth_left"],
        envs=["abandonedfactory"],
        sequences=[["P001"]],
    )

    for frame in dataset.stream_loader():
        left_view = frame["left"].get_view()
        right_view = frame["right"].get_view()
        Renderer().render([left_view, right_view], renderer="matplotlib")
        break
