# Dataset

## 1. TartanAir
- Information: https://theairlab.org/tartanair-dataset/
### Download:
- The original toolkit to download the dataset: https://github.com/castacks/tartanair_tools. Clone this repository.
- Because their dataset is huge, you should use `azcopy`. You can download `azcopy` from [this guide](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10?tabs=dnf#download-the-azcopy-portable-binary) (Download the AzCopy portable binary) and put it in the toolkit repository that you just downloaded.
- Replace the script `download_training.py` by this script: [`download_tartanair.py`](./utils/download_tartanair.py).
- Launch the command
```bash
python download_training.py --output-dir WHERE_TO_PUT_DS --rgb --depth --flow --seg --only-easy --azcopy
```
- After downloading, the zip files will be located in the `--output-dir`. To extract the dataset efficiently, use the provided script [`unzip_tartanair.sh`](./utils/unzip_tartanair.sh). Place this script in the `--output-dir` and execute it by running `./unzip_tartanair.sh`.

- Once the dataset is successfully uncompressed, you can delete the zip files to free up storage space. The extracted dataset will have the following directory structure:
```
.
├── abandonedfactory
│   └── abandonedfactory
│       └── Easy
│           ├── P000
│           ...
├── abandonedfactory_night
│   └── abandonedfactory_night
│       └── Easy
│           ├── P001
│           ...
│           ...
├── endofworld
│   └── endofworld
│       └── Easy
│           ├── P000
│           ...
├── gascola
│   └── gascola
│       └── Easy
│           ├── P001
│           ...
├── ocean
│   └── ocean
│       └── Easy
│           ├── P000
│           ...
├── office
│   └── office
│       └── Easy
│           ├── P000
│           ...
...

260 directories, 0 files
```
- `abandonedfactory`, `office`, `neighborhood`, ... are called **environment**.
- `Easy` is **level**
- `P001`, ... are **sequences**. In side each **sequence**, you will have following data
```
.
├── depth_left
├── image_left
├── image_right
├── pose_left.txt
├── pose_right.txt
└── seg_left
```
### Read dataset
- In order to read the dataset, there is [TartanairDataset](./tartanair_dataset.py). This dataset class is also used to create dataloader for training models.

**Example**
```
>>> import cv2
>>> dataset = TartanairDataset(
        dataset_dir="/data/tartanair",
        sequence_size=1,
        sequence_skip=2,
        labels=["depth", "disparity"],
        envs=["abandonedfactory"],
        sequences=[["P001"]],
    )
>>> frame = dataset[100]
>>> left_frame = frame["left"][0]
>>> image_left = left_frame.image.numpy().transpose((1, 2, 0)).astype(np.uint8)
>>> depth = left_frame.planar_depth
>>> disparity = left_frame.disparity
>>> viz_depth = depth.get_view(max=20)
>>> viz_disparity = disparity.get_view()
>>> cv2.imshow("image", image_left)
>>> cv2.imshow("depth", viz_depth)
>>> cv2.imshow("disparity", viz_disparity)
```
- The TartanAir dataset is organized as sequences of frames, allowing for flexible loading of consecutive frames. Two key parameters control this:

  - `sequence_size`: Specifies how many consecutive frames to load at once. For example, setting `sequence_size=3` would load 3 frames in a sequence.

  - `sequence_skip`: Determines the interval between loaded frames. For instance, with `sequence_skip=2`, the dataset would load frames at time t, t+2, t+4, etc.

This setup enables various use cases:
- Loading single frames: Set `sequence_size=1` and `sequence_skip=0`
- Loading consecutive frames: Set `sequence_size>1` and `sequence_skip=0`
- Loading frames with gaps: Set `sequence_size>1` and `sequence_skip>0`

This flexibility is particularly useful for tasks that require temporal information, such as optical flow estimation or video-based depth estimation.

- The dataset returns a dictionary containing lists of `Frame` objects for each camera view (e.g., `{"left": [frame_left1, frame_left2], "right": [frame_right1, frame_right2]}`).
- The `Frame` object is a powerful abstraction that encapsulates an image along with its associated metadata, including:
  - Disparity
  - Depth
  - Intrinsic camera parameters
  - Extrinsic camera parameters
  - Pose information
- Key benefits of using `Frame` objects:
  1. Simplified data handling: All relevant information for a single frame is contained in one object.
  2. Automatic consistency maintenance: When operations like resizing are performed, the `Frame` object automatically adjusts related parameters (e.g., disparity, intrinsic matrix) to maintain consistency.
  3. Encapsulation of common operations: The `Frame` class provides methods for common image processing tasks, ensuring correct handling of all associated data.
- For a comprehensive understanding of the `Frame` object and its capabilities, please refer to the [Scene object documentation](../scene/README.md).

## 2. Kitti
### Download
- Download [here](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).
- Unzip everything and put them in the structure below
```
.
├── testing
│   ├── calib_cam_to_cam
│   ├── calib_imu_to_velo
│   ├── calib_velo_to_cam
│   ├── image_2
│   └── image_3
└── training
    ├── calib_cam_to_cam
    ├── calib_imu_to_velo
    ├── calib_velo_to_cam
    ├── disp_noc_0
    ├── disp_noc_1
    ├── disp_occ_0
    ├── disp_occ_1
    ├── flow_noc
    ├── flow_occ
    ├── image_2
    ├── image_3
    ├── obj_map
    ├── viz_flow_occ
    └── viz_flow_occ_dilate_1
```

### Read dataset
- [KittiStereo2015](./kitti_stereo_2015.py).

**Example**
```
>>> # Visualize a sample
>>> from random import randint
>>> import cv2
>>> dataset = KittiStereo2015(sequence_start=10, sequence_end=10)
>>> obj = dataset[randint(0, len(dataset))]
>>> frame = obj["left"][0]
>>> disps = torch.stack([frame.disparity, frame.disparity], dim=0)
>>> disp_viz = disps.get_view()
>>> cv2.imshow("disp", disp_viz[0])
```
- Similar to the **TartanAir** dataset, it's important to be familiar with the **Scene** object concept when working with the KITTI dataset.
