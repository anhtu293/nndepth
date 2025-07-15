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


## 3. DIML
### Download
- Download [here](https://dimlrgbd.github.io/).

### Structure
- Unzip everything.
- The **DIML** dataset is very messy, you should put them in the structure below in order to read the dataset with the script [DIMLDataset](./diml.py).
```
.
├── indoor
│   ├── 04. Church
│   ├── 05. Computer Room
│   ├── 07. Library
│   ├── 1
│   ├── 10. Corridor
│   ├── 12_livingroom_2
│   ├── 15_restaurant_1
│   ├── 15_restaurant_2
│   ├── 16. Billiard Hall
│   ├── 2
│   ├── 3
│   ├── 4
│   └── others
└── outdoor
    ├── 130729_C0
    ├── 160319_C1
    ├── 160326_C0
    ├── 160408_C0
    ├── 160517_C0
    ├── 160627_C1
    ├── 160715_C1
    ├── 160725_C2
    ├── 160726_C0
    ├── 160727_C1
    ├── 160730_C1
    ├── 160801_C0
    ├── 161117_C0
    └── 170721_C0
```

> [!NOTE]
> `1`, `2`, `3`, `4`, `04. Church`, `05. Computer Room`, `07. Library`, `10. Corridor`, `12_livingroom_2`, `15_restaurant_1`, `15_restaurant_2`, `16. Billiard Hall` are the folders original folders in the zip files.

> [!NOTE]
> The `others` is the folder that I created to put the following folders (which are in the zip files): `16.01.19`, `16.01.20`, `16.01.25`, `16.01.26`, `16.01.27`, `16.01.28`, `16.01.29`, `16.02.01`, `16.02.02`, `16.02.03`, `16.02.04`, `16.02.05`, `16.02.06`, `16.02.15`, `16.02.16`, `16.02.17`, `16.02.19`, `16.02.25`, `16.03.03`, `16.03.15`, `16.03.17`, `16.03.22`. With this structure, all the folders in `indoor` subset have the same structure. For example, the `2` folder in `indoor` has the following structure:
```
2
├── 16.02.02
│   ├── 1
│   │   ├── col
│   │   ├── raw_png
│   │   ├── up_png
│   │   └── warp_png
│   ├── 2
│   │   ├── col
│   │   ├── raw_png
│   │   ├── up_png
│   │   └── warp_png
│   ├── 3
│   │   ├── col
│   │   ├── raw_png
│   │   ├── up_png
│   │   └── warp_png
```

> [!NOTE]
> For `16. Billiard Hall`, I must create a folder named `unknown` and put the `7` folder in it to respect the structure of the `indoor` subset.
```
16. Billiard Hall
└── unknown
    └── 7
        ├── col
        ├── raw_png
        ├── up_png
        └── warp_png
```

- Each folder in `outdoor` subset has the following structure:
```
.
├── 130729_C0
│   ├── A
│   ├── B
│   ├── ...
│   └── out_zed_160729_C0_campara.mat
```

> [!NOTE]
> The `160627_C1`: I removed this folder because it does not have the same structure as other folders in the `outdoor` subset.

## 4. HR-WSI
### Download
- Download the HR-WSI dataset and extract it to your desired location.

### Structure
- The [HR-WSI dataset](https://github.com/KexianHust/Structure-Guided-Ranking-Loss) has a simple and clean structure with train/val splits. After extraction, organize the dataset as follows:
```
.
├── train
│   ├── imgs
│   ├── gts
│   ├── valid_masks
│   └── instance_masks
└── val
    ├── imgs
    ├── gts
    ├── valid_masks
    └── instance_masks
```

- Each split contains:
  - `imgs/`: RGB images in `.jpg` format
  - `gts/`: Ground truth depth maps in `.png` format
  - `valid_masks/`: Valid pixel masks in `.png` format
  - `instance_masks/`: Instance segmentation masks in `.png` format

## 5. Hypersim
- Information: https://github.com/apple/ml-hypersim

### Download
- Follow the instructions in the [official repository](https://github.com/apple/ml-hypersim) to download the dataset.

### Structure
- The Hypersim dataset consists of 461 indoor scenes with 77,400 images total.
- Each scene has a name format `ai_VVV_NNN` where `VVV` is the volume number and `NNN` is the scene number.
- The download script will download the zip and automatically unzip it. It should have the following structure.
```
ai_001_003
├── _detail
│   ├── cam_00
│   ├── mesh
│   ├── metadata_cameras.csv
│   ├── metadata_nodes.csv
│   ├── metadata_node_strings.csv
│   └── metadata_scene.csv
└── images
    ├── scene_cam_00_final_hdf5
    ├── scene_cam_00_final_preview
    ├── scene_cam_00_geometry_hdf5
    └── scene_cam_00_geometry_preview
...
...
```

- `scene_cam_00_final_hdf5` contains raw distance data.
- `scene_cam_00_final_preview` contains the preview image.
- `scene_cam_00_geometry_hdf5` contains the geometry data: Distance (:warning: this is not depth but the distance between the camera optical center and the object. Our code will convert it to depth.), Normal, ...
- `scene_cam_00_geometry_preview` contains the geometry preview image.
