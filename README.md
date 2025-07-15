# nndepth
Depth Estimation by Neural Network

## 1. Introduction

Depth estimation using neural networks is a rapidly evolving field with applications across robotics, virtual and augmented reality (VR/AR), autonomous vehicles, and more. This repository addresses challenges in the fragmented landscape of depth estimation algorithms by:

1. Unifying multiple depth estimation neural networks in a single, cohesive framework.
2. Streamlining the entire pipeline from training to inference and deployment.
3. Focusing on both monocular and stereo camera-based depth estimation techniques.
4. Emphasizing lightweight algorithms suitable for real-time execution on edge devices.

🌟 **Key features** :star2:
- Comprehensive collection of state-of-the-art depth estimation algorithms
- Consistent implementation and interface across different models
- Simplified training, inference, and deployment processes
- Optimized for edge device compatibility and real-time performance
- Extensible architecture to easily incorporate new algorithms and techniques

## 2. Dependencies and Setup

### Clone
```bash
git clone git@github.com:anhtu293/nndepth.git
```

### Docker Installation (Recommended)

The easiest way to set up the working environment is by using Docker. Follow these steps:

1. Build the Docker image:
   ```bash
   cd docker && docker build -t nndepth -f Dockerfile.gpu .
   ```

   You can build the image with the version of pytorch and cuda you want.
   ```bash
   docker build -t nndepth --build-arg BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel --build-arg PYTHON_VERSION=3.10 --build-arg PYTORCH_VERSION=2.6.0 --build-arg TORCHVISION_VERSION=0.17.1 --build-arg TORCHAUDIO_VERSION=2.6.0 .
   ```

2. Launch the Docker container:
   ```bash
   xhost +si:localuser:root && docker run --gpus all --ipc host -e LOCAL_USER_ID=$(id -u) -it --rm \
     -v /PATH/TO/YOUR/DATASET:/data \
     -v /PATH/TO/NNDEPTH:/home/cv/nndepth \
     -v /home/YOUR_HOME/.config/:/home/cv/.config \
     -v /home/YOUR_HOME/.netrc:/home/cv/.netrc \
     --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix nndepth
   ```

   Replace `/PATH/TO/NNDEPTH` with the actual path to your nndepth directory.
   Replace `/PATH/TO/YOUR/DATASET` with the path to your dataset directory.
   `-v /home/YOUR_HOME/.config/:/home/cv/.config -v /home/YOUR_HOME/.netrc:/home/cv/.
   netrc` is useful if you want to track your training with **wandb**: these are necessary files which
   store the API key for **wandb**.


## 3. Project Structure
The project is organized into several key modules, each serving a specific purpose in the depth estimation pipeline:

1. **scene**: Core module for scene representation and manipulation:
   - Frame: Comprehensive representation of an image frame, including:
     - Raw image data
     - Camera intrinsics and extrinsics
     - Associated depth and disparity maps
     - Pose information
   - Depth: Robust depth map handling with features for:
     - Efficient resizing and interpolation
     - Customizable visualization options
     - Conversion to other formats (e.g., point clouds)
   - Disparity: Specialized class for stereo vision, offering:
     - Conversion between disparity and depth
     - Stereo-specific visualization tools
     - Compatibility with various stereo algorithms

2. **blocks**: Contains fundamental neural network building blocks, including:
   - Attention mechanisms
   - Positional encoding
   - Residual blocks
   - Transformer architectures
   - Update blocks for RAFT-based models

3. **datasets**: Houses data loader classes for various depth estimation datasets, ensuring efficient and standardized data handling across different algorithms.

4. **encoders**: Contains backbone architectures used as feature encoders in depth estimation models.

5. **models**: Implements complete depth estimation models, integrating components from other modules.

6. **utils**: Provides utility functions and helper classes used throughout the project.

This modular structure allows for easy maintenance, extensibility, and reusability of components across different depth estimation algorithms.



## 4. Training Pipeline
### 4.1 **Trainer** and **DataLoader**
- To maximize flexibility, our training pipeline is developed using PyTorch's native training loop instead of ready-to-use **Trainer** classes found in frameworks like `pytorch-lightning`. While this approach may slow down initial implementation and testing of new ideas, it allows for the implementation of complex features that might be challenging to achieve with pre-built Trainers.

- We provide a basic `BaseTrainer` class [here](nndepth/utils/base_trainer.py) with support functions, primarily for managing checkpoint names to ensure unique identifiers and consistent log directory formats. When creating a specific **Trainer** or **training loop**, you should inherit from this base class and implement the `train` and `validate` methods. An example implementation can be found [here](nndepth/models/raft_stereo/raft_trainer.py).

- We also offer a `BaseDataLoader` class [here](nndepth/utils/base_dataloader.py). This class includes two key members: `train_dataloader` and `val_dataloader`, which are PyTorch DataLoader instances for training and validation, respectively. When creating a custom dataloader for a specific training task, inherit from this base class and implement the `setup_train_dataloader` and `setup_val_dataloader` methods to initialize the respective dataloaders. An example implementation is available [here](nndepth/datasets/tartanair_disparity.py).


### 4.2 Launch training
- The training configurations are divided into three categories:
  1. Model configurations (e.g., model architecture, hyperparameters)
  2. Data configurations (e.g., dataset paths, data augmentation)
  3. Training configurations (e.g., learning rate, gradient accumulation, checkpoint save location)

  All of these configurations are defined in `Configuration` object which is a subclass of [BaseConfiguration](nndepth/utils/base_config.py). This class supports loading from YAML files and command line arguments. Command line arguments take priority over YAML values. Supports nested configurations.

- To initiate a training session, execute the following command:
```bash
python train.py --config_file PATH_TO_YAML_FILE --ARG_TO_OVERRIDE VALUE_TO_OVERRIDE
```

## 5. Supported algorithms and road map
- [x] [RAFT-Stereo](https://arxiv.org/pdf/2109.07547.pdf)
- [x] [CreStereo](https://arxiv.org/abs/2203.11483)
- [x] [IGEV-Stereo](https://arxiv.org/pdf/2303.06615.pdf)
- [x] Data processing script
- [x] Inference script
- [x] Evaluation script
- [x] More datasets: DIML, Hypersym
- [x] [MiDaS](https://arxiv.org/abs/1907.01341)
- [ ] More lightweight feature extractor with pretrained weights (MobilenetV4, mobileone, etc.)
- [ ] [DepthAnything](https://arxiv.org/pdf/2401.10891)
- [ ] [ZoeDepth](https://arxiv.org/abs/2302.12288)
- [ ] [Metric3D](https://jugghm.github.io/Metric3Dv2/)
- [ ] [Unidepth](https://github.com/lpiccinelli-eth/unidepth)

## 6. Acknowledgements
- [CreStereo](https://github.com/megvii-research/CREStereo)
- [RAFTStereo](https://github.com/princeton-vl/RAFT-Stereo/blob/main/evaluate_stereo.py#L13)
