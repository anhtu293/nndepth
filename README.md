# nndepth
Depth Estimation by Neural Network


# 1. Introduction
- Depth estimation by neural network is now a very active domain thanks to its application in various domains: robotic, VR/AR, autonomous car, ... The implementation of many algorithms is very diversed and sometime incomplete. The deployment (model exportation, execution script, ...) isn't always supported. Therefore, I implement this repository to unify many depth estimation neural network in a repository to simplify the training - inference - deployment.
- I mainly focus on Depth Estimation from Monocular and Stereo Camera, particularly lightweight algorithms which can be deployed and run in realtime on edge device: mobile phone, robots, AI Cameras.

# 2. Dependencies
- This repository is developed using [`aloception-oss`](https://github.com/Visual-Behavior/aloception-oss), an open source package of VisualBehavior which allows to handle easily the pipeline (data processing, training cycle, exportation, ...) in many Computer Vision application. You can take a look at `aloception-oss` to understand its basic concepts.
- The easiest way to install working environment is to use Docker.
- Build docker image
```
docker build -t nndepth .
```
- Launch docker container
```
docker run  --gpus all --ipc host -e LOCAL_USER_ID=$(id -u)  -it --rm  -v MOUNT_YOUR_DISK  --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix IMAGE_ID
```

# 3. Modules
- The project is organized in different modules:
    - **blocks**: basic neural network blocks. Some blocks can be found: attention block, positional encoding, residual blocks, transformer, update block for RAFT-based models.
    - **datasets**: Data loader classes.
    - **disparity**: Module to train - inference different neural networks for Stereo depth estimation.
    - **extractors**: Backbones.


# Supported algorithms and road map
- [x] [RAFT-Stereo](https://arxiv.org/pdf/2109.07547.pdf)
- [x] [CreStereo](https://arxiv.org/abs/2203.11483)
- [x] [IGEV-Stereo](https://arxiv.org/pdf/2303.06615.pdf)
- [x] Data processing script
- [ ] Inference script and Exportation script for Stereo module
- [ ] [MobileStereoNet](https://arxiv.org/pdf/2108.09770.pdf)
- [ ] [SCV-STEREO](https://arxiv.org/pdf/2107.08187.pdf)
- [ ] [DCVNet](https://arxiv.org/pdf/2103.17271.pdf)
- [ ] Lightweight Metric Monocular Depth Estimation based on [ZoeDepth](https://arxiv.org/abs/2302.12288)


# Acknowledgements
- [CreStereo](https://github.com/megvii-research/CREStereo)
