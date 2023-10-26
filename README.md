# nndepth
Depth Estimation by Neural Network


# 1. Introduction
- Depth estimation by neural network is now a very active domain thanks to its application in various domains: robotic, VR/AR, autonomous car, ... The implementation of many algorithms is very diverse and sometimes incomplete. The deployment (model exportation, execution script, ...) isn't always supported. Therefore, I implemented this repository to unify many depth estimation neural networks in a single repository to simplify the training - inference - deployment process.
- I mainly focus on Depth Estimation from Monocular and Stereo Cameras, particularly lightweight algorithms that can be deployed and run in real-time on edge devices such as mobile phones, robots, and AI Cameras.

# 2. Dependencies
- This repository is developed using [`aloception-oss`](https://github.com/Visual-Behavior/aloception-oss), an open-source package of VisualBehavior that allows for easy handling of the pipeline (data processing, training cycle, exportation, etc.) in many Computer Vision applications. You can take a look at `aloception-oss` to understand its basic concepts.
- The easiest way to install the working environment is to use Docker.
- Build the Docker image
```
docker build -t nndepth .
```
- Launch the Docker container
```
docker run  --gpus all --ipc host -e LOCAL_USER_ID=$(id -u)  -it --rm  -v MOUNT_YOUR_DISK  --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix IMAGE_ID
```

# 3. Modules
- The project is organized into different modules:
    - **blocks**: basic neural network blocks. Some blocks can be found: attention block, positional encoding, residual blocks, transformer, update block for RAFT-based models.
    - **datasets**: Data loader classes.
    - **disparity**: Module to train and infer different neural networks for Stereo depth estimation.
    - **extractors**: Backbones.


# Supported algorithms and road map
- [x] [RAFT-Stereo](https://arxiv.org/pdf/2109.07547.pdf)
- [x] [CreStereo](https://arxiv.org/abs/2203.11483)
- [x] [IGEV-Stereo](https://arxiv.org/pdf/2303.06615.pdf)
- [x] Data processing script
- [x] Inference script for Stereo module
- [ ] Integrate more datasets: Sceneflow, KITTI, [SANPO](https://blog.research.google/2023/10/sanpo-scene-understanding-accessibility.html), etc
- [ ] Implement evaluation script with some common metrics
- [ ] [GM-Stereo: Unifying Flow, Stereo and Depth Estimation](https://arxiv.org/pdf/2211.05783.pdf)
- [ ] [LEAStereo: Hierarchical Neural Architecture Search for Deep Stereo Matching](https://proceedings.neurips.cc/paper/2020/file/fc146be0b230d7e0a92e66a6114b840d-Paper.pdf)
- [ ] Lightweight CREStereo
- [ ] More lightweight feature extractor with pretrained weights
- [ ] [High-frequency Stereo Matching Network](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf)
- [ ] [MobileStereoNet](https://arxiv.org/pdf/2108.09770.pdf)
- [ ] [SCV-STEREO](https://arxiv.org/pdf/2107.08187.pdf)
- [ ] [DCVNet](https://arxiv.org/pdf/2103.17271.pdf)
- [ ] Lightweight Metric Monocular Depth Estimation based on [ZoeDepth](https://arxiv.org/abs/2302.12288)


# Acknowledgements
- [CreStereo](https://github.com/megvii-research/CREStereo)
