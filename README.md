# nndepth
Depth Estimation by Neural Network


# 1. Introduction
# 1. Introduction

Depth estimation using neural networks has become a rapidly evolving field with applications across various domains, including robotics, virtual and augmented reality (VR/AR), autonomous vehicles, and more. However, the landscape of depth estimation algorithms is diverse and often fragmented, with implementations varying in completeness and deployment support. This repository aims to address these challenges by:

1. Unifying multiple depth estimation neural networks in a single, cohesive framework.
2. Streamlining the entire pipeline from training to inference and deployment.
3. Focusing on both monocular and stereo camera-based depth estimation techniques.
4. Emphasizing lightweight algorithms suitable for real-time execution on edge devices such as mobile phones, robots, and AI cameras.

Key features of this repository:
- Comprehensive collection of state-of-the-art depth estimation algorithms
- Consistent implementation and interface across different models
- Simplified training, inference, and deployment processes
- Optimized for edge device compatibility and real-time performance
- Extensible architecture to easily incorporate new algorithms and techniques

By providing a unified platform for depth estimation, this project aims to accelerate research, development, and deployment of depth perception systems across a wide range of applications.

# 2. Dependencies
## Dependencies and Setup

This project is built upon [`aloception-oss`](https://github.com/Visual-Behavior/aloception-oss), an open-source package developed by VisualBehavior. It provides a comprehensive framework for handling various aspects of Computer Vision applications, including data processing, training cycles, and model exportation. We recommend familiarizing yourself with the basic concepts of `aloception-oss` to better understand this project's structure.

### Docker Installation (Recommended)

The easiest way to set up the working environment is by using Docker. Follow these steps:

1. Build the Docker image:
   ```
   docker build -t nndepth .
   ```

2. Launch the Docker container:
   ```bash
   docker run --gpus all --ipc host -e LOCAL_USER_ID=$(id -u) -it --rm -v /PATH/TO/YOUR/DATASET:/data -v /PATH/TO/NNDEPTH:/home/aloception/nndepth --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix nndepth
   ```

   Replace `/PATH/TO/NNDEPTH` with the actual path to your nndepth directory.
   Replace `/PATH/TO/YOUR/DATASET` with the path to your dataset directory.

   For the 1st time you launch this container, you need to do this additional step
   ```bash
   cd /home/aloception/nndepth && pip install -e aloception-oss
   ```


This setup ensures that you have all the necessary dependencies and a consistent environment for development and experimentation.

# 3. Project Structure

The project is organized into several key modules, each serving a specific purpose in the depth estimation pipeline:

1. **blocks**: Contains fundamental neural network building blocks, including:
   - Attention mechanisms
   - Positional encoding
   - Residual blocks
   - Transformer architectures
   - Update blocks for RAFT-based models

2. **datasets**: Houses data loader classes for various depth estimation datasets, ensuring efficient and standardized data handling across different algorithms.

3. **disparity**: Encompasses modules for training and inferencing various neural networks specifically designed for stereo depth estimation.

4. **extractors**: Contains backbone architectures used as feature extractors in depth estimation models.

5. **models**: Implements complete depth estimation models, integrating components from other modules.

6. **utils**: Provides utility functions and helper classes used throughout the project.

7. **scripts**: Contains various scripts for data preprocessing, model evaluation, and other auxiliary tasks.

This modular structure allows for easy maintenance, extensibility, and reusability of components across different depth estimation algorithms.


# Supported algorithms and road map
- [x] [RAFT-Stereo](https://arxiv.org/pdf/2109.07547.pdf)
- [x] [CreStereo](https://arxiv.org/abs/2203.11483)
- [x] [IGEV-Stereo](https://arxiv.org/pdf/2303.06615.pdf)
- [x] Data processing script
- [x] Inference script for Stereo module
- [ ] More lightweight feature extractor with pretrained weights
- [ ] Integrate more datasets: Sceneflow, KITTI, [SANPO](https://blog.research.google/2023/10/sanpo-scene-understanding-accessibility.html), etc
- [ ] Implement evaluation script with some common metrics
- [ ] [GM-Stereo: Unifying Flow, Stereo and Depth Estimation](https://arxiv.org/pdf/2211.05783.pdf)
- [ ] [LEAStereo: Hierarchical Neural Architecture Search for Deep Stereo Matching](https://proceedings.neurips.cc/paper/2020/file/fc146be0b230d7e0a92e66a6114b840d-Paper.pdf)
- [ ] [MobileStereoNet](https://arxiv.org/pdf/2108.09770.pdf)
- [ ] [SCV-STEREO](https://arxiv.org/pdf/2107.08187.pdf)
- [ ] [DCVNet](https://arxiv.org/pdf/2103.17271.pdf)
- [ ] [MiDaS](https://arxiv.org/abs/1907.01341)
- [ ] [ZoeDepth](https://arxiv.org/abs/2302.12288)
- [ ] [Metric3D](https://jugghm.github.io/Metric3Dv2/)
- [ ] [Unidepth](https://github.com/lpiccinelli-eth/unidepth)
- [ ] [DepthAnything](https://arxiv.org/pdf/2401.10891)


# Acknowledgements
- [CreStereo](https://github.com/megvii-research/CREStereo)
