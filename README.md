# nndepth
Depth Estimation by Neural Network

## 1. Introduction

Depth estimation using neural networks is a rapidly evolving field with applications across robotics, virtual and augmented reality (VR/AR), autonomous vehicles, and more. This repository addresses challenges in the fragmented landscape of depth estimation algorithms by:

1. Unifying multiple depth estimation neural networks in a single, cohesive framework.
2. Streamlining the entire pipeline from training to inference and deployment.
3. Focusing on both monocular and stereo camera-based depth estimation techniques.
4. Emphasizing lightweight algorithms suitable for real-time execution on edge devices.

Key features:
- Comprehensive collection of state-of-the-art depth estimation algorithms
- Consistent implementation and interface across different models
- Simplified training, inference, and deployment processes
- Optimized for edge device compatibility and real-time performance
- Extensible architecture to easily incorporate new algorithms and techniques

## 2. Dependencies and Setup

This project is built upon [`aloception-oss`](https://github.com/Visual-Behavior/aloception-oss), an open-source package by VisualBehavior. Familiarity with `aloception-oss`, especially `aloscene`, is recommended.

### Docker Installation (Recommended)

The easiest way to set up the working environment is by using Docker. Follow these steps:

1. Build the Docker image:
   ```bash
   docker build -t nndepth .
   ```

2. Launch the Docker container:
   ```bash
   docker run --gpus all --ipc host -e LOCAL_USER_ID=$(id -u) -it --rm \
     -v /PATH/TO/YOUR/DATASET:/data \
     -v /PATH/TO/NNDEPTH:/home/aloception/nndepth \
     -v /home/YOUR_HOME/.config/:/home/aloception/.config \
     -v /home/YOUR_HOME/.netrc:/home/aloception/.netrc \
     --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix nndepth
   ```

   Replace `/PATH/TO/NNDEPTH` with the actual path to your nndepth directory.
   Replace `/PATH/TO/YOUR/DATASET` with the path to your dataset directory.
   `-v /home/YOUR_HOME/.config/:/home/aloception/.config -v /home/YOUR_HOME/.netrc:/home/aloception/.
   netrc` is useful if you want to track your training with **wandb**: these are necessary files which
   store the API key for **wandb**.

3. For the 1st time you launch this container, you need to do this additional step
   ```bash
   cd /home/aloception/nndepth && pip install -e aloception-oss
   ```

## 3. Project Structure
The project is organized into several key modules, each serving a specific purpose in the depth estimation pipeline:

1. **blocks**: Contains fundamental neural network building blocks, including:
   - Attention mechanisms
   - Positional encoding
   - Residual blocks
   - Transformer architectures
   - Update blocks for RAFT-based models

2. **datasets**: Houses data loader classes for various depth estimation datasets, ensuring efficient and standardized data handling across different algorithms.

3. **disparity**: Encompasses modules for training and inferencing various neural networks specifically designed for stereo depth estimation. Read this [documentation](nndepth/disparity/README.md) for details.

4. **extractors**: Contains backbone architectures used as feature extractors in depth estimation models.

5. **models**: Implements complete depth estimation models, integrating components from other modules.

6. **utils**: Provides utility functions and helper classes used throughout the project.

7. **scripts**: Contains various scripts for data preprocessing, model evaluation, and other auxiliary tasks.

This modular structure allows for easy maintenance, extensibility, and reusability of components across different depth estimation algorithms.


## 4. Training Pipeline
### 4.1 **Trainer** and **DataLoader**
- To maximize flexibility, our training pipeline is developed using PyTorch's native training loop instead of ready-to-use **Trainer** classes found in frameworks like `pytorch-lightning`. While this approach may slow down initial implementation and testing of new ideas, it allows for the implementation of complex features that might be challenging to achieve with pre-built Trainers.

- For distributed training, we leverage Hugging Face's `accelerate` library, which simplifies the process.

- We provide a basic `BaseTrainer` class [here](nndepth/utils/base_trainer.py) with support functions, primarily for managing checkpoint names to ensure unique identifiers and consistent log directory formats. When creating a specific **Trainer** or **training loop**, you should inherit from this base class and implement the `train` and `validate` methods. An example implementation can be found [here](nndepth/disparity/train.py).

- We also offer a `BaseDataLoader` class [here](nndepth/utils/base_dataloader.py). This class includes two key members: `train_dataloader` and `val_dataloader`, which are PyTorch DataLoader instances for training and validation, respectively. When creating a custom dataloader for a specific training task, inherit from this base class and implement the `setup_train_dataloader` and `setup_val_dataloader` methods to initialize the respective dataloaders. An example implementation is available [here](nndepth/disparity/data_loaders/tartanair_disparity.py).


### 4.2 Launch training
- The training configurations are divided into three categories:
  1. Model configurations
  2. Data configurations
  3. Training configurations (e.g., learning rate, gradient accumulation, checkpoint save location)

  Each category corresponds to a specific object (**Model**, **Data loader**, **Trainer**) and is defined in a separate YAML configuration file.

- To initiate a training session, execute the following command:
```bash
python train.py --model_config PATH --data_config PATH --training_config PATH
```
- The `train.py` script follows a general structure as outlined below:
```python
import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from nndepth.utils.common import add_common_args, instantiate_with_config_file
from nndepth.utils.trackers.wandb import WandbTracker

from nndepth.disparity.criterions import RAFTCriterion


def main():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser) # Common arguments which are `model_config`, `data_config` and `training_config`
    args = parser.parse_args()

    # Instantiate the model, dataloader and trainer
    # the function instantiate_with_config_file is used to init an object based on the path to configuration and the path to script where the class is defined
    model, model_config = instantiate_with_config_file(args.model_config, "nndepth.disparity.models")
    dataloader, data_config = instantiate_with_config_file(args.data_config, "nndepth.disparity.data_loaders")
    trainer, training_config = instantiate_with_config_file(args.training_config, "nndepth.disparity.trainers")

    # Must run `.setup()` to setup training dataloader and val dataloader
    dataloader.setup()

    # Init the criterion, optimizer, scheduler here
    # All of these components should be defined yourself to maximize the flexibility.

    # Setup the tracker
    grouped_configs = {"model": model_config, "data": data_config, "training": training_config}
    wandb_tracker = WandbTracker(
        project_name=trainer.project_name,
        run_name=trainer.experiment_name,
        root_log_dir=trainer.artifact_dir,
        config=grouped_configs,
        resume=args.resume_from_checkpoint is not None,
    )

    # Resume from checkpoint if required
    if args.resume_from_checkpoint is not None:
        trainer.resume_from_checkpoint(args.resume_from_checkpoint)

    # Train the model
    trainer.train(
      ...
    )


if __name__ == "__main__":
    main()

```

### 4.3 How to create a correct yaml configuration
### 4.3 Creating YAML Configuration Files

- Each YAML file defines a single specific object (e.g., model, data loader, trainer).
- We provide a utility tool to generate YAML configuration files for specific objects, which can then be customized for training. The tool is located at [nndepth/utils/create_config_file.py](nndepth/utils/create_config_file.py).
- How to launch
```bash
python create_config_file.py --module_path MODULE_PATH --cls_name CLS_NAME --save_path SAVE_PATH --ignore_base_classes CLASSES_TO_IGNORE
```
- **MODULE_PATH**: Path to the script containing the class implementation.
- **CLS_NAME**: Name of the class for which to create the YAML configuration.
- **SAVE_PATH**: Destination path for saving the generated configuration file.
- **CLASSES_TO_IGNORE**: List of parent classes whose arguments should be excluded from the configuration file. This is useful for ignoring common base classes (e.g., `nn.Module`) in nested inheritance structures.

The tool generates a YAML configuration file that includes:
1. All necessary arguments to configure an object of the specified class.
2. Documentation from the class's `__init__` method docstring, providing context and explanation for each configuration option.



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
- [ ] [LightStereo](https://arxiv.org/abs/2406.19833)
- [ ] [MiDaS](https://arxiv.org/abs/1907.01341)
- [ ] [ZoeDepth](https://arxiv.org/abs/2302.12288)
- [ ] [Metric3D](https://jugghm.github.io/Metric3Dv2/)
- [ ] [Unidepth](https://github.com/lpiccinelli-eth/unidepth)
- [ ] [DepthAnything](https://arxiv.org/pdf/2401.10891)



# Acknowledgements
- [CreStereo](https://github.com/megvii-research/CREStereo)
