# Stereo Depth Estimation
This module is used to implement - train - inference different neural networks for Stereo depth estimation. The models are implemented in `pytorch` and the training pipeline is implemented on top of `pytorch lightning`. The training can be logged via **wandb** or **tensorboard**.

<p align="center">
  <img src="../../images/tartanair_disp.png"/>
</p>

## 1. Structure
```
.
├── configs       --> Configuration files
├── criterions    --> Loss functions & metrics
├── data_loaders  --> Data loaders for training
├── __init__.py
├── models        --> Model implementations
├── README.md
├── scripts
├── trainers      --> Trainers
└── train.py      --> Training scripts
```

## 2. How to launch the training
- There are different script in `scripts` corresponding different training on different datasets.
- At the moment, only the training on TartanAir dataset is supported.
- Training arguments:
```
usage: train.py [-h] --model_config MODEL_CONFIG --data_config DATA_CONFIG --training_config TRAINING_CONFIG [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]

options:
  -h, --help            show this help message and exit
  --model_config MODEL_CONFIG
                        Path to model config file
  --data_config DATA_CONFIG
                        Path to data config file
  --training_config TRAINING_CONFIG
                        Path to training config file
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        Path to checkpoint to resume.
```
- Details of models and how to run the training with those models are below.

## 3. How to launch the inference
- Script `nndepth/disparity/scripts/inference.py`:
```
usage: inference.py [-h] --model_config_file MODEL_CONFIG_FILE --weights WEIGHTS --left_path LEFT_PATH --right_path RIGHT_PATH [--HW HW [HW ...]] --output OUTPUT [--render]
                    [--save_format {image,video}]

options:
  -h, --help            show this help message and exit
  --model_config_file MODEL_CONFIG_FILE
                        Path to model config file
  --weights WEIGHTS     Path to model weight
  --left_path LEFT_PATH
                        Path to directory of left images
  --right_path RIGHT_PATH
                        Path to directory of right images
  --HW HW [HW ...]      Model input size
  --output OUTPUT       Path to save output. Directory in case of save_format == image, mp4 file in case of video
  --render              Render results
  --save_format {image,video}
                        Which format to save output. image or video are supported. Default: video
```
- `--weights`: Path to model weights.
- The directory of images you want to infer must have below structure. Each images must have format `{ID}_{side}.png`.
- You can render the results directly use `--render`.
- There are 2 formats to save results:
  - image: Each frame & result will be saved as a png file.
  - video: All frames & results will be saved as a video.
```
.
├── image_left
│   ├── 0_left.png
│   ├── 1_left.png
│   ├── 2_left.png
│   ├── 3_left.png
│   └── 4_left.png
├── image_right
│   ├── 0_right.png
│   ├── 1_right.png
│   ├── 2_right.png
│   ├── 3_right.png
│   └── 4_right.png
```
- Detail commands & weights for each model are in the next section.

## 4. Supported models
- Trained weights and configuration can be found [here](https://drive.google.com/drive/folders/1hoOflbJ_75kmucyyN7eTwFT6le44oDuJ)
<details>
  <summary><b> RAFT-Stereo</b></summary>

  ## Architecture
  - Detail at [RAFT-Stereo](https://arxiv.org/pdf/2109.07547.pdf)
  <p align="center">
  <img src="../../images/raftstereo.png"/>
  </p>

- `ResNet50` & `RepViT` are used as backbone.

  ## Training command
```bash
python nndepth/disparity/scripts/train_disparity_on_tartanair.py --model_config nndepth/disparity/configs/models/BaseRAFTStereo.yml --data_config nndepth/disparity/configs/data/BaseRAFTStereo_Tartanair2DisparityModel.yml --accumulate_grad_batches 2 --lr 2e-4 --limit_val_batches 200 --val_check_interval 5000 --max_step 100000 --expe_name baseline --log --save
```

  ## Inference command
- Download checkpoint trained on TartanAir [here](https://drive.google.com/drive/folders/1OZIqRjqlF2fD4wwbMsFf5Lxx7ovYdu1D)

```bash
python nndepth/disparity/scripts/inference.py --model_config_file nndepth/disparity/configs/models/BaseRAFTStereo.yml --weights  disparity-BaseRAFTStereo-baseline.ckpt --left_path samples/stereo/left/ --right_path samp
les/stereo/right/ --HW 480 640  --output test --save_format image
```

</details>

<details>
  <summary><b> CreStereo</b></summary>

  ## Architecture
  - Detail at [CreStereo](https://arxiv.org/abs/2203.11483)
  <p align="center">
  <img src="../../images/crestereo.png"/>
  </p>

- `ResNet50` is used as backbone.

  ## Training command
```bash
python nndepth/disparity/scripts/train_disparity_on_tartanair.py --model_config nndepth/disparity/configs/models/CREStereoBase.yml --data_config nndepth/disparity/configs/data/CREStereoBase_Tartanair2DisparityModel.yml --accumulate_grad_batches 2 --lr 2e-4 --limit_val_batches 200 --val_check_interval 5000 --max_step 100000 --expe_name baseline --log --save
```

  ## Inference command
- Download checkpoint trained on TartanAir [here](https://drive.google.com/drive/folders/1fTlVDc3NHCeiFTfOKZ_keQGlkmrgJPYG)
```bash
python nndepth/disparity/scripts/inference.py --model_config_file nndepth/disparity/configs/models/CREStereoBase.yml --weights disparity-CREStereoBase-baseline.ckpt --left_path samples/stereo/left/ --right_path samples/stereo/right/ --HW 480 640  --output test --save_format image
```
</details>

<details>
  <summary><b> IGEV Stereo</b></summary>

  ## Architecture
  - Detail at [IGEV-Stereo](https://arxiv.org/pdf/2303.06615.pdf)
  <p align="center">
  <img src="../../images/igev.png"/>
  </p>

- `MobilenetLarge-V3` is used as backbone.

  ## Training command
```bash
python nndepth/disparity/scripts/train_disparity_on_tartanair.py --model_config nndepth/disparity/configs/models/CREStereoBase.yml --data_config nndepth/disparity/configs/data/CREStereoBase_Tartanair2DisparityModel.yml --accumulate_grad_batches 2 --lr 2e-4 --limit_val_batches 200 --val_check_interval 5000 --max_step 100000 --expe_name baseline --log --save
```

</details>
