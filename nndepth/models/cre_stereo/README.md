# CRE-Stereo

## Architecture
- Detail at [CreStereo](https://arxiv.org/abs/2203.11483)
<p align="center">
<img src="../../../images/crestereo.png"/>
</p>

- `ResNet50` is used as backbone.

## Training Configuration
```yaml
model:
  fnet_cls: basic_encoder
  update_cls: basic_update_block
  iters: 12
  max_disp: 192
  num_fnet_channels: 256
  hidden_dim: 128
  context_dim: 128
  search_num: 9
  mixed_precision: false
  test_mode: false
  tracing: false
  include_preprocessing: false
  weights: null
  strict_load: true
data:
  batch_size: 4
  num_workers: 8
  dataset_dir: /data/tartanair
  HW:
  - 480
  - 640
  train_envs:
  - abandonedfactory
  - amusement
  - carwelding
  - endofworld
  - gascola
  - hospital
  - japanesealley
  - neighborhood
  - ocean
  - office
  - office2
  - oldtown
  - seasidetown
  - seasonsforest
  - seasonsforest_winter
  - soulcity
  - westerndesert
  val_envs:
  - abandonedfactory_night
trainer:
  workdir: /home/cv/nndepth/experiments
  project_name: crestereo
  experiment_name: base_cre_stereo
  num_epochs: 100
  max_steps: 250000
  gradient_accumulation_steps: 1
  val_interval: 1.0
  log_interval: 100
  num_val_samples: null
  save_best_k_cp: 3
  tracker: wandb
  checkpoint: null
  resume: false
  lr: 0.0001
  weight_decay: 0.0001
  epsilon: 1.0e-08
  dtype: bfloat16
  device: cuda
```

## How to Launch Training
```bash
python nndepth/nndepth/disparity/train.py --model_config nndepth/disparity/configs/models/CREStereoBase.yaml --data_config nndepth/disparity/configs/data/BaseRAFTStereo_Tartanair.yaml --training_config nndepth/disparity/configs/training/RAFTTrainer.yaml
```

## How to Launch Inference
- Download checkpoint trained on TartanAir [Update soon]()
```bash
python nndepth/models/cre_stereo/scripts/inference.py --weights PATH --left_path samples/stereo/left/ --right_path samples/stereo/
right/ --HW 480 640 --output test --save_format image
```

## How to Launch Evaluation
- Download the config and checkpoint from [here](https://drive.google.com/drive/folders/1OZIqRjqlF2fD4wwbMsFf5Lxx7ovYdu1D).
- Change the path to your kitti [stereo 2015 dataset](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) in the [configuration](../configs/data/Kitti2015DisparityDataLoader.yaml) (`dataset_dir`). If you do not have Kitti on your PC, you can use the path to the [kitti-stereo-2015](../../../samples/kitti-stereo-2015/) in our samples.
- Launch the command:
```bash
python nndepth/models/cre_stereo/scripts/evaluate.py --data_name kitti --data_config nndepth/models/cre_stereo/yaml/base/kitti_eval.yaml  --weights PATH --metric_name kitti-d1 --metric_threshold 3 --output results.txt
```

## Performance
| Dataset | Metric | Value |
| :------ | :----: | :---: |
| **Kitti** | d1 | WIP |
| **Kitti** | EPE | WIP |
