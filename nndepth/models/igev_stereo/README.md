# IGEV-Stereo

## Architecture
- Detail at [IGEV-Stereo](https://arxiv.org/pdf/2303.06615.pdf)
<p align="center">
<img src="../../../images/igev.png"/>
</p>

- `MobilenetLarge-V3` is used as backbone.

## Training Configuration
```yaml
model:
  update_cls: basic_update_block
  cv_groups: 8
  iters: 6
  hidden_dim: 64
  context_dim: 64
  corr_levels: 4
  corr_radius: 4
  tracing: false
  include_preprocessing: false
  weights: null
  strict_load: true
data:
  batch_size: 2
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
  project_name: igev_stereo
  experiment_name: mbnet_igev_stereo
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

```

## How to Launch Training
```bash
python nndepth/mdoels/igev_stereo/script/train.py --config_file nndepth/mdoels/igev_stereo/yaml/mbnet/training.yaml
```

## How to Launch Inference
- Download checkpoint trained on TartanAir [Update soon]()
```bash
python nndepth/models/igev_stereo/scripts/inference.py --weights PATH --left_path samples/stereo/left/ --right_path samples/stereo/
right/ --HW 480 640 --output test --save_format image
```

## How to Launch Evaluation
- Download the config and checkpoint from [here](https://drive.google.com/drive/folders/1OZIqRjqlF2fD4wwbMsFf5Lxx7ovYdu1D).
- Change the path to your kitti [stereo 2015 dataset](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) in the [configuration](../configs/data/Kitti2015DisparityDataLoader.yaml) (`dataset_dir`). If you do not have Kitti on your PC, you can use the path to the [kitti-stereo-2015](../../../samples/kitti-stereo-2015/) in our samples.
- Launch the command:
```bash
python nndepth/models/igev_stereo/scripts/evaluate.py --data_name kitti --data_config nndepth/models/igev_stereo/yaml/mbnet/kitti_eval.yaml  --weights PATH --metric_name kitti-d1 --metric_threshold 3 --output results.txt
```

## Performance
| Dataset | Metric | Value |
| :------ | :----: | :---: |
| **Kitti** | d1 | WIP |
| **Kitti** | EPE | WIP |
