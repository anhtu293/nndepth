# Stereo Depth Estimation

This module is used to implement - train - inference different neural networks for Depth estimation. The models are implemented in `pytorch`.

<p align="center">
  <img src="../../images/tartanair_disp.png"/>
</p>

## Model Zoo

| Model              | Kitti d1 | Kitti EPE | Download | Model Details |
| :---------------- | :------: | :----: | :---------: | :-----------: |
| **BaseRaftStereo** | 0.08448 | 1.47151| [Link](https://drive.google.com/drive/folders/1OZIqRjqlF2fD4wwbMsFf5Lxx7ovYdu1D)| [README](./raft_stereo/README.md) |
| **Coarse2FineGroupRepViTRAFTStereo** | WIP | WIP | WIP | [README](./raft_stereo/README.md) |
| **CREStereoBase** | WIP | WIP | WIP | [README](./cre_stereo/README.md) |
| **IGEVStereoBase** | WIP | WIP | WIP | [README](./igev_stereo/README.md) |
| **IGEVStereoMBNet** | WIP | WIP | WIP | [README](./igev_stereo/README.md) |

## General Structure of each model
```
.
├── configs       --> Configuration files
├── criterions    --> Loss functions & metrics
├── data_loaders  --> Data loaders for training
├── __init__.py
├── README.md
├── trainers      --> Trainers
└── train.py      --> Training scripts
```

## Getting Started
For detailed information about training, inference, and evaluation for each model, please refer to the individual model READMEs:

- **RAFT-Stereo**: [models/raft_stereo/README.md](./raft_stereo/README.md)
- **CRE-Stereo**: [models/cre_stereo/README.md](./cre_stereo/README.md)
- **IGEV-Stereo**: [models/igev_stereo/README.md](./igev_stereo/README.md)
