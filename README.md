# The Eyes Have It
An intuitive approach for 3D Occupancy Detection
## Installation
**1.** Create conda environment with python version 3.8

**2.** Install pytorch, torchvision, mmcv-full, mmdet, mmsegmentation and mmdet3d, following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html

**3.** conda install timm

## Data Preparation
**1.** Download NuScenes Dataset and create soft link to {Project}/data/nuscenes

**2.** Download our generated train/val pickle files and put them in data/ nuscenes_infos_train.pkl https://cloud.tsinghua.edu.cn/f/ede3023e01874b26bead/?dl=1 nuscenes_infos_val.pkl https://cloud.tsinghua.edu.cn/f/61d839064a334630ac55/?dl=1

**3.** Organize data as follow
```
TPVFormer/data
    nuscenes                 -    downloaded from www.nuscenes.org
        lidarseg
        maps
        samples
        sweeps
        v1.0-trainval
    nuscenes_infos_train.pkl
    nuscenes_infos_val.pkl
```
## Getting Started
### Training
1. Pretrained weights is highly recommended, please download from https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth and put it in ckpts/

2. Training VanillaOcc on 3090 with 24G GPU memory.
```
bash launcher.sh configs/vanilla_occ_r50_fpn_1x_nus-trainval.py work_dirs/vanilla_occ
```
3. Training InYourEyes, which is a GPU memory optimized version of VanillaOcc.
```
bash launcher.sh configs/in_your_eyes_r101_24_nus-trainval.py work_dirs/in_your_eyes
```
## Snapshot
### Motivation
**1.** Generate BEV freespace via 2D Segmentation is commonly used in adas industry, but it has limitations in many real-world scenarios.

**2.** 3D Occupancy has been proven to be a superior alternative to the previous perception scheme.
### Goal
**1.** A 3D OCC approach that balances accuracy, inference speed, deployability and simplicity.

**2.** A baseline that could be trained on generalized GPUs.

**3.** CVPR 2023 OCC challenge!

**4.** Deployed on an automotive-grade platform with real-time fps.
### Method
**1.** Inverse MatrixVT with complexity：
$$(N_c * H * W) * (X * Y + Y * Z + Z * X) * C$$
**2.** Decode 3D space with Tri-Perspective View

**3.** Sparse supervision on lidar annotations
## Experiments
TBD
