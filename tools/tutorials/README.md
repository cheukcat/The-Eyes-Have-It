# Tutorial

## Visualization

### Install
Recommend install order
1. Install [pytorch 1.10](https://pytorch.org/get-started/previous-versions/) at corresponding cuda
    ```shell
    pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
    ```
2. Install mmcv==1.4.0
    ```shell
    pip install openmim
    mim install mmcv-full==1.4.0
    mim install mmdet
    mim install mmsegmentation
    ```

3. other packages
   ```shell
    pip install timm
    pip install nuscenes-devkit
    pip install numba
    pip install tensorboard
    pip install setuptools==59.5.0  # to solve a tensorboard bug https://github.com/pytorch/pytorch/issues/69894#issuecomment-1080635462
    pip install easydict
    ```
4. If you want to use mayavi to visualize, you can follow the instruction below. Mayavi is a little tricky to use...

    ```shell
    pip install mayavi  # if meet error, try pip install again might solve it
    # OpenGL Glig Xrender
    apt install libgl1-mesa-glx libglib2.0-dev libxrender1
    ```
5. We use vedo to visualize in our project, which is faster and maintained frequently.
   ```shell
   pip install vedo
   ```

### Usage

run the `vis.py` to use 
```shell
python tools/tutorials/vis.py --config {config_path} --ckpt {ckpt_path} --mode 0
```
There are 3 modes:
1. `--mode 0` is to visualize predicted occupancy voxels
2. `--mode 1` is to visualize predicted points (TODO)
3. `--mode 2` is to visualize ground truth voxels