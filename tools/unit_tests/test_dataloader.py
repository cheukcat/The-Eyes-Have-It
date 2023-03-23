import sys
from pathlib import Path
from mmcv import Config

CUR_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(CUR_DIR))
from occnet import build_dataloader

cfg = Config.fromfile('configs/vanilla_occ_r50_fpn_1x_nus-mini.py')
dataset_config = cfg.dataset_params
print('dataset_config: ', dataset_config)
train_dataloader_config = cfg.train_data_loader
print('train_dataloader_config: ', train_dataloader_config)
val_dataloader_config = cfg.val_data_loader
print('val_dataloader_config: ', val_dataloader_config)
grid_size = cfg.grid_size
print('grid_size: ', grid_size)
version = dataset_config['version']

# build dataloader
train_dataloader, val_dataloader = \
    build_dataloader(
        dataset_config,
        train_dataloader_config,
        val_dataloader_config,
        grid_size=grid_size,
        version=version,
        dist=False,
        scale_rate=cfg.get('scale_rate', 1)
    )

print(f'len(dataloader)={len(train_dataloader)}')
# iter, next 的用法和 for循环本质是相同的，这里用iter next 代替为for 循环一次
iter_loader = iter(train_dataloader)
sample = next(iter_loader)
