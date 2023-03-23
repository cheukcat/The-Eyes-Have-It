from mmcv import Config
from occnet import build_dataloader

cfg = Config.fromfile('config/vanilla_occ_r50_fpn_1x_nus-mini.py')
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
