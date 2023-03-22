import sys
import os
from pathlib import Path
# TODO: we haven't build the project yet! So add the project to the env path
sys.path.append(str(Path(__file__).parents[2]))
print(str(Path(__file__).parents[2]))

import numpy as np
import shutil
from tools.visualization.vis_dataset import build_vis_dataset
from tools.visualization.vis_utils import draw

# set configs
# set the path to your own
version = 'v1.0-mini'

dataset_config = dict(
    version = version,
    ignore_label = 0,
    fill_label = 17,
    fixed_volume_space = True,
    label_mapping = "/github/The-Eyes-Have-It/configs/_base_/label_mapping/nuscenes-noIgnore.yaml",
    max_volume_space = [51.2, 51.2, 3],
    min_volume_space = [-51.2, -51.2, -5],
)
dataloader_config = dict(
    data_path = "/github/The-Eyes-Have-It/data/nuscenes",
    imageset = "/github/The-Eyes-Have-It/data/nuscenes_infos_train.pkl",
    batch_size = 1,
    shuffle = True,
    num_workers = 0,    # debug use
)

# build visualization dataset
vis_dataset = build_vis_dataset(dataset_config,
                                dataloader_config)

# visualization index
index = 0
# get the sample
batch_data, filelist, scene_meta, timestamp = vis_dataset[index]
imgs, img_metas, vox_label, grid, pt_label = batch_data

grid_size = [200, 200, 16]
imgs = np.stack([imgs]) # not used for now
grid = np.stack([grid])

voxel_origin = dataset_config['min_volume_space']
voxel_max = dataset_config['max_volume_space']
resolution = [(e - s) / l for e, s, l in zip(voxel_max, voxel_origin, grid_size)]

save_path = 'vis_output'

frame_dir = os.path.join(save_path, str(index))
os.makedirs(frame_dir, exist_ok=True)

for filename in filelist:   # not used for now            
    shutil.copy(filename, os.path.join(frame_dir, os.path.basename(filename)))

# create prediction placeholders
voxels_shape = [256, 256, 32]
predict_vox = np.ones(voxels_shape)
predict_pts = None

draw(predict_vox, 
     predict_pts,
     voxel_origin, 
     resolution, 
     grid.squeeze(0), 
     pt_label.squeeze(-1),
     frame_dir,
     img_metas['cam_positions'],
     img_metas['focal_positions'],
     timestamp=timestamp,
     offscreen=True,    # save the fig!
     mode=2)