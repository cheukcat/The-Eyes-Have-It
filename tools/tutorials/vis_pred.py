import sys
import os
from pathlib import Path
# TODO: we haven't build the project yet! So add the project to the env path
sys.path.append(str(Path(__file__).parents[2]))
print(str(Path(__file__).parents[2]))

import numpy as np
import shutil
import pickle
from tools.visualization.vis_dataset import build_vis_dataset
from tools.visualization.vis_utils import draw

# set configs
# set the path to your own
version = 'v1.0-trainval'

dataset_config = dict(
    version = version,
    ignore_label = 0,
    fill_label = 17,
    fixed_volume_space = True,
    label_mapping = "/works/cbh/The-Eyes-Have-It/configs/_base_/label_mapping/nuscenes-noIgnore.yaml",
    max_volume_space = [51.2, 51.2, 3],
    min_volume_space = [-51.2, -51.2, -5],
)
dataloader_config = dict(
    data_path = "/works/cbh/The-Eyes-Have-It/data/nuscenes",
    imageset = "/works/cbh/The-Eyes-Have-It/data/nuscenes_infos_val.pkl",
    batch_size = 1,
    shuffle = True,
    num_workers = 0,    # debug use
)
# grid_size = [200, 200, 16]
grid_size = [100, 100, 8]   # output is downsampled
# build visualization dataset
vis_dataset = build_vis_dataset(dataset_config,
                                dataloader_config,
                                grid_size)
# prediction
results = '/works/cbh/The-Eyes-Have-It/work_dirs/occ_val/results.pkl'
pred_list = pickle.load(open(results, 'rb'))

# visualization index
index = 0
# get the sample
batch_data, filelist, scene_meta, timestamp = vis_dataset[index]
imgs, img_metas, vox_label, grid, pt_label = batch_data

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

predict_vox = pred_list[index][0]
predict_pts = None  # not used for now

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
     mode=0)