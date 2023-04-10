import sys
import os
from pathlib import Path
# TODO: we haven't build the project yet! So add the project to the env path
sys.path.append(str(Path(__file__).parents[2]))
print(str(Path(__file__).parents[2]))

import numpy as np
import shutil
import pickle
from tools.visualization.vis_utils import draw, draw_occ

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
# vis_dataset = build_vis_dataset(dataset_config,
#                                 dataloader_config)
# build val dataset
# val_dataset = build_val_dataset(dataset_config,
                                # dataloader_config)
# prediction
result_file = '/github/The-Eyes-Have-It/results.pkl'
pred_list = pickle.load(open(result_file, 'rb'))

grid_size = [100, 100, 8]
# imgs = np.stack([imgs]) # not used for now
# grid = np.stack([grid])

voxel_origin = dataset_config['min_volume_space']
voxel_max = dataset_config['max_volume_space']
voxel_size = [(e - s) / l for e, s, l in zip(voxel_max, voxel_origin, grid_size)]
save_path = 'vis_output'

for index in range(len(pred_list)):

    # get the sample
    # batch_data, filelist, scene_meta, timestamp = vis_dataset[index]
    # imgs, img_metas, vox_label, grid, pt_label = batch_data
    save_path = Path('vis_output')
    os.makedirs(save_path, exist_ok=True)
    save_name = save_path / f'fig_{index:04d}.png'

    # for filename in filelist:   # not used for now            
        # shutil.copy(filename, os.path.join(frame_dir, os.path.basename(filename)))

    predict_vox = pred_list[index][0]
    predict_pts = None  # not used for now

    # draw(predict_vox, 
    #      predict_pts,
    #      voxel_origin, 
    #      resolution, 
    #      grid.squeeze(0), 
    #      pt_label.squeeze(-1),
    #      frame_dir,
    #      img_metas['cam_positions'],
    #      img_metas['focal_positions'],
    #      timestamp=timestamp,
    #      offscreen=True,    # save the fig!
    #      mode=0)

    draw_occ(predict_vox,
             voxel_origin,
             voxel_size,
             save_name,
             offscreen=True)
        