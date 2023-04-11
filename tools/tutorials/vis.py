import sys
import os
from pathlib import Path
# TODO: we haven't build the project yet! So add the project to the env path
project_dir = str(Path(__file__).parents[2])
sys.path.append(project_dir)
os.chdir(project_dir)

import numpy as np
import shutil
import pickle
from tqdm import tqdm
from mmcv import Config
from argparse import ArgumentParser
from tools.visualization.vis_dataset import build_vis_dataset
from tools.visualization.vis_utils import draw

def Visualiation(args):
    """ Visualize occupancy scenes with or without prediction results
    """
    config = Config.fromfile(args.config)
    dataset_config = config['dataset_params']
    val_dataset_config = config['val_data_loader']
    dataset_config.version = args.version

    grid_size = config.grid_size
    vis_dataset = build_vis_dataset(dataset_config,
                                    val_dataset_config,
                                    grid_size)    
    # get results
    if args.results is not None:
        results = pickle.load(open(args.results, 'rb'))
    else: results = None

    # create process bar
    pbar = tqdm(total=len(vis_dataset))

    for index in range(len(vis_dataset)):
        # get the sample
        batch_data, filelist, scene_meta, timestamp = vis_dataset[index]
        imgs, img_metas, vox_label, grid, pt_label = batch_data

        imgs = np.stack([imgs])     # imgs not used for now
        grid = np.stack([grid])

        voxel_origin = dataset_config['min_volume_space']
        voxel_max = dataset_config['max_volume_space']
        resolution = [(e - s) / l for e, s, l in
                      zip(voxel_max, voxel_origin, grid_size)]

        # create save dir
        save_path = Path('vis_output')
        frame_dir = save_path / str(index)
        frame_dir.mkdir(parents=True, exist_ok=True)

        for filename in filelist:   # imgs not used for now            
            filename_ = Path(filename).name
            shutil.copy(filename, frame_dir / filename_)

        if results is not None:
            predict_vox = results[index]
            predict_pts = None  # pts not used for now
        else: # create prediction placeholders
            predict_vox = np.ones(grid_size)
            predict_pts = None

        voxel_nums = draw(predict_vox, 
                          predict_pts,
                          voxel_origin, 
                          resolution, 
                          grid.squeeze(0), 
                          pt_label.squeeze(-1),
                          frame_dir,
                          img_metas['cam_positions'],
                          img_metas['focal_positions'],
                          timestamp=timestamp,
                          offscreen=True,    # save fig
                          mode=2)

        pbar.set_postfix({'voxel numbers': voxel_nums})
        pbar.update()

if __name__ == '__main__':
    parser = ArgumentParser(description='visualize the occupancy grid')
    parser.add_argument('--config', default='configs/vanilla_occ_r50_fpn_1x_nus-trainval.py')
    parser.add_argument('--version', default='v1.0-trainval', help='v1.0-trainval or v1.0-mini')
    parser.add_argument('--results', default=None, help='prediction result file')
    parser.add_argument('--mode', default=2, type=int, help='0: pred occ, 1: pred pts, 2: gt occ')

    args = parser.parse_args()
    Visualiation(args)