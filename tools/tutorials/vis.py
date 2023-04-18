import sys
import os
from pathlib import Path
# TODO: we haven't build the project yet! So add the project to the env path
project_dir = str(Path(__file__).parents[2])
sys.path.append(project_dir)
os.chdir(project_dir)

import numpy as np
import torch
import shutil
from tqdm import tqdm
from mmcv import Config
from argparse import ArgumentParser
from collections import OrderedDict
from tools.visualization.vis_dataset import build_vis_dataset
from occnet.builder import build_model
from tools.visualization.vis_utils import draw_with_vedo

def parse_args():

    parser = ArgumentParser(description='visualize the occupancy grid')
    parser.add_argument('--config', default='configs/vanilla_occ_r50_fpn_1x_nus-trainval.py')
    parser.add_argument('--version', default='v1.0-trainval', help='v1.0-trainval or v1.0-mini')
    parser.add_argument('--ckpt', default=None, help='checkpoint path')
    parser.add_argument('--device', default='cuda:0', help='device to use for inference')
    parser.add_argument('--mode', default=2, type=int, help='0: pred occ, 1: pred pts, 2: gt occ')

    return parser.parse_args()

def reverse_module_key(state_dict):
    """ What we save is ddp model, which's state_dict all start with `module.`,
    this function can remove the `module` prefix`
    """
    import re
    pattern = re.compile('^module.')    # start with module
    tmp_k = list(state_dict.keys())[0]
    if tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {pattern.sub('', k): v
             for k, v in state_dict.items()})
    return state_dict

def Visualiation(args):
    """ Visualize occupancy scenes with or without prediction results
    """
    config = Config.fromfile(args.config)
    dataset_config = config['dataset_params']
    val_dataset_config = config['val_data_loader']
    dataset_config.version = args.version

    # build dataset
    grid_size = config.grid_size
    vis_dataset = build_vis_dataset(dataset_config,
                                    val_dataset_config,
                                    grid_size)    
    # build model
    device = args.device
    my_model = build_model(config.model).to(args.device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        print(my_model.load_state_dict(reverse_module_key(ckpt)))
    else: 
        print('No checkpoint provided! Use random weights.')
    my_model.eval()

    # create process bar
    pbar = tqdm(total=len(vis_dataset))

    for index in range(len(vis_dataset)):
        # get the sample
        batch_data, filelist, scene_meta, timestamp = vis_dataset[index]
        imgs, img_metas, vox_label, grid, pt_label = batch_data

        imgs = torch.from_numpy(np.stack([imgs]).astype(np.float32)).to(device)
        grid = torch.from_numpy(np.stack([grid]).astype(np.float32)).to(device)

        with torch.no_grad():
            outputs_vox = my_model(img=imgs, 
                                img_metas=[img_metas],
                                points=grid.clone())
        
            predict_vox = torch.argmax(outputs_vox, dim=1) # bs, w, h, z
            predict_vox = predict_vox.squeeze(0).cpu().numpy() # w, h, z

            predict_pts = None   # haven't support seg task yet
            # predict_pts = torch.argmax(outputs_pts, dim=1) # bs, n, 1, 1
            # predict_pts = predict_pts.squeeze().cpu().numpy() # n

        voxel_origin = dataset_config['min_volume_space']
        voxel_max = dataset_config['max_volume_space']
        resolution = [(e - s) / l for e, s, l in
                      zip(voxel_max, voxel_origin, grid_size)]

        # create save dir
        save_path = Path('vis_output')
        frame_dir = save_path / 'per_frame' / f'{index:04d}'
        frame_dir.mkdir(parents=True, exist_ok=True)
        all_frame_dir = save_path / str('all_frames')
        all_frame_dir.mkdir(parents=True, exist_ok=True)

        for filename in filelist:   # imgs not used for now            
            filename_ = Path(filename).name
            shutil.copy(filename, frame_dir / filename_)

        voxel_nums = draw_with_vedo(predict_vox, 
                          predict_pts,
                          voxel_origin, 
                          resolution, 
                          grid.squeeze(0).cpu().numpy(), 
                          pt_label.squeeze(-1),
                          frame_dir,
                          img_metas['cam_positions'],
                          img_metas['focal_positions'],
                          timestamp=timestamp,
                          offscreen=True,    # save fig
                          mode=args.mode)

        pbar.set_postfix({'voxel numbers': voxel_nums})
        pbar.update()

if __name__ == '__main__':

    args = parse_args()
    
    Visualiation(args)