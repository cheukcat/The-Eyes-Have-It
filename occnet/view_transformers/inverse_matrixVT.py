import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner import BaseModule, force_fp32, auto_fp16
from occnet import VIEW_TRANSFORMERS


@VIEW_TRANSFORMERS.register_module()
class InverseMatrixVT(BaseModule):
    def __init__(self,
                 grid_size=[100, 100, 8],
                 x_bound=[-51.2, 51.2],
                 y_bound=[-51.2, 51.2],
                 z_bound=[-5., 3.],
                 sampling_rate=4):
        super().__init__()
        self.grid_size = torch.tensor(grid_size)
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.sampling_rate = sampling_rate
        self.coord = self._create_gridmap_anchor()

    def _create_gridmap_anchor(self):
        # create a gridmap anchor with shape of (X, Y, Z, sampling_rate**3, 3)
        grid_size = self.sampling_rate * self.grid_size
        coord = torch.zeros(grid_size[0], grid_size[1], grid_size[2], 3)
        x_coord = torch.linspace(self.x_bound[0], self.x_bound[1], grid_size[0])
        y_coord = torch.linspace(self.y_bound[0], self.y_bound[1], grid_size[1])
        z_coord = torch.linspace(self.z_bound[0], self.z_bound[1], grid_size[2])
        ones = torch.ones(grid_size[0], grid_size[1], grid_size[2], 1)
        coord[:, :, :, 0] = x_coord.reshape(-1, 1, 1)
        coord[:, :, :, 1] = y_coord.reshape(1, -1, 1)
        coord[:, :, :, 2] = z_coord.reshape(1, 1, -1)
        coord = torch.cat([coord, ones], dim=-1)
        # taking multi sampling points into a single grid
        new_coord = coord.reshape(self.grid_size[0], self.sampling_rate,
                                  self.grid_size[1], self.sampling_rate,
                                  self.grid_size[2], self.sampling_rate, 4).\
            permute(0, 2, 4, 1, 3, 5, 6).reshape(self.grid_size[0], self.grid_size[1],
                                                 self.grid_size[2], -1, 4)
        return new_coord

    @force_fp32(apply_to=('img_metas'))
    def get_ref_points(self, img_feats, img_metas):
        B, N, C, H, W = img_feats.shape
        # lidar2img: (B, N, 4, 4)
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = img_feats.new_tensor(lidar2img)
        # global_coord: (B, N, X * Y * Z, S, 4, 1)
        global_coord = self.coord.clone()
        X, Y, Z, S, _ = global_coord.shape
        global_coord = global_coord.unsqueeze(0).unsqueeze(1)\
            .view(1, 1, X * Y * Z, S, 4, 1)\
            .repeat(B, N, 1, 1, 1, 1)
        # lidar2img: (B, N, X * Y * Z, S, 4, 4)
        lidar2img = lidar2img.unsqueeze(2).unsqueeze(3)\
            .repeat(1, 1, X * Y * Z, S, 1, 1)
        # ref_points: (B, N, X * Y * Z, S, 4)
        ref_points = torch.matmul(lidar2img.to(torch.float32),
                                  global_coord.to(torch.float32)).squeeze(-1)
        return ref_points


