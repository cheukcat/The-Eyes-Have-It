import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner import BaseModule, force_fp32, auto_fp16
from occnet import VIEW_TRANSFORMERS
from occnet.utils import multi_apply


@VIEW_TRANSFORMERS.register_module()
class InverseMatrixVT(BaseModule):
    def __init__(self,
                 grid_size=[100, 100, 8],
                 x_bound=[-51.2, 51.2],
                 y_bound=[-51.2, 51.2],
                 z_bound=[-5., 3.],
                 sampling_rate=4,
                 ds_rate=32.):
        super().__init__()
        self.grid_size = torch.tensor(grid_size)
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.sampling_rate = sampling_rate
        self.ds_rate = ds_rate
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
                                  self.grid_size[2], self.sampling_rate, 4). \
            permute(0, 2, 4, 1, 3, 5, 6).reshape(self.grid_size[0], self.grid_size[1],
                                                 self.grid_size[2], -1, 4)
        return new_coord

    def get_vt_matrix(self, img_feats, img_metas):
        batch_vt = multi_apply(self._get_vt_matrix_single,
                               img_feats,
                               img_metas)
        return torch.stack(batch_vt[0])

    @force_fp32(apply_to=('img_feat', 'img_meta'))
    def _get_vt_matrix_single(self, img_feat, img_meta):
        Nc, C, H, W = img_feat.shape
        # lidar2img: (Nc, 4, 4)
        lidar2img = img_meta['lidar2img']
        lidar2img = np.asarray(lidar2img)
        lidar2img = img_feat.new_tensor(lidar2img)
        img_shape = img_meta['img_shape']
        # global_coord: (X * Y * Z, Nc, S, 4, 1)
        global_coord = self.coord.clone()
        X, Y, Z, S, _ = global_coord.shape
        global_coord = global_coord.view(X * Y * Z, 1, S, 4, 1) \
            .repeat(1, Nc, 1, 1, 1)
        # lidar2img: (X * Y * Z, Nc, S, 4, 4)
        lidar2img = lidar2img.view(1, Nc, 1, 4, 4) \
            .repeat(X * Y * Z, 1, S, 1, 1)
        # ref_points: (X * Y * Z, Nc, S, 4), 4: (λW, λH, λ, 1)
        ref_points = torch.matmul(lidar2img.to(torch.float32),
                                  global_coord.to(torch.float32)).squeeze(-1)
        ref_points[..., 0] = ref_points[..., 0] / ref_points[..., 2]
        ref_points[..., 1] = ref_points[..., 1] / ref_points[..., 2]
        # remove invalid sampling points
        invalid_w = torch.logical_or(ref_points[..., 0] < 0.,
                                     ref_points[..., 0] > (img_shape[0][1] - 1))
        invalid_h = torch.logical_or(ref_points[..., 1] < 0.,
                                     ref_points[..., 1] > (img_shape[0][0] - 1))
        invalid_d = ref_points[..., 2] < 0.

        ref_points = torch.div(ref_points[..., :2],
                               self.ds_rate,
                               rounding_mode='floor').to(torch.long)
        cam_index = torch.arange(Nc).unsqueeze(0).unsqueeze(2) \
            .repeat(X * Y * Z, 1, S).unsqueeze(-1)
        # ref_points: (X * Y * Z, Nc * S, 3), 3: (W, H, Nc)
        ref_points = torch.cat([ref_points, cam_index], dim=-1)
        ref_points[(invalid_w | invalid_h | invalid_d)] = -1
        ref_points = ref_points.view(X * Y * Z, -1, 3)
        # ref_points_flatten: (X * Y * Z, Nc * S), 1: H * W * nc + W * h + w
        ref_points_flatten = ref_points[..., 2] * H * W + \
                             ref_points[..., 1] * W + ref_points[..., 0]
        # create vt matrix
        vt = img_feat.new_zeros(Nc * H * W, X * Y * Z)
        valid_idx = torch.nonzero(ref_points_flatten > 0)
        vt[ref_points_flatten[valid_idx[:, 0], valid_idx[:, 1]], valid_idx[:, 0]] = 1
        vt /= vt.sum(0).clip(min=1)
        return vt.unsqueeze(0)
