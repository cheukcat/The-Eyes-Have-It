import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from occnet import VIEW_TRANSFORMERS
from occnet.utils import (multi_apply, resize, BilinearDeconvolution)


@VIEW_TRANSFORMERS.register_module()
class InverseMatrixVT(BaseModule):
    def __init__(self,
                 feature_strides,
                 in_index=-1,
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
        self.in_index = in_index
        if isinstance(feature_strides, list):
            self.ds_rate = feature_strides[in_index]
        else:
            self.ds_rate = feature_strides
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
        global_coord = self.coord.clone().to(lidar2img.device)
        X, Y, Z, S, _ = global_coord.shape
        global_coord = global_coord.view(X * Y * Z, 1, S, 4, 1) \
            .repeat(1, Nc, 1, 1, 1)
        # lidar2img: (X * Y * Z, Nc, S, 4, 4)
        lidar2img = lidar2img.unsqueeze(0).unsqueeze(2) \
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
        cam_index = torch.arange(Nc, device=lidar2img.device) \
            .unsqueeze(0).unsqueeze(2) \
            .repeat(X * Y * Z, 1, S).unsqueeze(-1)
        # ref_points: (X * Y * Z, Nc * S, 3), 3: (W, H, Nc)
        ref_points = torch.cat([ref_points, cam_index], dim=-1)
        ref_points[(invalid_w | invalid_h | invalid_d)] = -1
        ref_points = ref_points.view(X * Y * Z, -1, 3)
        # ref_points_flatten: (X * Y * Z, Nc * S), 1: H * W * nc + W * h + w
        ref_points_flatten = ref_points[..., 2] * H * W + \
                             ref_points[..., 1] * W + ref_points[..., 0]
        # factorize 3D
        # TODO: not check yet!
        ref_points_flatten = ref_points_flatten.reshape(X, Y, Z, -1)
        ref_points_x = ref_points_flatten.permute(1, 2, 3, 0).view(Y * Z, -1)
        ref_points_y = ref_points_flatten.permute(0, 2, 3, 1).view(X * Z, -1)
        ref_points_z = ref_points_flatten.permute(0, 1, 3, 2).view(X * Y, -1)

        # create vt matrix
        vt_x = img_feat.new_zeros(Nc * H * W, Y * Z)
        vt_y = img_feat.new_zeros(Nc * H * W, X * Z)
        vt_z = img_feat.new_zeros(Nc * H * W, X * Y)
        valid_idx_x = torch.nonzero(ref_points_x > 0)
        valid_idx_y = torch.nonzero(ref_points_y > 0)
        valid_idx_z = torch.nonzero(ref_points_z > 0)
        vt_x[ref_points_x[valid_idx_x[:, 0],
                          valid_idx_x[:, 1]],
             valid_idx_x[:, 0]] = 1
        vt_x /= vt_x.sum(0).clip(min=1)
        vt_y[ref_points_y[valid_idx_y[:, 0],
                          valid_idx_y[:, 1]],
             valid_idx_y[:, 0]] = 1
        vt_y /= vt_y.sum(0).clip(min=1)
        vt_z[ref_points_z[valid_idx_z[:, 0],
                          valid_idx_z[:, 1]],
             valid_idx_z[:, 0]] = 1
        vt_z /= vt_z.sum(0).clip(min=1)

        return vt_x.unsqueeze(0), vt_y.unsqueeze(0), vt_z.unsqueeze(0)

    def forward(self, img_feats, img_metas):
        img_feats = img_feats[self.in_index]
        vt_x, vt_y, vt_z = self.get_vt_matrix(img_feats, img_metas)
        # flatten img_feats
        B, N, C, H, W = img_feats.shape
        img_feats = img_feats.permute(0, 2, 1, 3, 4).reshape(B, C, -1)
        # B, C, (Y * Z, X * Z, X * Y)
        occ_feats_x = torch.matmul(img_feats, vt_x)
        occ_feats_y = torch.matmul(img_feats, vt_y)
        occ_feats_z = torch.matmul(img_feats, vt_z)
        return occ_feats_x, occ_feats_y, occ_feats_z
