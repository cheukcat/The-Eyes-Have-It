import torch
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from occnet import VIEW_TRANSFORMERS
from .inverse_matrixVT import InverseMatrixVT


@VIEW_TRANSFORMERS.register_module()
class AllYouNeedIsInYourEyes(BaseModule):
    def __init__(self,
                 feature_strides,
                 in_index=-1,
                 grid_size=[100, 100, 8],
                 x_bound=[-51.2, 51.2],
                 y_bound=[-51.2, 51.2],
                 z_bound=[-5., 3.],
                 sampling_rate=4,
                 num_cams=3):
        super().__init__()
        self.grid_size = grid_size
        self.num_cams = num_cams
        self.sub_grid_size = [grid_size[0] // 2,
                              grid_size[1] // 2,
                              grid_size[2]]
        # [front_left, front_right, rear_left, rear_right]
        self.x_bound = [[x_bound[0], 0], [0, x_bound[1]],
                        [x_bound[0], 0], [0, x_bound[1]]]
        self.y_bound = [[0, y_bound[1]], [0, y_bound[1]],
                        [y_bound[0], 0], [y_bound[0], 0]]
        self.z_bound = z_bound

        self.imvts = nn.ModuleList()
        for i in range(4):
            imvt = InverseMatrixVT(feature_strides,
                                   in_index,
                                   grid_size=self.sub_grid_size,
                                   x_bound=self.x_bound[i],
                                   y_bound=self.y_bound[i],
                                   z_bound=self.z_bound,
                                   sampling_rate=sampling_rate,
                                   num_cams=3)
            self.imvts.append(imvt)

    def forward(self, img_feats, img_metas):
        # [front_left, front_right, rear_left, rear_right]
        occ_feats_yz = []
        occ_feats_xz = []
        occ_feats_xy = []
        for i in range(4):
            occ_feat_yz, occ_feat_xz, occ_feat_xy = self.imvts[i](img_feats, img_metas)
            occ_feats_yz.append(occ_feat_yz)  # B, C, Y, Z
            occ_feats_xz.append(occ_feat_xz)  # B, C, X, Z
            occ_feats_xy.append(occ_feat_xy)  # B, C, X, Y
        front_xy = torch.cat([occ_feats_xy[0], occ_feats_xy[1]], dim=2)
        rear_xy = torch.cat([occ_feats_xy[2], occ_feats_xy[3]], dim=2)
        global_xy = torch.cat([rear_xy, front_xy], dim=3)
        # yz plane
        left_yz = torch.cat([occ_feats_yz[2], occ_feats_yz[0]], dim=2)
        right_yz = torch.cat([occ_feats_yz[3], occ_feats_yz[1]], dim=2)
        global_yz = torch.cat([left_yz, right_yz], dim=0)
        # xz plane
        front_xz = torch.cat([occ_feats_xz[0], occ_feats_xz[1]], dim=2)
        rear_xz = torch.cat([occ_feats_xz[2], occ_feats_xz[3]], dim=2)
        global_xz = torch.cat([rear_xz, front_xz], dim=0)
        return global_yz, global_xz, global_xy
