import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from mmseg.models.builder import HEADS
from occnet.utils import Upsample
from .vanilla_head import VanillaHead


@HEADS.register_module()
class MultiViewFusionHead(VanillaHead):
    def __init__(self, in_channels, channels, num_classes, **kwargs):
        super().__init__(in_channels, channels, num_classes, **kwargs)

    def forward(self, occ_feats, **kwargs):
        # occ_feats [2B, C, 0.5Y, Z], [2B, C, 0.5X, Z], [B, C, X, Y],
        yz_feat, xz_feat, xy_feat = occ_feats
        # extract features
        xy_feat = self.xy_conv(xy_feat)
        xz_feat = self.xz_conv(xz_feat)
        yz_feat = self.yz_conv(yz_feat)
        B, C, X, Y, Z = *xy_feat.size(), xz_feat.size(3)
        left_yz_feat = yz_feat[:B]
        right_yz_feat = yz_feat[B:]
        front_xz_feat = xz_feat[:B]
        rear_xz_feat = xz_feat[B:]
        #TODO