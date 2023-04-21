import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from mmseg.models.builder import HEADS
from occnet.utils import Upsample
from .vanilla_head import VanillaHead


@HEADS.register_module()
class MultiViewFusionHead(VanillaHead):
    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 depth_size=[50, 50, 8],  # attention, depth is only half of grid size on bev!
                 **kwargs):
        super().__init__(in_channels, channels, num_classes, **kwargs)
        # xy, xz, yz depth extractor
        self.xy_depthnet = self.build_depthnet(depth_size[2])
        self.xz_depthnet = self.build_depthnet(depth_size[1])
        self.yz_depthnet = self.build_depthnet(depth_size[0])

    def build_depthnet(self, depth):
        depthnet = nn.Conv2d(self.channels,
                             self.channels + depth,
                             kernel_size=1, padding=0)
        return depthnet

    def decode_3d(self, x):
        depth = x[:, self.channels:].softmax(dim=1)
        x = x[:, :self.channels]
        occ_feats = depth.unsqueeze(1) * x.unsqueeze(2)
        return occ_feats

    def forward(self, occ_feats, **kwargs):
        # occ_feats [2B, C, Y, Z], [2B, C, X, Z], [B, C, X, Y],
        yz_feat, xz_feat, xy_feat = occ_feats
        # extract & decode features to 3D,
        # [B, C, X, Y, Z], [2B, C, X, Y // 2, Z], [2B, C, X // 2, Y, Z]
        xy_feat = self.decode_3d(self.xy_depthnet(self.xy_conv(xy_feat))).permute(0, 1, 3, 4, 2)
        xz_feat = self.decode_3d(self.xz_depthnet(self.xz_conv(xz_feat))).permute(0, 1, 3, 2, 4)
        yz_feat = self.decode_3d(self.xz_depthnet(self.yz_conv(yz_feat)))
        B, C, X, Y, Z = xy_feat.shape
        # rear, front
        xz_feat = torch.cat([xz_feat[:B], xz_feat[B:]], dim=3)
        # left, right
        yz_feat = torch.cat([yz_feat[:B], yz_feat[B:]], dim=2)
        xyz_feat = xy_feat + xz_feat + yz_feat
        # reshape and fc
        xyz_feat = self.fc(xyz_feat.view(B, C, -1).transpose(1, 2))
        logtis = self.classifier(xyz_feat)  # (B, XYZ, C)
        logtis = logtis.permute(0, 2, 1).reshape(B, -1, X, Y, Z)

        return logtis
