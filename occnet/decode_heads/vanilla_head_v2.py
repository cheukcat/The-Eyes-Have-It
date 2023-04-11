import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from mmseg.models.builder import HEADS
from occnet.utils import Upsample


import warnings
import torch.nn.functional as F


@HEADS.register_module()
class VanillaHeadv2(BaseModule):
    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 scale=1,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # xy, xz, yz conv layers
        self.xy_conv = self.build_3x3_conv_block(scale)
        self.xz_conv = self.build_3x3_conv_block(scale)
        self.yz_conv = self.build_3x3_conv_block(scale)

        # xy, xz, yz weight layers
        self.xy_weight = ConvModule(in_channels, 1, 1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=None)
        self.xz_weight = ConvModule(in_channels, 1, 1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=None)
        self.yz_weight = ConvModule(in_channels, 1, 1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=None)
        
        # xyz fc layers
        self.fc = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.Softplus(),
            nn.Linear(self.channels, self.channels)
        )
        self.classifier = nn.Linear(self.channels, num_classes)

    def build_3x3_conv_block(self, layers):
        conv_list = []
        for i in range(layers):
            if i != 0:
                conv_list.append(
                    Upsample(
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=False))
            conv_list.append(
                ConvModule(
                    self.in_channels if i == 0 else self.channels,
                    self.channels,
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        return nn.Sequential(*conv_list)

    def attention_fusion(self, t1, w1, t2, w2):
        norm_weight = torch.softmax(torch.cat([w1, w2], dim=1), dim=1)
        feat = t1 * norm_weight[:, 0:1] + t2 * norm_weight[:, 1:2]
        return feat

    def expand_to_XYZ(self, xy_feat, xz_feat, yz_feat):
        B, C, X, Y, Z = *xy_feat.size(), xz_feat.size(3)
        xy_feat = xy_feat.view(B, C, X, Y, 1)
        xz_feat = xz_feat.view(B, C, X, 1, Z)
        yz_feat = yz_feat.view(B, C, 1, Y, Z)
        return torch.broadcast_tensors(xy_feat, xz_feat, yz_feat)

    def forward(self, occ_feats, **kwargs):
        # occ_feats (NCXY, NCXZ, NCYZ)
        yz_feat, xz_feat, xy_feat = occ_feats
        B, C, X, Y, Z = *xy_feat.size(), xz_feat.size(3)
        
        # extract features
        xy_feat = self.xy_conv(xy_feat)
        xz_feat = self.xz_conv(xz_feat)
        yz_feat = self.yz_conv(yz_feat)

        # perspective attention weight
        xy_weight = self.xy_weight(xy_feat)
        xz_weight = self.xz_weight(xz_feat)
        yz_weight = self.yz_weight(yz_feat)

        # expand feat and weight to (B, C, X, Y, Z)
        xy_feat, xz_feat, yz_feat = self.expand_to_XYZ(xy_feat, xz_feat, yz_feat)
        xy_weight, xz_weight, yz_weight = self.expand_to_XYZ(xy_weight, xz_weight, yz_weight)

        # perspective attention fusion
        xyz_feat = self.attention_fusion(xy_feat, xy_weight, xz_feat, xz_weight) + \
                   self.attention_fusion(xy_feat, xy_weight, yz_feat, yz_weight)

        # reshape and fc
        xyz_feat = self.fc(xyz_feat.view(B, C, -1).transpose(1, 2))
        logtis = self.classifier(xyz_feat)  # (B, XYZ, C)
        logtis = logtis.permute(0, 2, 1).reshape(B, -1, X, Y, Z)

        return logtis