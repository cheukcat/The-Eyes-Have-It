import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from mmseg.models.builder import HEADS
from occnet.utils import Upsample


@HEADS.register_module()
class VanillaHead(BaseModule):
    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 scale=1,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.fp16_enabled = False
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # xy, xz, yz conv layers
        self.xy_conv = self.build_3x3_conv_block(scale)
        self.xz_conv = self.build_3x3_conv_block(scale)
        self.yz_conv = self.build_3x3_conv_block(scale)
        # xyz fc layers
        self.fc = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.Softplus(),
            nn.Linear(self.channels, self.channels)
        )
        self.classifier = nn.Linear(self.channels, num_classes)

    def build_3x3_conv_block(self, layers):
        conv_list = []
        conv_list.append(
            ConvModule(
                self.in_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(layers):
            if i != 0:
                conv_list.append(
                    Upsample(
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=False))
            conv_list.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        return nn.Sequential(*conv_list)

    # @force_fp32(apply_to=('cls_score', ))
    # def loss(self, cls_score, gt_label):
    #     loss = dict()
    #     loss['loss_cls'] = nn.functional.cross_entropy(
    #         cls_score, gt_label, reduction='mean')
    #     return loss

    def forward(self, occ_feats, **kwargs):
        # occ_feats (NCXY, NCXZ, NCYZ)
        yz_feat, xz_feat, xy_feat = occ_feats
        # extract features
        xy_feat = self.xy_conv(xy_feat)
        xz_feat = self.xz_conv(xz_feat)
        yz_feat = self.yz_conv(yz_feat)
        B, C, X, Y, Z = *xy_feat.size(), xz_feat.size(3)
        # multiply to expand, and add up
        xyz_feat = xy_feat.view(B, C, X, Y, 1) * torch.sigmoid(xz_feat).view(B, C, X, 1, Z) + \
                   xy_feat.view(B, C, X, Y, 1) * torch.sigmoid(yz_feat).view(B, C, 1, Y, Z) + \
                   xz_feat.view(B, C, X, 1, Z) * torch.sigmoid(xy_feat).view(B, C, X, Y, 1) + \
                   yz_feat.view(B, C, 1, Y, Z) * torch.sigmoid(xy_feat).view(B, C, X, Y, 1)
        # reshape and fc
        xyz_feat = self.fc(xyz_feat.view(B, C, -1).transpose(1, 2))
        logtis = self.classifier(xyz_feat)  # (B, XYZ, C)
        logtis = logtis.permute(0, 2, 1).reshape(B, -1, X, Y, Z)

        return logtis
