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
                 **kwargs):
        super().__init__(in_channels, channels, num_classes, **kwargs)
        # xy, xz, yz depth extractor
        self.xy_decoder = self.build_3d_decoder()
        self.xz_decoder = self.build_3d_decoder()
        self.yz_decoder = self.build_3d_decoder()

    def build_3d_decoder(self):
        decoder = nn.Conv2d(self.channels, self.channels,
                            kernel_size=1, padding=0)
        return decoder

    def forward(self, occ_feats, **kwargs):
        # occ_feats [2B, C, Y, Z], [2B, C, X, Z], [B, C, X, Y],
        yz_feat, xz_feat, xy_feat = occ_feats
        # extract features
        xy_feat = self.xy_conv(xy_feat)
        xz_feat = self.xz_conv(xz_feat)
        yz_feat = self.yz_conv(yz_feat)
        B, C, X, Y, Z = *xy_feat.size(), xz_feat.size(3)
        # 3d decoder
        xy_decoder = self.xy_decoder(xy_feat).sigmoid()
        xz_decoder = self.xz_decoder(xz_feat).sigmoid()
        yz_decoder = self.yz_decoder(yz_feat).sigmoid()
        # extract & decode features to 3D,
        rear_feats = xy_feat[:, :, :, :(Y // 2)].unsqueeze(-1) * xz_decoder[:B].unsqueeze(3) + \
                     xz_feat[:B].unsqueeze(3) * xy_decoder[:, :, :, :(Y // 2)].unsqueeze(-1)
        front_feats = xy_feat[:, :, :, (Y // 2):].unsqueeze(-1) * xz_decoder[B:].unsqueeze(3) + \
                      xz_feat[B:].unsqueeze(3) * xy_decoder[:, :, :, (Y // 2):].unsqueeze(-1)
        left_feats = xy_feat[:, :, :(X // 2), :].unsqueeze(-1) * yz_decoder[:B].unsqueeze(2) + \
                     yz_feat[:B].unsqueeze(2) * xy_decoder[:, :, :(X // 2), :].unsqueeze(-1)
        right_feats = xy_feat[:, :, (X // 2):, :].unsqueeze(-1) * yz_decoder[B:].unsqueeze(2) + \
                      yz_feat[B:].unsqueeze(2) * xy_decoder[:, :, (X // 2):, :].unsqueeze(-1)

        xyz_feat = torch.cat([rear_feats, front_feats], dim=3) + \
                   torch.cat([left_feats, right_feats], dim=2)

        # reshape and fc
        xyz_feat = self.fc(xyz_feat.reshape(B, C, -1).transpose(1, 2))
        logtis = self.classifier(xyz_feat)  # (B, XYZ, C)
        logtis = logtis.permute(0, 2, 1).reshape(B, -1, X, Y, Z)

        return logtis
