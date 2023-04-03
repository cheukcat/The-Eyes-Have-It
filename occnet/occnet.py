from mmcv.runner import auto_fp16, BaseModule
from mmseg.models import SEGMENTORS, builder
from occnet import build_view_transformer
import warnings


@SEGMENTORS.register_module()
class VanillaOccupancy(BaseModule):

    def __init__(self,
                 img_backbone,
                 img_neck=None,
                 view_transformer=None,
                 occ_head=None,
                 pretrained=None):
        super().__init__()
        self.fp16_enabled = False
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            img_backbone.pretrained = pretrained
        self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        self.view_transformer = build_view_transformer(view_transformer)
        self.occ_head = builder.build_head(occ_head)

    @auto_fp16(apply_to='img')
    def extract_img_feat(self, img):
        """Extract features of images."""
        B, N, C, H, W = img.shape
        img = img.view(B * N, C, H, W)
        img_feats = self.img_backbone(img)
        if hasattr(self, 'img_neck'):
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.shape
            img_feats_reshaped.append(img_feat.view(B, -1, C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img', 'points'))
    def forward(self, img, img_metas=None, points=None, **kwargs):
        """Forward training function."""
        img_feats = self.extract_img_feat(img=img)
        occ_feats = self.view_transformer(img_feats, img_metas, **kwargs)
        occ_outs = self.occ_head(occ_feats, points, **kwargs)
        return occ_outs
