_base_ = [
    './_base_/dataset.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

load_from = './ckpts/r101_dcn_fcos3d_pretrain.pth'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

_num_cams_ = 6
_dim_ = 512
tpv_h_ = 100
tpv_w_ = 100
tpv_z_ = 8
scale = 2
grid_size = [tpv_h_ * scale, tpv_w_ * scale, tpv_z_ * scale]
nbr_class = 18

# model settings
find_unused_parameters = True
model = dict(
    type='VanillaOccupancy',
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=2,
        relu_before_extra_convs=True),
    view_transformer=dict(
        type='AllYouNeedIsInYourEyes',
        feature_strides=[16, 32],
        in_index=1,
        grid_size=[tpv_h_, tpv_w_, tpv_z_],
        x_bound=[-51.2, 51.2],
        y_bound=[-51.2, 51.2],
        z_bound=[-5., 3.],
        num_cams_per_quadrant=_num_cams_ // 2),
    occ_head=dict(
        type='MultiViewFusionHead',
        in_channels=_dim_,
        channels=_dim_ // 2,
        num_classes=nbr_class,
        depth_size=[tpv_h_ * scale // 2, tpv_w_ * scale // 2, tpv_z_ * scale],
        scale=scale)
)
