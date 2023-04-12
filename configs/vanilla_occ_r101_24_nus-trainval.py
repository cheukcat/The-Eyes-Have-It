_base_ = [
    './_base_/dataset.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

load_from = './ckpts/r101_dcn_fcos3d_pretrain.pth'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

_num_cams_ = 6
_dim_ = 256
tpv_h_ = 100
tpv_w_ = 100
tpv_z_ = 8
scale = 2
grid_size_vt = [tpv_h_, tpv_w_, tpv_z_]
grid_size = [tpv_h_ * scale, tpv_w_ * scale, tpv_z_ * scale]
num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]
nbr_class = 18

# model settings
find_unused_parameters=True
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='VanillaOccupancy',
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    view_transformer=dict(
        type='InverseMatrixVT',
        feature_strides=[4, 8, 16, 32],
        in_index=2,
        grid_size=grid_size_vt,
        x_bound=[-51.2, 51.2],
        y_bound=[-51.2, 51.2],
        z_bound=[-5., 3.]),
    occ_head=dict(
        type='VanillaHead',
        in_channels=_dim_,
        channels=_dim_,
        num_classes=nbr_class,
        scale=scale,
        norm_cfg=norm_cfg)
)
