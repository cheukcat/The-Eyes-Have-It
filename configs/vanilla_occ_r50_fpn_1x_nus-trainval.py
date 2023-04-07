_base_ = [
    './_base_/dataset.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

_num_cams_ = 6
tpv_h_ = 100
tpv_w_ = 100
tpv_z_ = 8
scale_h = 1
scale_w = 1
scale_z = 1
grid_size = [tpv_h_ * scale_h, tpv_w_ * scale_w, tpv_z_ * scale_z]
num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]
nbr_class = 18

# model settings
find_unused_parameters=True
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='VanillaOccupancy',
    img_backbone=dict(
        # _delete_=True,
        type='RegNet',
        arch='regnetx_3.2gf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://regnetx_3.2gf')),
    img_neck=dict(
        type='BiFPN',
        in_channels=[96, 192, 432, 1008],
        out_channels=256,
        strides=[8, 16, 32, 64],
        num_outs=5,
        stack=2,
        norm_cfg=norm_cfg),
    view_transformer=dict(
        type='InverseMatrixVT',
        feature_strides=[4, 8, 16, 32, 64],
        in_index=2,
        grid_size=grid_size,
        x_bound=[-51.2, 51.2],
        y_bound=[-51.2, 51.2],
        z_bound=[-5., 3.]),
    occ_head=dict(
        type='VanillaHead',
        in_channels=256,
        channels=256,
        num_classes=nbr_class,
        norm_cfg=norm_cfg)
)
