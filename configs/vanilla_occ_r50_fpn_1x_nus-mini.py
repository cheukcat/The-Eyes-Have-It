_base_ = [
    './_base_/dataset.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
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
