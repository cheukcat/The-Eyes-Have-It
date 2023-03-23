import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
from occnet import VIEW_TRANSFORMERS


@VIEW_TRANSFORMERS.register_module()
class InverseMatrixVT(BaseModule):
    def __init__(self,
                 x_bound=[-51.2, 51.2],
                 y_bound=[-51.2, 51.2],
                 z_bound=[-5., 3.]):
        super().__init__()
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound

    def create_gridmap_anchor(self, grid_size):
        # cal
        pass