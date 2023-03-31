import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import ConvTranspose2d


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)


# init bilinear kernel weights
def bilinear_kernel(in_channels, out_channels, kernel_size):
    # return a bilinear filter tensor
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


class BilinearDeconvolution(nn.Module):
    """
    This module is for J3 platform, bilinear upsample is not supported on J3, but deconv is,
    therefore we implement a deconv layer initialized with bilinear upsample kernel.
    """

    def __init__(self,
                 channels,
                 scale_factor=2,
                 requires_grad=False):
        super(BilinearDeconvolution, self).__init__()
        # only upsample scale factor 2 is supported
        assert scale_factor == 2
        self.channels = channels
        self.deconv = ConvTranspose2d(
            self.channels,
            self.channels,
            4,
            stride=2,
            padding=1,
            groups=self.channels,
            bias=False
        )
        self.deconv.weight.data = bilinear_kernel(self.channels, 1, 4)
        if not requires_grad:
            self.deconv.requires_grad_(False)

    def forward(self, x):
        return self.deconv(x)
