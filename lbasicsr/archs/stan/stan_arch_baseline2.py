import math

import torch
import torch.nn.functional as F
from torch import nn as nn

from lbasicsr.utils.registry import ARCH_REGISTRY
from torch.nn.modules.utils import _triple
# from .FAC.kernelconv2d import KernelConv2D
from lbasicsr.archs.arch_util import make_layer, Upsample


class ResnetBlock(nn.Module):
    def __init__(self, num_feat=64, kernel_size=3, dilation=[1, 1], bias=True):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size, 1, (kernel_size-1)//2*dilation[0], dilation[0], bias=bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(num_feat, num_feat, kernel_size, 1, (kernel_size-1)//2*dilation[1], dilation[1], bias=bias)
        )

    def forward(self, x):
        out = self.stem(x) + x
        return out


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

        Args:
            num_feat (int): Channel number of intermediate features.
            squeeze_factor (int): Channel squeeze factor. Default: 16.
            res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x


def conv2dlrelu(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding=((kernel_size-1)//2)*dilation, dilation=dilation, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )


@ARCH_REGISTRY.register()
class STAN(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_feat=64,
                 num_frame=7,
                 n_resgroups=10,
                 n_resblocks=20,
                 reduction=16,
                 center_frame_idx=None,
                 scale=4,
                 res_scale=1):
        super(STAN, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.scale = scale
        kernel_size = 3
        self.gamma = nn.Parameter(torch.ones(1))
        self.act = nn.ReLU(True)
        # self.conv = default_conv

        # define head module
        modules_head = [
            nn.Conv2d(num_in_ch*num_frame, num_feat, kernel_size, stride=1, padding=1),
            nn.Conv2d(num_feat, num_feat, kernel_size, stride=1, padding=1),
            ResnetBlock(num_feat, kernel_size),
            ResnetBlock(num_feat, kernel_size)]
        self.fac_warp = nn.Sequential(
            conv2dlrelu(num_feat, num_feat, kernel_size),
            ResnetBlock(num_feat, kernel_size),
            ResnetBlock(num_feat, kernel_size),
            conv2dlrelu(num_feat, num_feat, kernel_size=1))

        # define body module
        self.RG = nn.ModuleList(
            [ResidualGroup(num_feat, num_block=n_resblocks, squeeze_factor=reduction, res_scale=res_scale)
             for _ in range(n_resgroups)])
        self.conv_last = nn.Conv2d(num_feat, num_feat, kernel_size, stride=1, padding=1)

        # define tail module
        modules_tail = [
            nn.Conv2d(num_feat, num_feat, kernel_size, stride=1, padding=1),
            Upsample(scale, num_feat),
            nn.Conv2d(num_feat, num_in_ch, kernel_size, stride=1, padding=1)]

        self.ref_frame_ex = nn.Conv2d(num_in_ch, num_feat, kernel_size, stride=1, padding=1)
        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)
        self.out_fea_fusion = nn.Conv2d(2 * num_feat, num_feat, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.fusion_conv2d = nn.Conv2d(2 * num_feat, num_feat, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.aligned_fusion_conv2d = nn.Conv2d(2 * num_feat, num_feat, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.epoch = 0

    def forward(self, x, output_last_fea=None):
        b, t, c, h, w = x.size()
        x_center = x[:, self.center_frame_idx, ...].contiguous()    # B x 3 x H x W
        x = x.view(b, -1, h, w)     # B x 3*frame(21) x H x W

        x = self.head(x)    # B x 64 x H x W
        x_center_feature = self.ref_frame_ex(x_center)  # B x 64 x H x W

        if output_last_fea is None:
            output_last_fea = torch.cat([x, x_center_feature], dim=1)   # B x 128 x 64 x 64
        x = self.fac_warp(x)    # B x 64 x H x W
        output_last_fea = self.out_fea_fusion(output_last_fea)     # B x 128 x H x W -> B x 64 x H x W

        # 当前帧邻域与上一帧领域进行特征融合
        x = torch.cat([x, output_last_fea], dim=1)      # B x 128 x H x W
        x = self.fusion_conv2d(x)     # B x 64 x H x W
        # TAda Conv
        # x = self.tadaconv(x, self.conv_rf(x))

        aligned_cat = torch.cat([x_center_feature, x], dim=1)   # B x 128 x H x W
        x = self.aligned_fusion_conv2d(aligned_cat)
        share_source = x
        # =========================================================
        # Share Source Skip Connection
        for i, rg in enumerate(self.RG):
            x = rg(x) + self.gamma * share_source
        x = self.conv_last(x)
        x += share_source
        # =========================================================

        sr = self.tail(x)
        base = F.interpolate(x_center, scale_factor=self.scale, mode='bilinear', align_corners=False)
        sr = sr + base
        return sr, aligned_cat


if __name__ == '__main__':
    # baseline2
    # 将共享的 fea 分开
    pass

