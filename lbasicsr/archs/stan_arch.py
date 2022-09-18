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


class RouteFuncMLP(nn.Module):
    """The routing function for generating the calibration weights.

    Args:
        c_in (int): number of input channels.
        ratio (int): reduction ratio for the routing function.
        kernels (list): temporal kernel size of the stacked 1D convolutions
    """
    def __init__(self, c_in, ratio, kernels, bn_eps=1e-5, bn_mmt=0.1):
        super(RouteFuncMLP, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0
        )
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernels[0], 1, 1],
            padding=[kernels[0]//2, 0, 0]
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernels[1], 1, 1],
            padding=[kernels[1]//2, 0, 0],
            bias=False
        )
        self.b.skip_init = True
        self.b.weight.data.zero_()

    def forward(self, x):
        g = self.globalpool(x)      # torch.Size([4, 64, 1, 1, 1])
        x = self.avgpool(x)         # torch.Size([4, 64, 7, 1, 1])
        x = self.a(x + self.g(g))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x    # torch.Size([4, 64, 7, 1, 1])


class TadaConv2d(nn.Module):
    """
    Performs temporally adaptive 2D convolution.
    Currently, only application on 5D tensors is supported, which makes TAdaConv2d
        essentially a 3D convolution with temporal kernel size of 1.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel_size (list): kernel size of TAdaConv2d.
        stride (list): stride for the convolution in TAdaConv2d.
        padding (list): padding for the convolution in TAdaConv2d.
        dilation (list): dilation of the convolution in TAdaConv2d.
        groups (int): number of groups for TAdaConv2d.
        bias (bool): whether to use bias in TAdaConv2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cal_dim='cin'):
        super(TadaConv2d, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1
        assert cal_dim in ['cin', 'cout']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim

        # base weights (W_b)
        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels//groups, kernel_size[1], kernel_size[2])
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):
        """
        Args:
            :param x: feature to perform convolution on.
            :param alpha: calibration weight for the base weights.
                W_t = alpha_t * W_b
            :return:
        """
        _, _, c_out, c_in, kh, kw = self.weight.size()      # torch.Size([1, 1, 64, 64, 3, 3])
        b, c_in, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(1, -1, h, w)   # torch.Size([1, 1792, 32, 32])

        if self.cal_dim == 'cin':
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, 1, C, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(2) * self.weight).reshape(-1, c_in // self.groups, kh, kw)     # torch.Size([1792, 64, 3, 3])
        elif self.cal_dim == "cout":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, C, 1, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(3) * self.weight).reshape(-1, c_in // self.groups, kh, kw)

        bias = None
        if self.bias is not None:
            # in the official implementation of TAda2D,
            # there is no bias term in the convs
            # hence the performance with bias is not validated
            bias = self.bias.repeat(b, t, 1).reshape(-1)
        output = F.conv2d(
            x, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:],
            dilation=self.dilation[1:], groups=self.groups * b * t)     # torch.Size([1, 1792, 30, 30])

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0, 2, 1, 3, 4)

        return output

    def __repr__(self):
        return f"TAdaConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " + \
               f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim=\"{self.cal_dim}\")"


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

        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)
        self.fea = nn.Conv2d(2*num_feat, num_feat, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.epoch = 0
        # ============================================================
        # self.kconv_warp = KernelConv2D.KernelConv2D(kernel_size=3)
        self.conv_rf = RouteFuncMLP(
            c_in=num_feat,
            ratio=4,
            kernels=[3, 3]
        )
        self.tadaconv = TadaConv2d(
            in_channels=num_feat,
            out_channels=num_feat,
            kernel_size=[1, 3, 3],
            stride=[1, 1, 1],
            padding=[0, 1, 1],
            bias=False,
            cal_dim='cin'
        )
        # ===========================================================
        self.fac_warp = nn.Sequential(
            conv2dlrelu(num_feat, num_feat, kernel_size),
            ResnetBlock(num_feat, kernel_size),
            ResnetBlock(num_feat, kernel_size),
            conv2dlrelu(num_feat, num_feat, kernel_size=1))
        self.kconv4 = conv2dlrelu(num_feat, num_feat, kernel_size=1)
        self.ex = nn.Conv2d(num_in_ch, num_feat, kernel_size, stride=1, padding=1)

    def forward(self, x, output_last_fea=None):
        b, t, c, h, w = x.size()
        x_center = x[:, self.center_frame_idx, ...].contiguous()    # B x 3 x H x W
        x = x.view(b, -1, h, w)     # B x 3*frame(21) x H x W

        x = self.head(x)    # B x 64 x H x W
        x_center_feature = self.ex(x_center)  # B x 64 x H x W

        if output_last_fea is None:
            output_last_fea = torch.cat([x, x_center_feature], dim=1)   # B x 128 x 64 x 64
        x = self.fac_warp(x)    # B x 64 x H x W
        output_last_fea = self.fea(output_last_fea)     # B x 128 x H x W -> B x 64 x H x W

        # 当前帧邻域与上一帧领域进行特征融合
        x = torch.cat([x, output_last_fea], dim=1)      # B x 128 x H x W
        x = self.fea(x)     # B x 64 x H x W
        # TAda Conv
        # x = self.tadaconv(x, self.conv_rf(x))

        aligned_cat = torch.cat([x_center_feature, x], dim=1)   # B x 128 x H x W
        x = self.fea(aligned_cat)
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
    batch_size = 4
    frame_num = 7
    num_feature = 64
    x = torch.randn((batch_size, num_feature, frame_num, 32, 32))

    conv_rf = RouteFuncMLP(c_in=64, ratio=4, kernels=[3, 3])
    conv = TadaConv2d(
        in_channels=64,
        out_channels=64,
        kernel_size=[1, 3, 3],
        stride=[1, 1, 1],
        bias=False,
        cal_dim='cin'
    )

    out = conv(x, conv_rf(x))

