import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from lbasicsr.utils.registry import ARCH_REGISTRY
from torch.nn.modules.utils import _triple
from lbasicsr.archs.arch_util import make_layer, Upsample


class ResnetBlock(nn.Module):
    def __init__(self, num_feat=64, kernel_size=3, dilation=[1, 1], bias=True):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size, 1, (kernel_size - 1) // 2 * dilation[0], dilation[0], bias=bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(num_feat, num_feat, kernel_size, 1, (kernel_size - 1) // 2 * dilation[1], dilation[1], bias=bias)
        )

    def forward(self, x):
        out = self.stem(x) + x
        return out


# class DenseTAdaBlocks(nn.Module):
#     def __init__(self, num_block, num_feat=64, ratio=4, num_grow_ch=16):
#         super(DenseTAdaBlocks, self).__init__()
#
#         self.dense_tada_blocks = nn.ModuleList()
#         for i in range(0, num_block):
#             self.dense_tada_blocks.append(
#                 nn.Sequential(
#                     RouteFuncMLP(num_feat, ratio, kernels=[3, 3]),
#                     TadaConv2d(num_feat, num_feat, kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1],
#                                bias=True, cal_dim='cin'),
#                     nn.LeakyReLU(0.1, True),
#                     RouteFuncMLP(num_feat, ratio, kernels=[3, 3]),
#                     TadaConv2d(num_feat, num_feat, kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1],
#                                bias=True, cal_dim='cin'),
#                 )
#             )


class ResTAdaBlock(nn.Module):
    """

    """

    def __init__(self, num_feat, res_scale=1):
        super(ResTAdaBlock, self).__init__()
        self.res_scale = res_scale
        self.stem = nn.Sequential(
            TAdaConv(num_feat),
            nn.LeakyReLU(0.1, True),
            TAdaConv(num_feat)
        )

    def forward(self, x):
        return x + self.res_scale * self.stem(x)


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


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res


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
                  padding=((kernel_size - 1) // 2) * dilation, dilation=dilation, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )


class TAdaConv(nn.Module):
    def __init__(self, num_feat, ratio: int = 4, kernel_size: int = 3, stride: int = 1, padding: int = 1, bias=True):
        super(TAdaConv, self).__init__()
        self.conv_rf = RouteFuncMLP(
            c_in=num_feat,
            ratio=ratio,
            kernels=[kernel_size, kernel_size]
        )
        self.tadaconv = TadaConv2d(
            in_channels=num_feat,
            out_channels=num_feat,
            kernel_size=[1, kernel_size, kernel_size],
            stride=[1, stride, stride],
            padding=[0, padding, padding],
            bias=bias,
            cal_dim='cin'
        )

    def forward(self, x):
        return self.tadaconv(x, self.conv_rf(x))


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
            out_channels=int(c_in // ratio),
            kernel_size=[kernels[0], 1, 1],
            padding=[kernels[0] // 2, 0, 0]
        )
        self.bn = nn.BatchNorm3d(int(c_in // ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in // ratio),
            out_channels=c_in,
            kernel_size=[kernels[1], 1, 1],
            padding=[kernels[1] // 2, 0, 0],
            bias=False
        )
        self.b.skip_init = True
        self.b.weight.data.zero_()

    def forward(self, x):
        g = self.globalpool(x)  # torch.Size([4, 64, 1, 1, 1])
        x = self.avgpool(x)  # torch.Size([4, 64, 7, 1, 1])
        x = self.a(x + self.g(g))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x  # torch.Size([4, 64, 7, 1, 1])


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
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2])
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
        _, _, c_out, c_in, kh, kw = self.weight.size()  # torch.Size([1, 1, 64, 64, 3, 3])
        # ==================================================================================
        b, c_in, t, h, w = x.size()
        # b, t, c_in, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(1, -1, h, w)  # torch.Size([1, 1792, 32, 32])
        # x = x.reshape(1, -1, h, w)
        # ==================================================================================

        if self.cal_dim == 'cin':
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, 1, C, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(2) * self.weight).reshape(-1, c_in // self.groups, kh, kw)
            # weight: torch.Size([1792, 64, 3, 3])
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
            dilation=self.dilation[1:], groups=self.groups * b * t)  # torch.Size([1, 1792, 30, 30])

        # =========================================================================================
        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0, 2, 1, 3, 4)
        # output = output.view(b, t, c_out, output.size(-2), output.size(-1))
        # =========================================================================================

        return output

    def __repr__(self):
        return f"TAdaConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " + \
               f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim=\"{self.cal_dim}\")"


class SA_adapt(nn.Module):
    def __init__(self, channels):
        super(SA_adapt, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.adapt = SA_conv(channels, channels, 3, 1, 1)

    def forward(self, x, scale, scale2):
        mask = self.mask(x)
        adapted = self.adapt(x, scale, scale2)

        return x + adapted * mask


class SA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False, num_experts=4):
        super(SA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        self.bias = bias

        # FC layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )

        # initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(channels_out, channels_in, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(num_experts, channels_out))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x, scale, scale2):
        # generate routing weights
        scale = torch.ones(1, 1).to(x.device) / scale
        scale2 = torch.ones(1, 1).to(x.device) / scale2
        routing_weights = self.routing(torch.cat((scale, scale2), 1)).view(self.num_experts, 1, 1)

        # fuse experts
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, self.channels_in, self.kernel_size, self.kernel_size)

        if self.bias:
            fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
        else:
            fused_bias = None

        # convolution
        out = F.conv2d(x, fused_weight, fused_bias, stride=self.stride, padding=self.padding)

        return out


def grid_sample(x, offset, scale, scale2):
    # generate grids
    b, _, h, w = x.size()
    grid = np.meshgrid(range(round(scale2 * w)), range(round(scale * h)))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale2 - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) - 1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) - 1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    grid = grid + torch.cat((offset_0, offset_1), 1)
    grid = grid.permute(0, 2, 3, 1)

    # sampling
    output = F.grid_sample(x, grid, padding_mode='zeros', align_corners=True)
    # UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0.
    # Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.

    return output


class SA_upsample(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False):
        super(SA_upsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels // 8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels // 8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale, scale2):
        b, c, h, w = x.size()

        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, round(w * scale2), 1).unsqueeze(0).float().to(x.device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale2) - (torch.floor((coor_hr[1] + 0.5) / scale2 + 1e-3)) - 0.5

        input = torch.cat((
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale2,
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale2 * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)

        # (2) predict filters and offsets
        embedding = self.body(input)
        ## offsets
        offset = self.offset(embedding)

        ## filters
        routing_weights = self.routing(embedding)
        routing_weights = routing_weights.view(self.num_experts, round(scale * h) * round(scale2 * w)).transpose(0,
                                                                                                                 1)  # (h*w) * n

        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(1, round(scale * h), round(scale2 * w), self.channels // 8,
                                               self.channels)

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(1, round(scale * h), round(scale2 * w), self.channels, self.channels // 8)

        # (3) grid sample & spatially varying filtering
        ## grid sample
        fea0 = grid_sample(x, offset, scale, scale2)  ## b * h * w * c * 1
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)  ## b * h * w * c * 1

        ## spatially varying filtering
        out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
        out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        # self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode,
                              align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        # ffted = self.relu(self.bn(ffted))
        ffted = self.lrelu(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=False, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            # nn.BatchNorm2d(out_channels // 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()     # torch.Size([6, 16, 25, 50])
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()    # torch.Size([6, 32, 25, 25])
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class SFB(nn.Module):
    def __init__(self, num_feat=64):
        super(SFB, self).__init__()
        self.resblock = ResnetBlock(num_feat, kernel_size=3)
        self.ffc_path = SpectralTransform(num_feat, num_feat)
        self.conv = nn.Conv2d(
            num_feat * 2, num_feat, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.resblock(x)
        x2 = self.ffc_path(x)
        out = self.conv(torch.cat([x1, x2], dim=1))
        return out


class SRCNN_Module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(9, 1, 5)):
        super(SRCNN_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        out = self.lrelu(self.conv2(out))
        out = self.conv3(out)
        return out


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

        # shallow feature extraction
        # self.ref_frame_ex = nn.Conv2d(num_in_ch, num_feat, kernel_size, stride=1, padding=1)
        self.ref_frame_ex = SRCNN_Module(num_in_ch, num_feat)
        self.shallow_conv3d = nn.Conv3d(num_in_ch, num_feat, (1, kernel_size, kernel_size),
                                        stride=(1, 1, 1), padding=(0, 1, 1), bias=True)

        # deep feature extraction
        # self.sfb = SFB(num_feat)
        # TAda feature extraction
        modules_head = [
            TAdaConv(num_feat),
            ResTAdaBlock(num_feat),
            ResTAdaBlock(num_feat)
        ]
        self.head = nn.Sequential(*modules_head)

        self.fusion_conv1d = nn.Conv2d(num_feat * num_frame, num_feat, kernel_size=1)
        self.fea_ex = nn.Sequential(
            conv2dlrelu(num_feat, num_feat, kernel_size),
            ResnetBlock(num_feat, kernel_size),
            ResnetBlock(num_feat, kernel_size),
            conv2dlrelu(num_feat, num_feat, kernel_size))
        self.out_fea_fusion = nn.Conv2d(2 * num_feat, num_feat, kernel_size=kernel_size, stride=1, padding=1)
        self.fusion_conv2d = nn.Conv2d(2 * num_feat, num_feat, kernel_size=kernel_size, stride=1, padding=1)

        # image reconstruction
        self.align_fusion_conv2d = nn.Conv2d(2 * num_feat, num_feat, kernel_size=kernel_size, stride=1, padding=1)
        self.RG = nn.ModuleList(
            [ResidualGroup(num_feat, num_block=n_resblocks, squeeze_factor=reduction, res_scale=res_scale)
             for _ in range(n_resgroups)])

        # scale-aware feature adaption block
        self.K = 1
        self.sa_adapt = nn.ModuleList(
            [SA_adapt(num_feat) for _ in range(n_resgroups // self.K)])
        self.conv_last = nn.Conv2d(num_feat, num_feat, kernel_size, stride=1, padding=1)

        # ====================== arbitrary scale upsampling =============================
        self.sa_upsample = SA_upsample(num_feat)
        self.tail = nn.Conv2d(num_feat, num_in_ch, kernel_size=kernel_size, stride=1,
                              padding=(kernel_size // 2), bias=True)
        # ===============================================================================

    def set_scale(self, scale):
        self.scale = scale

    def forward(self, x, output_last_fea=None):
        b, t, c, h, w = x.size()
        x_center = x[:, self.center_frame_idx, ...].contiguous()  # B x 3 x H x W

        # ======================== shallow feature extraction =========================
        x_center_feature = self.ref_frame_ex(x_center)  # B x 64 x H x W
        x = x.permute(0, 2, 1, 3, 4)  # B x T x C x H x W -> B x C x T x H x W for Conv3D
        x = self.shallow_conv3d(x)  # B x 64 x 7 x H x W
        # =============================================================================

        # ======================== deep feature extraction ============================
        # x_center_feature = self.sfb(x_center_feature)

        x = self.head(x)  # B x 64 x 7 x H x W
        x = x.permute(0, 2, 1, 3, 4).reshape(b, -1, h, w)  # B x 7 x 64 x H x W -> B x (7*64) x H x W
        x = self.fusion_conv1d(x)  # B x 64 x H x W
        x = self.fea_ex(x)  # B x 64 x H x W

        # ----------------------------------------------------------------------------
        # F_{t-1}
        if output_last_fea is None:
            output_last_fea = torch.cat([x, x_center_feature], dim=1)  # B x 128 x 64 x 64
        output_last_fea = self.out_fea_fusion(output_last_fea)  # B x 128 x H x W -> B x 64 x H x W
        # ----------------------------------------------------------------------------

        # 当前帧邻域与上一帧邻域进行特征融合
        x = torch.cat([x, output_last_fea], dim=1)  # B x 128 x H x W
        x = self.fusion_conv2d(x)  # B x 64 x H x W

        aligned_cat = torch.cat([x_center_feature, x], dim=1)  # B x 128 x H x W
        x = self.align_fusion_conv2d(aligned_cat)
        # =============================================================================

        # ========================== image reconstruction =============================
        # Share Source Skip Connection
        share_source = x
        for i, rg in enumerate(self.RG):
            # x = rg(x) + self.gamma * share_source
            x = rg(x)
            if (i + 1) % self.K == 0:
                x = self.sa_adapt[i](x, self.scale, self.scale)
            x = x + self.gamma * share_source
        x = self.conv_last(x)
        x += share_source

        # ---------- arbitrary scale upsampling -----------------
        sr = self.sa_upsample(x, self.scale, self.scale)
        sr = self.tail(sr)
        # =======================================================

        return sr, aligned_cat


if __name__ == '__main__':
    # add
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
    # nohup python -u lbasicsr/train.py -opt options/train/STAN/train_STAN+tada+asup_Vimeo90K_BIx4.yml 2>&1 &
