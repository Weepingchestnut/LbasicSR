import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from lbasicsr.archs.arch_util import make_layer
from lbasicsr.utils.registry import ARCH_REGISTRY


# ---------------------------
# arbitrary scale upsampling
# ---------------------------
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


def generate_it(x, t, win_size=3):
    index = np.array([t - win_size // 2 + i for i in range(win_size)])
    # print("index =", index)
    it = x[:, index, ...]

    return it


class SAResidualBlock(nn.Module):
    """
    先仿照 OVSR 中的 PFRB 构建多帧隐特征的残差融合块
    """

    def __init__(self, num_feat=64, num_frame=3, act=nn.LeakyReLU(0.2, True)):
        super(SAResidualBlock, self).__init__()
        self.nfe = num_feat
        self.nfr = num_frame
        self.act = act
        self.conv0 = nn.Sequential(*[nn.Conv2d(self.nfe, self.nfe, kernel_size=3, stride=1, padding=1)
                                     for _ in range(num_frame)])
        # 1x1 conv to reduce dim
        self.conv1 = nn.Conv2d(self.nfe * num_frame, self.nfe, kernel_size=1, stride=1)
        self.conv2 = nn.Sequential(*[nn.Conv2d(self.nfe * 2, self.nfe, kernel_size=3, stride=1, padding=1)
                                     for _ in range(num_frame)])

    def forward(self, x):
        x1 = [self.act(self.conv0[i](x[i])) for i in range(self.nfr)]

        merge = torch.cat(x1, dim=1)
        base = self.act(self.conv1(merge))

        x2 = [torch.cat([base, i], 1) for i in x1]
        x2 = [self.act(self.conv2[i](x2[i])) for i in range(self.nfr)]

        return [torch.add(x[i], x2[i]) for i in range(self.nfr)]


class WindowUnit_l1(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_feat=64,
                 win_size=3,
                 num_block=4):
        super().__init__()
        self.nf = num_feat
        self.ws = win_size
        self.act = nn.LeakyReLU(0.2, True)

        # center frame conv
        self.conv_c = nn.Conv2d(num_in_ch, self.nf, kernel_size=3, stride=1, padding=1)
        # support frame conv
        self.conv_sup = nn.Conv2d(num_in_ch * (win_size - 1), self.nf, kernel_size=3, stride=1, padding=1)

        self.blocks = nn.Sequential(*[SAResidualBlock(self.nf, 3, self.act) for i in range(num_block)])
        self.merge = nn.Conv2d(3 * self.nf, self.nf, kernel_size=3, stride=1, padding=1)

    def forward(self, x, h_past):
        b, t, c, h, w = x.size()

        # center frame in slide window
        x_c = x[:, t // 2]
        # the index of support frame
        sup_index = list(range(t))
        sup_index.pop(t // 2)
        # support frame
        x_sup = x[:, sup_index]
        x_sup = x_sup.reshape(b, (t - 1) * c, h, w)
        # hidden feature of support frame and center frame
        h_sup = self.act(self.conv_sup(x_sup))
        h_c = self.act(self.conv_c(x_c))
        # 聚合中间帧、支持帧以及相对的过去帧的信息
        h_feat = [h_c, h_sup, h_past]

        h_feat = self.blocks(h_feat)  # after some residual block
        h_feat = self.merge(torch.cat(h_feat, dim=1))

        return h_feat


class WindowUnit_l2(nn.Module):
    def __init__(self,
                 num_feat=64,
                 win_size=3,
                 num_block=4):
        super().__init__()
        self.nf = num_feat
        self.ws = win_size
        self.act = nn.LeakyReLU(0.2, True)

        # hidden feature conv
        self.conv_h = nn.Sequential(*[nn.Conv2d(self.nf * 2, self.nf, kernel_size=3, stride=1, padding=1)
                                      for i in range(win_size)])
        self.blocks = nn.Sequential(*[SAResidualBlock(self.nf, self.ws, self.act) for i in range(num_block)])
        self.merge = nn.Conv2d(self.ws * self.nf, self.nf, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h_feat = [self.act(self.conv_h[i](x[i])) for i in range(self.ws)]
        h_feat = self.blocks(h_feat)
        h_feat = self.merge(torch.cat(h_feat, dim=1))

        return h_feat


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
        mask = self.mask(x)  # torch.Size([1, 1, 96, 116])
        adapted = self.adapt(x, scale, scale2)  # torch.Size([1, 64, 96, 117])

        if mask.size(-1) != adapted.size(-1):
            if mask.size(-1) - adapted.size(-1) > 0:
                mask = mask[..., 0:adapted.size(-1)]
            else:
                mask = F.pad(mask, (0, 1), "constant", 1)
        if mask.size(-2) != adapted.size(-2):
            if mask.size(-2) - adapted.size(-2) > 0:
                mask = mask[..., 0:adapted.size(-2), :]
            else:
                mask = F.pad(mask, (0, 0, 0, 1), "constant", 1)

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


@ARCH_REGISTRY.register()
class ASVSR(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_feat=64,
                 num_frame=7,
                 window_size=5,
                 n_resgroups=5,
                 n_resblocks=10,
                 center_frame_idx=None,
                 ):
        super(ASVSR, self).__init__()
        self.scale = (4, 4)
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.num_feat = num_feat
        self.num_frame = num_frame
        self.window_size = window_size

        # hidden state to alignment
        self.f2p_win = WindowUnit_l1(num_in_ch, num_feat, win_size=window_size, num_block=4)
        self.p2f_win = WindowUnit_l1(num_in_ch, num_feat, win_size=window_size, num_block=4)
        self.h_win = WindowUnit_l2(num_feat, win_size=(num_frame - window_size + 1), num_block=2)

        # image reconstruction
        self.RG = nn.ModuleList(
            [ResidualGroup(num_feat, num_block=n_resblocks, squeeze_factor=16, res_scale=1)
             for _ in range(n_resgroups)])
        self.K = 1
        self.sa_adapt = nn.ModuleList([SA_adapt(num_feat) for _ in range(n_resgroups // self.K)])
        self.gamma = nn.Parameter(torch.ones(1))
        self.conv_last = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)

        # arbitrary scale upsampling
        self.sa_upsample = SA_upsample(num_feat)
        self.tail = nn.Conv2d(num_feat, num_in_ch, kernel_size=3, stride=1, padding=1, bias=True)

    def set_scale(self, scale):
        self.scale = scale

    def forward(self, x):
        b, t, c, h, w = x.size()
        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract hidden features ----------------------------------------------------------------------
        h_f2p_list = []  # hidden feature (future to past)
        h_p2f_list = []  # hidden feature (past to future)
        ht_f2p = torch.zeros((b, self.num_feat, h, w), dtype=torch.float, device=x.device)
        ht_p2f = torch.zeros((b, self.num_feat, h, w), dtype=torch.float, device=x.device)

        for idx in range(t - self.window_size + 1):
            # past <------ future (f2p): 0,1,[2,3,(4),5,6] --> 0,[1,2,(3),4,5],6 --> [0,1,(2),3,4],5,6
            cur_t = t - 1 - self.window_size // 2 - idx
            # print(cur_t)
            it = generate_it(x, cur_t, self.window_size)  # torch.Size([b, 5, 3, 100, 100])
            ht_f2p = self.f2p_win(it, ht_f2p)  # first hidden layer
            h_f2p_list.insert(0, ht_f2p)

            # past ------> future (p2f): [0,1,(2),3,4],5,6 --> 0,[1,2,(3),4,5],6 --> 0,1,[2,3,(4),5,6]
            cur_t = idx + self.window_size // 2
            # print(cur_t)
            it = generate_it(x, cur_t, self.window_size)
            ht_p2f = self.p2f_win(it, ht_p2f)  # first hidden layer
            h_p2f_list.append(ht_p2f)

        h_feat = [torch.cat([h_f2p_list[i], h_p2f_list[i]], dim=1) for i in range(3)]
        h_feat = self.h_win(h_feat)

        # image reconstruction -------------------------------------------------
        share_source = h_feat
        for i, rg in enumerate(self.RG):
            h_feat = rg(h_feat)
            if (i + 1) % self.K == 0:
                h_feat = self.sa_adapt[i](h_feat, self.scale[0], self.scale[1])
            h_feat = h_feat + self.gamma * share_source
        h_feat = self.conv_last(h_feat)
        h_feat += share_source

        # arbitrary scale upsampling -------------------------------------------
        sr = self.sa_upsample(h_feat, self.scale[0], self.scale[1])
        sr = self.tail(sr)
        sr = sr + T.Resize(size=(round(h*self.scale[0]), round(w*self.scale[1])), interpolation=InterpolationMode.BICUBIC,
                           antialias=True)(x_center)

        return sr


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    scale = (2, 1.5)
    asvsr = ASVSR().to(device)
    asvsr.set_scale(scale)
    asvsr.eval()

    print(
        "ASVSR have {:.3f}M parameters in total".format(sum(x.numel() for x in asvsr.parameters()) / 1000000.0))

    input = torch.rand(1, 7, 3, 100, 100).to(device)

    with torch.no_grad():
        out = asvsr(input)

    print(out.shape)
