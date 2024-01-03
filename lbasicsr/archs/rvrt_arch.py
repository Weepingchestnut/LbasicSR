import os
from distutils.version import LooseVersion

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from functools import reduce, lru_cache
from operator import mul

from einops import rearrange
from einops.layers.torch import Rearrange

from lbasicsr.archs.op.deform_attn import deform_attn, DeformAttnPack
from lbasicsr.metrics.runtime import VSR_runtime_test
from lbasicsr.utils.registry import ARCH_REGISTRY


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.


    Returns:
        Tensor: Warped image or feature map.
    """
    n, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device),
                                    torch.arange(0, w, dtype=x.dtype, device=x.device), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    return output


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
        return_levels (list[int]): return flows of different levels. Default: [5].
    """

    def __init__(self, load_path=None, return_levels=[5]):
        super(SpyNet, self).__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            if not os.path.exists(load_path):
                import requests
                url = 'https://github.com/JingyunLiang/RVRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth'
                r = requests.get(url, allow_redirects=True)
                print(f'downloading SpyNet pretrained model from {url}')
                os.makedirs(os.path.dirname(load_path), exist_ok=True)
                open(load_path, 'wb').write(r.content)

            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

            if level in self.return_levels:
                scale = 2 ** (5 - level)  # level=5 (scale=1), level=4 (scale=2), level=3 (scale=4), level=2 (scale=8)
                flow_out = F.interpolate(input=flow, size=(h // scale, w // scale), mode='bilinear',
                                         align_corners=False)
                flow_out[:, 0, :, :] *= float(w // scale) / float(w_floor // scale)
                flow_out[:, 1, :, :] *= float(h // scale) / float(h_floor // scale)
                flow_list.insert(0, flow_out)

        return flow_list

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list


class GuidedDeformAttnPack(DeformAttnPack):
    """Guided deformable attention module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        attention_window (int or tuple[int]): Attention window size. Default: [3, 3].
        attention_heads (int): Attention head number.  Default: 12.
        deformable_groups (int): Deformable offset groups.  Default: 12.
        clip_size (int): clip size. Default: 2.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
    Ref:
        Recurrent Video Restoration Transformer with Guided Deformable Attention

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(GuidedDeformAttnPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv3d(self.in_channels * (1 + self.clip_size) + self.clip_size * 2, 64, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, self.clip_size * self.deformable_groups * self.attn_size * 2, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0)),
        )
        self.init_offset()

        # proj to a higher dimension can slightly improve the performance
        self.proj_channels = int(self.in_channels * 2)
        self.proj_q = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj_k = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj_v = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                  nn.Linear(self.proj_channels, self.in_channels),
                                  Rearrange('n d h w c -> n d c h w'))
        self.mlp = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                 Mlp(self.in_channels, self.in_channels * 2, self.in_channels),
                                 Rearrange('n d h w c -> n d c h w'))

    def init_offset(self):
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, q, k, v, v_prop_warped, flows, return_updateflow):
        offset1, offset2 = torch.chunk(self.max_residue_magnitude * torch.tanh(
            self.conv_offset(torch.cat([q] + v_prop_warped + flows, 2).transpose(1, 2)).transpose(1, 2)), 2, dim=2)
        offset1 = offset1 + flows[0].flip(2).repeat(1, 1, offset1.size(2) // 2, 1, 1)
        offset2 = offset2 + flows[1].flip(2).repeat(1, 1, offset2.size(2) // 2, 1, 1)
        offset = torch.cat([offset1, offset2], dim=2).flatten(0, 1)

        b, t, c, h, w = offset1.shape
        q = self.proj_q(q).view(b * t, 1, self.proj_channels, h, w)
        kv = torch.cat([self.proj_k(k), self.proj_v(v)], 2)
        v = deform_attn(q, kv, offset, self.kernel_h, self.kernel_w, self.stride, self.padding, self.dilation,
                        self.attention_heads, self.deformable_groups, self.clip_size).view(b, t, self.proj_channels, h,
                                                                                           w)
        v = self.proj(v)
        v = v + self.mlp(v)

        if return_updateflow:
            return v, offset1.view(b, t, c // 2, 2, h, w).mean(2).flip(2), offset2.view(b, t, c // 2, 2, h, w).mean(
                2).flip(2)
        else:
            return v


def window_partition(x, window_size):
    """ Partition the input into windows. Attention will be conducted within the windows.

    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)

    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """Reverse windows back to the original input. Attention was conducted within the windows.

    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    """ Get the window size and the shift size """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """ Compute attnetion mask for input of size (D, H, W). @lru_cache caches each stage results. """

    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class Mlp(nn.Module):
    """ Multilayer perceptron.

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    """ Window based multi-head self attention.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        self.register_buffer("relative_position_index", self.get_position_index(window_size))
        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """

        # self attention
        B_, N, C = x.shape
        qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        x_out = self.attention(q, k, v, mask, (B_, N, C))

        # projection
        x = self.proj(x_out)

        return x

    def attention(self, q, k, v, mask, x_shape):
        B_, N, C = x_shape
        attn = (q * self.scale) @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)  # Wd*Wh*Ww, Wd*Wh*Ww,nH
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, -1, dtype=q.dtype)  # Don't use attn.dtype after addition!
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return x

    def get_position_index(self, window_size):
        ''' Get pair-wise relative position index for each token inside the window. '''

        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

        return relative_position_index


class STL(nn.Module):
    """ Swin Transformer Layer (STL).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=(2, 8, 8),
                 shift_size=(0, 0, 0),
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint_attn = use_checkpoint_attn
        self.use_checkpoint_ffn = use_checkpoint_ffn

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # attention / shifted attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        return x

    def forward_part2(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        # attention
        if self.use_checkpoint_attn:
            x = x + checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = x + self.forward_part1(x, mask_matrix)

        # feed-forward
        if self.use_checkpoint_ffn:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class STG(nn.Module):
    """ Swin Transformer Group (STG).

    Args:
        dim (int): Number of feature channels
        input_resolution (tuple[int]): Input resolution.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (6,8,8).
        shift_size (tuple[int]): Shift size for mutual and self attention. Default: None.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=[2, 8, 8],
                 shift_size=None,
                 mlp_ratio=2.,
                 qkv_bias=False,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = list(i // 2 for i in window_size) if shift_size is None else shift_size

        # build blocks
        self.blocks = nn.ModuleList([
            STL(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                use_checkpoint_attn=use_checkpoint_attn,
                use_checkpoint_ffn=use_checkpoint_ffn
            )
            for i in range(depth)])

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for attention
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')

        return x


class RSTB(nn.Module):
    """ Residual Swin Transformer Block (RSTB).

    Args:
        kwargs: Args for RSTB.
    """

    def __init__(self, **kwargs):
        super(RSTB, self).__init__()
        self.input_resolution = kwargs['input_resolution']

        self.residual_group = STG(**kwargs)
        self.linear = nn.Linear(kwargs['dim'], kwargs['dim'])

    def forward(self, x):
        return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4)


class RSTBWithInputConv(nn.Module):
    """RSTB with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        kernel_size (int): Size of kernel of the first conv.
        stride (int): Stride of the first conv.
        group (int): Group of the first conv.
        num_blocks (int): Number of residual blocks. Default: 2.
         **kwarg: Args for RSTB.
    """

    def __init__(self, in_channels=3, kernel_size=(1, 3, 3), stride=1, groups=1, num_blocks=2, **kwargs):
        super().__init__()

        main = []
        main += [Rearrange('n d c h w -> n c d h w'),
                 nn.Conv3d(in_channels,
                           kwargs['dim'],
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
                           groups=groups),
                 Rearrange('n c d h w -> n d h w c'),
                 nn.LayerNorm(kwargs['dim']),
                 Rearrange('n d h w c -> n c d h w')]

        # RSTB blocks
        kwargs['use_checkpoint_attn'] = kwargs.pop('use_checkpoint_attn')[0]
        kwargs['use_checkpoint_ffn'] = kwargs.pop('use_checkpoint_ffn')[0]
        main.append(make_layer(RSTB, num_blocks, **kwargs))

        main += [Rearrange('n c d h w -> n d h w c'),
                 nn.LayerNorm(kwargs['dim']),
                 Rearrange('n d h w c -> n d c h w')]

        self.main = nn.Sequential(*main)

    def forward(self, x):
        """
        Forward function for RSTBWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, t, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, t, out_channels, h, w)
        """
        return self.main(x)


class Upsample(nn.Sequential):
    """Upsample module for video SR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        assert LooseVersion(torch.__version__) >= LooseVersion('1.8.1'), \
            'PyTorch version >= 1.8.1 to support 5D PixelShuffle.'

        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv3d(num_feat, 4 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
                m.append(Rearrange('n c d h w -> n d c h w'))
                m.append(nn.PixelShuffle(2))
                m.append(Rearrange('n c d h w -> n d c h w'))
                m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        elif scale == 3:
            m.append(nn.Conv3d(num_feat, 9 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            m.append(Rearrange('n c d h w -> n d c h w'))
            m.append(nn.PixelShuffle(3))
            m.append(Rearrange('n c d h w -> n d c h w'))
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


# @ARCH_REGISTRY.register()
class RVRT(nn.Module):
    """ Recurrent Video Restoration Transformer with Guided Deformable Attention (RVRT).
            A PyTorch impl of : `Recurrent Video Restoration Transformer with Guided Deformable Attention`  -
              https://arxiv.org/pdf/2205.00000

        Args:
            upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
            clip_size (int): Size of clip in recurrent restoration transformer.
            img_size (int | tuple(int)): Size of input video. Default: [2, 64, 64].
            window_size (int | tuple(int)): Window size. Default: (2,8,8).
            num_blocks (list[int]): Number of RSTB blocks in each stage.
            depths (list[int]): Depths of each RSTB.
            embed_dims (list[int]): Number of linear projection output channels.
            num_heads (list[int]): Number of attention head of each stage.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
            qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
            norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
            inputconv_groups (int): Group of the first convolution layer in RSTBWithInputConv. Default: [1,1,1,1,1,1]
            spynet_path (str): Pretrained SpyNet model path.
            deformable_groups (int): Number of deformable groups in deformable attention. Default: 12.
            attention_heads (int): Number of attention heads in deformable attention. Default: 12.
            attention_window (list[int]): Attention window size in aeformable attention. Default: [3, 3].
            nonblind_denoising (bool): If True, conduct experiments on non-blind denoising. Default: False.
            use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
            use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
            no_checkpoint_attn_blocks (list[int]): Layers without torch.checkpoint for attention modules.
            no_checkpoint_ffn_blocks (list[int]): Layers without torch.checkpoint for feed-forward modules.
            cpu_cache_length: (int): Maximum video length without cpu caching. Default: 100.
        """

    def __init__(self,
                 upscale=4,
                 clip_size=2,
                 img_size=[2, 64, 64],
                 window_size=[2, 8, 8],
                 num_blocks=[1, 2, 1],
                 depths=[2, 2, 2],
                 embed_dims=[144, 144, 144],
                 num_heads=[6, 6, 6],
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 inputconv_groups=[1, 1, 1, 1, 1, 1],
                 spynet_path=None,
                 max_residue_magnitude=10,
                 deformable_groups=12,
                 attention_heads=12,
                 attention_window=[3, 3],
                 nonblind_denoising=False,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 no_checkpoint_attn_blocks=[],
                 no_checkpoint_ffn_blocks=[],
                 cpu_cache_length=100
                 ):

        super().__init__()
        self.upscale = upscale
        self.clip_size = clip_size
        self.nonblind_denoising = nonblind_denoising
        use_checkpoint_attns = [False if i in no_checkpoint_attn_blocks else use_checkpoint_attn for i in range(100)]
        use_checkpoint_ffns = [False if i in no_checkpoint_ffn_blocks else use_checkpoint_ffn for i in range(100)]
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        self.spynet = SpyNet(spynet_path)

        # shallow feature extraction
        if self.upscale != 1:
            # video sr
            self.feat_extract = RSTBWithInputConv(in_channels=3,
                                                  kernel_size=(1, 3, 3),
                                                  groups=inputconv_groups[0],
                                                  num_blocks=num_blocks[0],
                                                  dim=embed_dims[0],
                                                  input_resolution=[1, img_size[1], img_size[2]],
                                                  depth=depths[0],
                                                  num_heads=num_heads[0],
                                                  window_size=[1, window_size[1], window_size[2]],
                                                  mlp_ratio=mlp_ratio,
                                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  norm_layer=norm_layer,
                                                  use_checkpoint_attn=[False],
                                                  use_checkpoint_ffn=[False]
                                                  )
        else:
            # video deblurring/denoising
            self.feat_extract = nn.Sequential(Rearrange('n d c h w -> n c d h w'),
                                              nn.Conv3d(4 if self.nonblind_denoising else 3, embed_dims[0], (1, 3, 3),
                                                        (1, 2, 2), (0, 1, 1)),
                                              nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                              nn.Conv3d(embed_dims[0], embed_dims[0], (1, 3, 3), (1, 2, 2), (0, 1, 1)),
                                              nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                              Rearrange('n c d h w -> n d c h w'),
                                              RSTBWithInputConv(
                                                                in_channels=embed_dims[0],
                                                                kernel_size=(1, 3, 3),
                                                                groups=inputconv_groups[0],
                                                                num_blocks=num_blocks[0],
                                                                dim=embed_dims[0],
                                                                input_resolution=[1, img_size[1], img_size[2]],
                                                                depth=depths[0],
                                                                num_heads=num_heads[0],
                                                                window_size=[1, window_size[1], window_size[2]],
                                                                mlp_ratio=mlp_ratio,
                                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                                norm_layer=norm_layer,
                                                                use_checkpoint_attn=[False],
                                                                use_checkpoint_ffn=[False]
                                                               )
                                              )

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        # recurrent feature refinement
        self.backbone = nn.ModuleDict()
        self.deform_align = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            # deformable attention
            self.deform_align[module] = GuidedDeformAttnPack(embed_dims[1],
                                                             embed_dims[1],
                                                             attention_window=attention_window,
                                                             attention_heads=attention_heads,
                                                             deformable_groups=deformable_groups,
                                                             clip_size=clip_size,
                                                             max_residue_magnitude=max_residue_magnitude)

            # feature propagation
            self.backbone[module] = RSTBWithInputConv(
                                                     in_channels=(2 + i) * embed_dims[0],
                                                     kernel_size=(1, 3, 3),
                                                     groups=inputconv_groups[i + 1],
                                                     num_blocks=num_blocks[1],
                                                     dim=embed_dims[1],
                                                     input_resolution=img_size,
                                                     depth=depths[1],
                                                     num_heads=num_heads[1],
                                                     window_size=window_size,
                                                     mlp_ratio=mlp_ratio,
                                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                     norm_layer=norm_layer,
                                                     use_checkpoint_attn=[use_checkpoint_attns[i]],
                                                     use_checkpoint_ffn=[use_checkpoint_ffns[i]]
                                                     )

        # reconstruction
        self.reconstruction = RSTBWithInputConv(
                                               in_channels=5 * embed_dims[0],
                                               kernel_size=(1, 3, 3),
                                               groups=inputconv_groups[5],
                                               num_blocks=num_blocks[2],

                                               dim=embed_dims[2],
                                               input_resolution=[1, img_size[1], img_size[2]],
                                               depth=depths[2],
                                               num_heads=num_heads[2],
                                               window_size=[1, window_size[1], window_size[2]],
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                                               norm_layer=norm_layer,
                                               use_checkpoint_attn=[False],
                                               use_checkpoint_ffn=[False]
                                               )
        self.conv_before_upsampler = nn.Sequential(
                                                  nn.Conv3d(embed_dims[-1], 64, kernel_size=(1, 1, 1),
                                                            padding=(0, 0, 0)),
                                                  nn.LeakyReLU(negative_slope=0.1, inplace=True)
                                                  )
        self.upsampler = Upsample(self.upscale, 64)
        self.conv_last = nn.Conv3d(64, 3, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def propagate(self, feats, flows, module_name, updated_flows=None):
        """Propagate the latent clip features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, clip_size, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
            updated_flows dict(list[tensor]): Each component is a list of updated
                optical flows with shape (n, clip_size, 2, h, w).

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()
        if 'backward' in module_name:
            flow_idx = range(0, t + 1)[::-1]                        # [6, 5, 4, 3, 2, 1, 0]
            clip_idx = range(0, (t + 1) // self.clip_size)[::-1]    # [3, 2, 1, 0]
        else:
            flow_idx = range(-1, t)                                 # [0, 1, 2, 3, 4, 5, 6]
            clip_idx = range(0, (t + 1) // self.clip_size)          # [0, 1, 2, 3]

        if '_1' in module_name:
            updated_flows[f'{module_name}_n1'] = []
            updated_flows[f'{module_name}_n2'] = []

        feat_prop = torch.zeros_like(feats['shallow'][0])
        if self.cpu_cache:
            feat_prop = feat_prop.cuda()

        last_key = list(feats)[-2]
        for i in range(0, len(clip_idx)):
            idx_c = clip_idx[i]
            if i > 0:
                if '_1' in module_name:
                    flow_n01 = flows[:, flow_idx[self.clip_size * i - 1], :, :, :]
                    flow_n12 = flows[:, flow_idx[self.clip_size * i], :, :, :]
                    flow_n23 = flows[:, flow_idx[self.clip_size * i + 1], :, :, :]
                    flow_n02 = flow_n12 + flow_warp(flow_n01, flow_n12.permute(0, 2, 3, 1))
                    flow_n13 = flow_n23 + flow_warp(flow_n12, flow_n23.permute(0, 2, 3, 1))
                    flow_n03 = flow_n23 + flow_warp(flow_n02, flow_n23.permute(0, 2, 3, 1))
                    flow_n1 = torch.stack([flow_n02, flow_n13], 1)
                    flow_n2 = torch.stack([flow_n12, flow_n03], 1)
                    if self.cpu_cache:
                        flow_n1 = flow_n1.cuda()
                        flow_n2 = flow_n2.cuda()
                else:
                    module_name_old = module_name.replace('_2', '_1')
                    flow_n1 = updated_flows[f'{module_name_old}_n1'][i - 1]
                    flow_n2 = updated_flows[f'{module_name_old}_n2'][i - 1]

                if self.cpu_cache:
                    if 'backward' in module_name:
                        feat_q = feats[last_key][idx_c].flip(1).cuda()
                        feat_k = feats[last_key][clip_idx[i - 1]].flip(1).cuda()
                    else:
                        feat_q = feats[last_key][idx_c].cuda()
                        feat_k = feats[last_key][clip_idx[i - 1]].cuda()
                else:
                    if 'backward' in module_name:
                        feat_q = feats[last_key][idx_c].flip(1)
                        feat_k = feats[last_key][clip_idx[i - 1]].flip(1)
                    else:
                        feat_q = feats[last_key][idx_c]
                        feat_k = feats[last_key][clip_idx[i - 1]]

                feat_prop_warped1 = flow_warp(feat_prop.flatten(0, 1),
                                           flow_n1.permute(0, 1, 3, 4, 2).flatten(0, 1))\
                    .view(n, feat_prop.shape[1], feat_prop.shape[2], h, w)
                feat_prop_warped2 = flow_warp(feat_prop.flip(1).flatten(0, 1),
                                           flow_n2.permute(0, 1, 3, 4, 2).flatten(0, 1))\
                    .view(n, feat_prop.shape[1], feat_prop.shape[2], h, w)

                if '_1' in module_name:
                    feat_prop, flow_n1, flow_n2 = self.deform_align[module_name](feat_q, feat_k, feat_prop,
                                                                                 [feat_prop_warped1, feat_prop_warped2],
                                                                                 [flow_n1, flow_n2],
                                                                                 True)
                    updated_flows[f'{module_name}_n1'].append(flow_n1)
                    updated_flows[f'{module_name}_n2'].append(flow_n2)
                else:
                    feat_prop = self.deform_align[module_name](feat_q, feat_k, feat_prop,
                                                               [feat_prop_warped1, feat_prop_warped2],
                                                               [flow_n1, flow_n2],
                                                               False)

            if 'backward' in module_name:
                feat = [feats[k][idx_c].flip(1) for k in feats if k not in [module_name]] + [feat_prop]
            else:
                feat = [feats[k][idx_c] for k in feats if k not in [module_name]] + [feat_prop]

            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat_prop = feat_prop + self.backbone[module_name](torch.cat(feat, dim=2))
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]
            feats[module_name] = [f.flip(1) for f in feats[module_name]]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        feats['shallow'] = torch.cat(feats['shallow'], 1)
        feats['backward_1'] = torch.cat(feats['backward_1'], 1)
        feats['forward_1'] = torch.cat(feats['forward_1'], 1)
        feats['backward_2'] = torch.cat(feats['backward_2'], 1)
        feats['forward_2'] = torch.cat(feats['forward_2'], 1)

        if self.cpu_cache:
            outputs = []
            for i in range(0, feats['shallow'].shape[1]):
                hr = torch.cat([feats[k][:, i:i + 1, :, :, :] for k in feats], dim=2)
                hr = self.reconstruction(hr.cuda())
                hr = self.conv_last(self.upsampler(self.conv_before_upsampler(hr.transpose(1, 2)))).transpose(1, 2)
                hr += torch.nn.functional.interpolate(lqs[:, i:i + 1, :, :, :].cuda(), size=hr.shape[-3:],
                                                      mode='trilinear', align_corners=False)
                hr = hr.cpu()
                outputs.append(hr)
                torch.cuda.empty_cache()

            return torch.cat(outputs, dim=1)

        else:
            hr = torch.cat([feats[k] for k in feats], dim=2)
            hr = self.reconstruction(hr)
            hr = self.conv_last(self.upsampler(self.conv_before_upsampler(hr.transpose(1, 2)))).transpose(1, 2)
            hr += torch.nn.functional.interpolate(lqs, size=hr.shape[-3:], mode='trilinear', align_corners=False)

            return hr

    def forward(self, lqs):
        """Forward function for RVRT.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, _, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        # if self.upscale == 4:
        if self.upscale != 1:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(lqs[:, :, :3, :, :].view(-1, 3, h, w), scale_factor=0.25, mode='bicubic')\
                .view(n, t, 3, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        # shallow feature extractions
        feats = {}
        if self.cpu_cache:
            feats['shallow'] = []
            for i in range(0, t // self.clip_size):
                feat = self.feat_extract(lqs[:, i * self.clip_size:(i + 1) * self.clip_size, :, :, :]).cpu()
                feats['shallow'].append(feat)
            flows_forward, flows_backward = self.compute_flow(lqs_downsample)

            lqs = lqs.cpu()
            lqs_downsample = lqs_downsample.cpu()
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()
            torch.cuda.empty_cache()
        else:
            feats['shallow'] = list(torch.chunk(self.feat_extract(lqs), t // self.clip_size, dim=1))
            flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # recurrent feature refinement
        updated_flows = {}
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                if direction == 'backward':
                    flows = flows_backward
                else:
                    flows = flows_forward if flows_forward is not None else flows_backward.flip(1)

                module_name = f'{direction}_{iter_}'
                feats[module_name] = []
                feats = self.propagate(feats, flows, module_name, updated_flows)

        # reconstruction
        return self.upsample(lqs[:, :, :3, :, :], feats)
    
    # -------------------------------------------
    # test influence, from vrt_model _test_video
    # -------------------------------------------
    # def patch_forward(self, lqs):
    #     """Forward function for RVRT.

    #     Args:
    #         lqs (tensor): Input low quality (LQ) sequence with
    #             shape (n, t, c, h, w).

    #     Returns:
    #         Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
    #     """

    #     n, t, _, h, w = lqs.size()

    #     # whether to cache the features in CPU
    #     self.cpu_cache = True if t > self.cpu_cache_length else False

    #     # if self.upscale == 4:
    #     if self.upscale != 1:
    #         lqs_downsample = lqs.clone()
    #     else:
    #         lqs_downsample = F.interpolate(lqs[:, :, :3, :, :].view(-1, 3, h, w), scale_factor=0.25, mode='bicubic')\
    #             .view(n, t, 3, h // 4, w // 4)

    #     # check whether the input is an extended sequence
    #     self.check_if_mirror_extended(lqs)

    #     # shallow feature extractions
    #     feats = {}
    #     if self.cpu_cache:
    #         feats['shallow'] = []
    #         for i in range(0, t // self.clip_size):
    #             feat = self.feat_extract(lqs[:, i * self.clip_size:(i + 1) * self.clip_size, :, :, :]).cpu()
    #             feats['shallow'].append(feat)
    #         flows_forward, flows_backward = self.compute_flow(lqs_downsample)

    #         lqs = lqs.cpu()
    #         lqs_downsample = lqs_downsample.cpu()
    #         flows_backward = flows_backward.cpu()
    #         flows_forward = flows_forward.cpu()
    #         torch.cuda.empty_cache()
    #     else:
    #         feats['shallow'] = list(torch.chunk(self.feat_extract(lqs), t // self.clip_size, dim=1))
    #         flows_forward, flows_backward = self.compute_flow(lqs_downsample)

    #     # recurrent feature refinement
    #     updated_flows = {}
    #     for iter_ in [1, 2]:
    #         for direction in ['backward', 'forward']:
    #             if direction == 'backward':
    #                 flows = flows_backward
    #             else:
    #                 flows = flows_forward if flows_forward is not None else flows_backward.flip(1)

    #             module_name = f'{direction}_{iter_}'
    #             feats[module_name] = []
    #             feats = self.propagate(feats, flows, module_name, updated_flows)

    #     # reconstruction
    #     return self.upsample(lqs[:, :, :3, :, :], feats)

    # def forward(self, lq):
    #     '''test the video as a whole or as clips (divided temporally). '''
        
    #     num_frame_testing = 0
        
    #     if num_frame_testing:
    #         # test as multiple clips if out-of-memory
    #         sf = 4
    #         # if isinstance(self.opt['scale'], tuple):
    #         #     sf = self.opt['scale'][0]
    #         # else:
    #         #     sf = self.opt['scale']
    #         num_frame_overlapping = 2
    #         not_overlap_border = False
    #         b, d, c, h, w = lq.size()
    #         # c = c - 1 if self.opt['network_g'].get('nonblind_denoising', False) else c
    #         c = c - 1 if False else c
    #         stride = num_frame_testing - num_frame_overlapping
    #         d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
    #         if sf in [2, 3, 4]:
    #             E = torch.zeros(b, d, c, h*sf, w*sf)
    #         else:
    #             E = torch.zeros(b, d, c, h * (int(sf)+1), w * (int(sf)+1))
    #         W = torch.zeros(b, d, 1, 1, 1)

    #         for d_idx in d_idx_list:
    #             lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
    #             out_clip = self._test_clip(lq_clip)
    #             out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

    #             if not_overlap_border:
    #                 if d_idx < d_idx_list[-1]:
    #                     out_clip[:, -num_frame_overlapping//2:, ...] *= 0
    #                     out_clip_mask[:, -num_frame_overlapping//2:, ...] *= 0
    #                 if d_idx > d_idx_list[0]:
    #                     out_clip[:, :num_frame_overlapping//2, ...] *= 0
    #                     out_clip_mask[:, :num_frame_overlapping//2, ...] *= 0

    #             E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
    #             W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)
    #         output = E.div_(W)
    #     else:
    #         # test as one clip (the whole video) if you have enough memory
    #         # window_size = self.opt['network_g'].get('window_size', [6,8,8])
    #         window_size = [2, 8, 8]
    #         d_old = lq.size(1)
    #         d_pad = (d_old // window_size[0] + 1) * window_size[0] - d_old
    #         lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1)
    #         output = self._test_clip(lq)
    #         output = output[:, :d_old, :, :, :]

    #     return output

    # def _test_clip(self, lq):
    #     ''' test the clip as a whole or as patches. '''

    #     # if isinstance(self.opt['scale'], tuple):
    #     #     sf = self.opt['scale'][0]
    #     # else:
    #     #     sf = self.opt['scale']
    #     sf = 4
    #     # window_size = self.opt['network_g'].get('window_size', [6, 8, 8])
    #     window_size = [2, 8, 8]
    #     # size_patch_testing = self.opt['val'].get('size_patch_testing', 0)
    #     size_patch_testing = 128
    #     assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

    #     if size_patch_testing:
    #         # divide the clip to patches (spatially only, tested patch by patch)
    #         # overlap_size = self.opt['val'].get('overlap_size', 20)
    #         overlap_size = 20
    #         not_overlap_border = True

    #         # test patch by patch
    #         b, d, c, h, w = lq.size()
    #         # c = c - 1 if self.opt['network_g'].get('nonblind_denoising', False) else c
    #         c = c - 1 if False else c
    #         stride = size_patch_testing - overlap_size
    #         h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
    #         w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
    #         if sf in [2, 3, 4]:
    #             E = torch.zeros(b, d, c, h*sf, w*sf)
    #         else:
    #             E = torch.zeros(b, d, c, h * (int(sf)+1), w * (int(sf)+1))
    #         W = torch.zeros_like(E)

    #         for h_idx in h_idx_list:
    #             for w_idx in w_idx_list:
    #                 in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
    #                 # if hasattr(self, 'netE'):
    #                 #     out_patch = self.netE(in_patch).detach().cpu()
    #                 # else:
    #                 #     out_patch = self.net_g(in_patch).detach().cpu()     # torch.Size([1, 32, 3, 512, 512])
    #                 out_patch = self.patch_forward(in_patch).detach().cpu()

    #                 out_patch_mask = torch.ones_like(out_patch)

    #                 if not_overlap_border:
    #                     if h_idx < h_idx_list[-1]:
    #                         out_patch[..., -overlap_size//2:, :] *= 0
    #                         out_patch_mask[..., -overlap_size//2:, :] *= 0
    #                     if w_idx < w_idx_list[-1]:
    #                         out_patch[..., :, -overlap_size//2:] *= 0
    #                         out_patch_mask[..., :, -overlap_size//2:] *= 0
    #                     if h_idx > h_idx_list[0]:
    #                         out_patch[..., :overlap_size//2, :] *= 0
    #                         out_patch_mask[..., :overlap_size//2, :] *= 0
    #                     if w_idx > w_idx_list[0]:
    #                         out_patch[..., :, :overlap_size//2] *= 0
    #                         out_patch_mask[..., :, :overlap_size//2] *= 0
    #                 if sf in [2, 3, 4]:
    #                     E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch)
    #                     W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch_mask)
    #                 else:
    #                     E[..., h_idx * (int(sf)+1):(h_idx+size_patch_testing) * (int(sf)+1), 
    #                            w_idx * (int(sf)+1):(w_idx+size_patch_testing) * (int(sf)+1)].add_(out_patch)
    #                     W[..., h_idx * (int(sf)+1):(h_idx+size_patch_testing) * (int(sf)+1), 
    #                            w_idx * (int(sf)+1):(w_idx+size_patch_testing) * (int(sf)+1)].add_(out_patch_mask)
    #         output = E.div_(W)

    #     else:
    #         _, _, _, h_old, w_old = lq.size()
    #         # h_pad = (h_old// window_size[1]+1)*window_size[1] - h_old   # (144 // 8 + 1) * 8 - 144 = 8
    #         # w_pad = (w_old// window_size[2]+1)*window_size[2] - w_old   # (180 // 8 + 1) * 8 - 180 = 4
    #         # ref RVRT test
    #         h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]      # (8 - 144 % 8) % 8 = 0
    #         w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]      # (8 - 180 % 8) % 8 = 4

    #         lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
    #         lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq
            
    #         output = self.net_g(lq)
    #         # for arbitrary-scale VSR, use BI post-process
    #         if sf in [2, 3, 4]:
    #             output = output[:, :, :, :h_old*sf, :w_old*sf]
    #         else:
    #             output = output[:, :, :, :h_old * (int(sf)+1), :w_old * (int(sf)+1)]
                

    #     return output
        


if __name__ == '__main__':
    from torch.profiler import profile, record_function, ProfilerActivity
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scale = (4, 4)
    model = RVRT(upscale=4, 
                 clip_size=2, 
                 img_size=[2, 64, 64], 
                 window_size=[2, 8, 8], 
                 num_blocks=[1, 2, 1],
                 depths=[2, 2, 2], 
                 embed_dims=[144, 144, 144], 
                 num_heads=[6, 6, 6],
                 inputconv_groups=[1, 1, 1, 1, 1, 1], 
                 spynet_path=None,
                 deformable_groups=12, 
                 attention_heads=12,
                 attention_window=[3, 3], 
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 no_checkpoint_attn_blocks=[],
                 no_checkpoint_ffn_blocks=[],
                 cpu_cache_length=100).to(device)
    model.eval()
    
    # input = torch.rand(1, 7, 3, 180, 320).to(device)    # the number of frame should be even偶数
    input = torch.rand(1, 8, 3, 64, 64).to(device)
    
    # ------ torch profile -------------------------
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("model_inference"):
            out = model(input)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # ------ Runtime ------------------------------
    VSR_runtime_test(model, input, scale)

    # ------ Parameter ----------------------------
    print(
        "Model have {:.3f}M parameters in total".format(sum(x.numel() for x in model.parameters()) / 1000000.0))
    
    # ------ FLOPs --------------------------------
    # get_flops(net, [5, 3, 180, 320])
    with torch.no_grad():
        print('Input:', input.shape)
        print(flop_count_table(FlopCountAnalysis(model, input), activations=ActivationCountAnalysis(model, input)))
        out = model(input)
        print('Output:', out.shape)

"""
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        model_inference        71.84%       16.007s       100.00%       22.282s       22.282s       0.000us         0.00%        5.200s        5.200s      84.37 Mb    -372.60 Mb      48.00 Mb    -461.92 Gb             1  
                                     DeformAttnFunction         0.10%      21.985ms         1.11%     246.621ms       3.425ms     209.204ms         4.02%        1.784s      24.775ms           0 b           0 b       2.53 Gb    -238.18 Gb            72  
                                            aten::copy_         0.20%      43.815ms         8.04%        1.792s     305.445us        1.557s        29.94%        1.557s     265.375us           0 b           0 b           0 b           0 b          5868  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us        1.519s        29.20%        1.519s     308.176us           0 b           0 b           0 b           0 b          4928  
                                            aten::clone         0.12%      27.035ms         1.73%     385.875ms      99.095us       0.000us         0.00%        1.438s     369.187us           0 b           0 b     272.59 Gb           0 b          3894  
                                          aten::flatten         0.01%       2.821ms         0.14%      30.972ms      47.796us       0.000us         0.00%        1.126s       1.738ms           0 b           0 b     182.25 Gb           0 b           648  
                                      aten::convolution         0.03%       6.202ms         6.93%        1.544s       1.660ms       0.000us         0.00%     921.644ms     991.015us           0 b           0 b      32.86 Gb           0 b           930  
                                     aten::_convolution         0.07%      15.090ms         6.90%        1.538s       1.654ms       0.000us         0.00%     921.644ms     991.015us           0 b           0 b      32.86 Gb           0 b           930  
                                           aten::matmul         0.18%      40.073ms         3.19%     710.542ms     210.718us       0.000us         0.00%     912.255ms     270.538us           0 b           0 b     161.41 Gb      -2.53 Gb          3372  
                                           aten::conv3d         0.01%       2.657ms         6.51%        1.450s       2.545ms       0.000us         0.00%     868.015ms       1.523ms           0 b           0 b      31.86 Gb           0 b           570  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 22.282s
Self CUDA time total: 5.200s

Warm up ...

Testing ...

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [26:07<00:00,  5.22s/it]

Average Runtime: 5222.508 ms

Model have 10.782M parameters in total
Input: torch.Size([1, 7, 3, 180, 320])
| module                                           | #parameters or shape   | #flops     | #activations   |
|:-------------------------------------------------|:-----------------------|:-----------|:---------------|
| model                                            | 10.782M                | 8.697T     | 38.142G        |
|  spynet.basic_module                             |  1.44M                 |  0.44T     |  0.268G        |
|   spynet.basic_module.0.basic_module             |   0.24M                |   0.322G   |   0.196M       |
|    spynet.basic_module.0.basic_module.0          |    12.576K             |    16.859M |    43.008K     |
|    spynet.basic_module.0.basic_module.2          |    0.1M                |    0.135G  |    86.016K     |
|    spynet.basic_module.0.basic_module.4          |    0.1M                |    0.135G  |    43.008K     |
|    spynet.basic_module.0.basic_module.6          |    25.104K             |    33.718M |    21.504K     |
|    spynet.basic_module.0.basic_module.8          |    1.57K               |    2.107M  |    2.688K      |
|   spynet.basic_module.1.basic_module             |   0.24M                |   1.29G    |   0.785M       |
|    spynet.basic_module.1.basic_module.0          |    12.576K             |    67.437M |    0.172M      |
|    spynet.basic_module.1.basic_module.2          |    0.1M                |    0.539G  |    0.344M      |
|    spynet.basic_module.1.basic_module.4          |    0.1M                |    0.539G  |    0.172M      |
|    spynet.basic_module.1.basic_module.6          |    25.104K             |    0.135G  |    86.016K     |
|    spynet.basic_module.1.basic_module.8          |    1.57K               |    8.43M   |    10.752K     |
|   spynet.basic_module.2.basic_module             |   0.24M                |   5.159G   |   3.14M        |
|    spynet.basic_module.2.basic_module.0          |    12.576K             |    0.27G   |    0.688M      |
|    spynet.basic_module.2.basic_module.2          |    0.1M                |    2.158G  |    1.376M      |
|    spynet.basic_module.2.basic_module.4          |    0.1M                |    2.158G  |    0.688M      |
|    spynet.basic_module.2.basic_module.6          |    25.104K             |    0.539G  |    0.344M      |
|    spynet.basic_module.2.basic_module.8          |    1.57K               |    33.718M |    43.008K     |
|   spynet.basic_module.3.basic_module             |   0.24M                |   20.636G  |   12.558M      |
|    spynet.basic_module.3.basic_module.0          |    12.576K             |    1.079G  |    2.753M      |
|    spynet.basic_module.3.basic_module.2          |    0.1M                |    8.632G  |    5.505M      |
|    spynet.basic_module.3.basic_module.4          |    0.1M                |    8.632G  |    2.753M      |
|    spynet.basic_module.3.basic_module.6          |    25.104K             |    2.158G  |    1.376M      |
|    spynet.basic_module.3.basic_module.8          |    1.57K               |    0.135G  |    0.172M      |
|   spynet.basic_module.4.basic_module             |   0.24M                |   82.542G  |   50.233M      |
|    spynet.basic_module.4.basic_module.0          |    12.576K             |    4.316G  |    11.01M      |
|    spynet.basic_module.4.basic_module.2          |    0.1M                |    34.528G |    22.02M      |
|    spynet.basic_module.4.basic_module.4          |    0.1M                |    34.528G |    11.01M      |
|    spynet.basic_module.4.basic_module.6          |    25.104K             |    8.632G  |    5.505M      |
|    spynet.basic_module.4.basic_module.8          |    1.57K               |    0.539G  |    0.688M      |
|   spynet.basic_module.5.basic_module             |   0.24M                |   0.33T    |   0.201G       |
|    spynet.basic_module.5.basic_module.0          |    12.576K             |    17.264G |    44.04M      |
|    spynet.basic_module.5.basic_module.2          |    0.1M                |    0.138T  |    88.08M      |
|    spynet.basic_module.5.basic_module.4          |    0.1M                |    0.138T  |    44.04M      |
|    spynet.basic_module.5.basic_module.6          |    25.104K             |    34.528G |    22.02M      |
|    spynet.basic_module.5.basic_module.8          |    1.57K               |    2.158G  |    2.753M      |
|  feat_extract.main                               |  0.363M                |  0.313T    |  2.642G        |
|   feat_extract.main.1                            |   4.032K               |   3.058G   |   0.113G       |
|    feat_extract.main.1.weight                    |    (144, 3, 1, 3, 3)   |            |                |
|    feat_extract.main.1.bias                      |    (144,)              |            |                |
|   feat_extract.main.3                            |   0.288K               |   0.566G   |   0            |
|    feat_extract.main.3.weight                    |    (144,)              |            |                |
|    feat_extract.main.3.bias                      |    (144,)              |            |                |
|   feat_extract.main.5.0                          |   0.359M               |   0.308T   |   2.529G       |
|    feat_extract.main.5.0.residual_group.blocks   |    0.338M              |    0.292T  |    2.416G      |
|    feat_extract.main.5.0.linear                  |    20.88K              |    16.307G |    0.113G      |
|   feat_extract.main.7                            |   0.288K               |   0.566G   |   0            |
|    feat_extract.main.7.weight                    |    (144,)              |            |                |
|    feat_extract.main.7.bias                      |    (144,)              |            |                |
|  backbone                                        |  5.527M                |  4.759T    |  25.518G       |
|   backbone.backward_1.main                       |   1.102M               |   0.97T    |   6.38G        |
|    backbone.backward_1.main.1                    |    0.373M              |    0.294T  |    0.113G      |
|    backbone.backward_1.main.3                    |    0.288K              |    0.566G  |    0           |
|    backbone.backward_1.main.5                    |    0.728M              |    0.675T  |    6.266G      |
|    backbone.backward_1.main.7                    |    0.288K              |    0.566G  |    0           |
|   backbone.forward_1.main                        |   1.288M               |   1.116T   |   6.38G        |
|    backbone.forward_1.main.1                     |    0.56M               |    0.44T   |    0.113G      |
|    backbone.forward_1.main.3                     |    0.288K              |    0.566G  |    0           |
|    backbone.forward_1.main.5                     |    0.728M              |    0.675T  |    6.266G      |
|    backbone.forward_1.main.7                     |    0.288K              |    0.566G  |    0           |
|   backbone.backward_2.main                       |   1.475M               |   1.263T   |   6.38G        |
|    backbone.backward_2.main.1                    |    0.747M              |    0.587T  |    0.113G      |
|    backbone.backward_2.main.3                    |    0.288K              |    0.566G  |    0           |
|    backbone.backward_2.main.5                    |    0.728M              |    0.675T  |    6.266G      |
|    backbone.backward_2.main.7                    |    0.288K              |    0.566G  |    0           |
|   backbone.forward_2.main                        |   1.662M               |   1.41T    |   6.38G        |
|    backbone.forward_2.main.1                     |    0.933M              |    0.734T  |    0.113G      |
|    backbone.forward_2.main.3                     |    0.288K              |    0.566G  |    0           |
|    backbone.forward_2.main.5                     |    0.728M              |    0.675T  |    6.266G      |
|    backbone.forward_2.main.7                     |    0.288K              |    0.566G  |    0           |
|  deform_align                                    |  1.816M                |  1.066T    |  5.172G        |
|   deform_align.backward_1                        |   0.454M               |   0.267T   |   1.293G       |
|    deform_align.backward_1.proj_q.1              |    41.76K              |    24.461G |    0.17G       |
|    deform_align.backward_1.proj_k.1              |    41.76K              |    24.461G |    0.17G       |
|    deform_align.backward_1.proj_v.1              |    41.76K              |    24.461G |    0.17G       |
|    deform_align.backward_1.mlp.1                 |    83.376K             |    48.922G |    0.255G      |
|    deform_align.backward_1.conv_offset           |    0.204M              |    0.12T   |    0.444G      |
|    deform_align.backward_1.proj.1                |    41.616K             |    24.461G |    84.935M     |
|   deform_align.forward_1                         |   0.454M               |   0.267T   |   1.293G       |
|    deform_align.forward_1.proj_q.1               |    41.76K              |    24.461G |    0.17G       |
|    deform_align.forward_1.proj_k.1               |    41.76K              |    24.461G |    0.17G       |
|    deform_align.forward_1.proj_v.1               |    41.76K              |    24.461G |    0.17G       |
|    deform_align.forward_1.mlp.1                  |    83.376K             |    48.922G |    0.255G      |
|    deform_align.forward_1.conv_offset            |    0.204M              |    0.12T   |    0.444G      |
|    deform_align.forward_1.proj.1                 |    41.616K             |    24.461G |    84.935M     |
|   deform_align.backward_2                        |   0.454M               |   0.267T   |   1.293G       |
|    deform_align.backward_2.proj_q.1              |    41.76K              |    24.461G |    0.17G       |
|    deform_align.backward_2.proj_k.1              |    41.76K              |    24.461G |    0.17G       |
|    deform_align.backward_2.proj_v.1              |    41.76K              |    24.461G |    0.17G       |
|    deform_align.backward_2.mlp.1                 |    83.376K             |    48.922G |    0.255G      |
|    deform_align.backward_2.conv_offset           |    0.204M              |    0.12T   |    0.444G      |
|    deform_align.backward_2.proj.1                |    41.616K             |    24.461G |    84.935M     |
|   deform_align.forward_2                         |   0.454M               |   0.267T   |   1.293G       |
|    deform_align.forward_2.proj_q.1               |    41.76K              |    24.461G |    0.17G       |
|    deform_align.forward_2.proj_k.1               |    41.76K              |    24.461G |    0.17G       |
|    deform_align.forward_2.proj_v.1               |    41.76K              |    24.461G |    0.17G       |
|    deform_align.forward_2.mlp.1                  |    83.376K             |    48.922G |    0.255G      |
|    deform_align.forward_2.conv_offset            |    0.204M              |    0.12T   |    0.444G      |
|    deform_align.forward_2.proj.1                 |    41.616K             |    24.461G |    84.935M     |
|  reconstruction.main                             |  1.292M                |  1.043T    |  2.642G        |
|   reconstruction.main.1                          |   0.933M               |   0.734T   |   0.113G       |
|    reconstruction.main.1.weight                  |    (144, 720, 1, 3, 3) |            |                |
|    reconstruction.main.1.bias                    |    (144,)              |            |                |
|   reconstruction.main.3                          |   0.288K               |   0.566G   |   0            |
|    reconstruction.main.3.weight                  |    (144,)              |            |                |
|    reconstruction.main.3.bias                    |    (144,)              |            |                |
|   reconstruction.main.5.0                        |   0.359M               |   0.308T   |   2.529G       |
|    reconstruction.main.5.0.residual_group.blocks |    0.338M              |    0.292T  |    2.416G      |
|    reconstruction.main.5.0.linear                |    20.88K              |    16.307G |    0.113G      |
|   reconstruction.main.7                          |   0.288K               |   0.566G   |   0            |
|    reconstruction.main.7.weight                  |    (144,)              |            |                |
|    reconstruction.main.7.bias                    |    (144,)              |            |                |
|  conv_before_upsampler.0                         |  9.28K                 |  7.248G    |  50.332M       |
|   conv_before_upsampler.0.weight                 |   (64, 144, 1, 1, 1)   |            |                |
|   conv_before_upsampler.0.bias                   |   (64,)                |            |                |
|  upsampler                                       |  0.332M                |  1.044T    |  1.812G        |
|   upsampler.0                                    |   0.148M               |   0.116T   |   0.201G       |
|    upsampler.0.weight                            |    (256, 64, 1, 3, 3)  |            |                |
|    upsampler.0.bias                              |    (256,)              |            |                |
|   upsampler.5                                    |   0.148M               |   0.464T   |   0.805G       |
|    upsampler.5.weight                            |    (256, 64, 1, 3, 3)  |            |                |
|    upsampler.5.bias                              |    (256,)              |            |                |
|   upsampler.10                                   |   36.928K              |   0.464T   |   0.805G       |
|    upsampler.10.weight                           |    (64, 64, 1, 3, 3)   |            |                |
|    upsampler.10.bias                             |    (64,)               |            |                |
|  conv_last                                       |  1.731K                |  21.743G   |  37.749M       |
|   conv_last.weight                               |   (3, 64, 1, 3, 3)     |            |                |
|   conv_last.bias                                 |   (3,)                 |            |                |
Output: torch.Size([1, 7, 3, 720, 1280])
"""
