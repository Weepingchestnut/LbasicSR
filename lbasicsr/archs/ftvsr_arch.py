import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmengine.runner import load_checkpoint

from lbasicsr.archs.arch_util import make_layer, ResidualBlockNoBN, PixelShufflePack
from lbasicsr.utils.logger import get_root_logger
from lbasicsr.utils.registry import ARCH_REGISTRY


class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, num_feat=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


class LTAM(nn.Module):
    def __init__(self, stride: int = 4):
        super().__init__()

        self.stride = stride
        self.fusion = nn.Conv2d(3 * 64, 64, 3, 1, 1, bias=True)

    def forward(self, curr_feat, index_feat_set_s1, anchor_feat, sparse_feat_set_s1, sparse_feat_set_s2,
                sparse_feat_set_s3, location_feat):
        """
        input :   anchor_feat  # n * c * h * w
        input :   sparse_feat_set_s1      # n * t * (c*4*4) * (h//4) * (w//4)
        input :   sparse_feat_set_s2      # n * t * (c*4*4) * (h//4) * (w//4)
        input :   sparse_feat_set_s3      # n * t * (c*4*4) * (h//4) * (w//4)
        input :   location_feat  #  n * (2*t) * (h//4) * (w//4)
        output :   fusion_feature  # n * c * h * w
        """
        n, c, h, w = anchor_feat.size()
        t = sparse_feat_set_s1.size(1)
        feat_len = int(c * self.stride * self.stride)
        feat_num = int((h // self.stride) * (w // self.stride))

        # grid_flow [0,h-1][0,w-1] -> [-1,1][-1,1]
        grid_flow = location_feat.contiguous().view(n, t, 2, h // self.stride, w // self.stride).permute(0, 1, 3, 4, 2)
        grid_flow_x = 2.0 * grid_flow[:, :, :, :, 0] / max(w // self.stride - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, :, 1] / max(h // self.stride - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=4)

        output_s1 = F.grid_sample(
            sparse_feat_set_s1.contiguous().view(-1, (c * self.stride * self.stride), (h // self.stride),
                                                 (w // self.stride)),
            grid_flow.contiguous().view(-1, (h // self.stride), (w // self.stride), 2), mode='nearest',
            padding_mode='zeros', align_corners=True)  # (nt) * (c*4*4) * (h//4) * (w//4)
        output_s2 = F.grid_sample(
            sparse_feat_set_s2.contiguous().view(-1, (c * self.stride * self.stride), (h // self.stride),
                                                 (w // self.stride)),
            grid_flow.contiguous().view(-1, (h // self.stride), (w // self.stride), 2), mode='nearest',
            padding_mode='zeros', align_corners=True)  # (nt) * (c*4*4) * (h//4) * (w//4)
        output_s3 = F.grid_sample(
            sparse_feat_set_s3.contiguous().view(-1, (c * self.stride * self.stride), (h // self.stride),
                                                 (w // self.stride)),
            grid_flow.contiguous().view(-1, (h // self.stride), (w // self.stride), 2), mode='nearest',
            padding_mode='zeros', align_corners=True)  # (nt) * (c*4*4) * (h//4) * (w//4)

        index_output_s1 = F.grid_sample(
            index_feat_set_s1.contiguous().view(-1, (c * self.stride * self.stride), (h // self.stride),
                                                (w // self.stride)),
            grid_flow.contiguous().view(-1, (h // self.stride), (w // self.stride), 2), mode='nearest',
            padding_mode='zeros', align_corners=True)  # (nt) * (c*4*4) * (h//4) * (w//4)

        # n * c * h * w --> # n * (c*4*4) * (h//4*w//4)
        curr_feat = F.unfold(curr_feat, kernel_size=(self.stride, self.stride), padding=0, stride=self.stride)
        # n * (c*4*4) * (h//4*w//4) --> n * (h//4*w//4) * (c*4*4)
        curr_feat = curr_feat.permute(0, 2, 1)
        curr_feat = F.normalize(curr_feat, dim=2).unsqueeze(3)  # n * (h//4*w//4) * (c*4*4) * 1

        # cross-scale attention * 4
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        index_output_s1 = index_output_s1.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
                                                            (w // self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        index_output_s1 = F.unfold(index_output_s1, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len,
                                                                                                  feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * (h//4*w//4) * t * (c*4*4)
        index_output_s1 = index_output_s1.permute(0, 3, 1, 2)
        index_output_s1 = F.normalize(index_output_s1, dim=3)  # n * (h//4*w//4) * t * (c*4*4)
        # [ n * (h//4*w//4) * t * (c*4*4) ]  *  [ n * (h//4*w//4) * (c*4*4) * 1 ]  -->  n * (h//4*w//4) * t
        matrix_index = torch.matmul(index_output_s1, curr_feat).squeeze(3)  # n * (h//4*w//4) * t
        matrix_index = matrix_index.view(n, feat_num, t)  # n * (h//4*w//4) * t
        corr_soft, corr_index = torch.max(matrix_index, dim=2)  # n * (h//4*w//4)
        # n * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        corr_soft = corr_soft.unsqueeze(1).expand(-1, feat_len, -1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        corr_soft = F.fold(corr_soft, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
                           stride=self.stride)

        # Aggr
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        output_s1 = output_s1.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
                                                (w // self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s1 = F.unfold(output_s1, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len, feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
        output_s1 = torch.gather(output_s1.contiguous().view(n, t, feat_len, feat_num), 1,
                                 corr_index.view(n, 1, 1, feat_num).expand(-1, -1, feat_len,
                                                                           -1))  # n * 1 * (c*4*4) * (h//4*w//4)
        # n * 1 * (c*4*4) * (h//4*w//4)  --> n * (c*4*4) * (h//4*w//4)
        output_s1 = output_s1.squeeze(1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s1 = F.fold(output_s1, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
                           stride=self.stride)

        # Aggr
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        output_s2 = output_s2.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
                                                (w // self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s2 = F.unfold(output_s2, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len, feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
        output_s2 = torch.gather(output_s2.contiguous().view(n, t, feat_len, feat_num), 1,
                                 corr_index.view(n, 1, 1, feat_num).expand(-1, -1, feat_len,
                                                                           -1))  # n * 1 * (c*4*4) * (h//4*w//4)
        # n * 1 * (c*4*4) * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        output_s2 = output_s2.squeeze(1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s2 = F.fold(output_s2, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
                           stride=self.stride)

        # Aggr
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        output_s3 = output_s3.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
                                                (w // self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s3 = F.unfold(output_s3, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len, feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
        output_s3 = torch.gather(output_s3.contiguous().view(n, t, feat_len, feat_num), 1,
                                 corr_index.view(n, 1, 1, feat_num).expand(-1, -1, feat_len,
                                                                           -1))  # n * 1 * (c*4*4) * (h//4*w//4)
        # n * 1 * (c*4*4) * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        output_s3 = output_s3.squeeze(1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s3 = F.fold(output_s3, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
                           stride=self.stride)

        out = torch.cat([output_s1, output_s2, output_s3], dim=1)
        out = self.fusion(out)
        out = out * corr_soft
        out += anchor_feat

        return out


def build_filter(pos, freq, POS):
    result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)


def generate_dct_matrix(h=8, w=8):
    matrix = np.zeros((h, w, h, w))
    
    us = list(range(h))
    vs = list(range(w))
    

    
    for u in range(h):
        for v in range(w):
            for i in range(h):
                for j in range(w):
                    matrix[u, v, i, j] = build_filter(i, u, h) * build_filter(j, v, w)

    matrix = matrix.reshape(-1, h, w)
    
    return matrix


class dct_layer(nn.Module):
    def __init__(self, in_c=3, h=8, w=8):
        super(dct_layer, self).__init__()
        assert h == w

        self.dct_conv = nn.Conv2d(in_c, in_c*h*w, h, h, bias=False, groups=in_c) 
        matrix = generate_dct_matrix(h=h, w=w)
        self.weight = torch.from_numpy(matrix).float().unsqueeze(1) #64,1,8,8
        
        self.dct_conv.weight.data = torch.cat([self.weight] * in_c, dim=0) #192,1,8,8
        self.dct_conv.weight.requires_grad = False

    def forward(self, x):
        dct = self.dct_conv(x)

        return dct


class reverse_dct_layer(nn.Module):
    def __init__(self, out_c=3, h=8, w=8):
        super(reverse_dct_layer, self).__init__()

        assert h == w

        self.reverse_dct_conv = nn.ConvTranspose2d(out_c * h * w, out_c, h, h, bias=False, groups=out_c)
        matrix = generate_dct_matrix(h=h, w=w)
        self.weight = torch.from_numpy(matrix).float().unsqueeze(1) #64,1,8,8

        self.reverse_dct_conv.weight.data = torch.cat([self.weight] * out_c, dim=0) #192,1,8,8
        self.reverse_dct_conv.weight.requires_grad = False

    def forward(self, x):
        rdct = self.reverse_dct_conv(x)

        return rdct


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.
    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.
    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


def remove_image_padding(imgs, padding_h, padding_w):
    n, c, h, w = imgs.shape

    new_imgs = imgs[:, :, :h-padding_h, :w-padding_w]

    return new_imgs


class FTT(nn.Module):
    """
    SR_imgs, (n, t, c, h, w)
    high_frequency_imgs, (n, t, c, h, w)
    flows, (n, t-1, 2, h, w)

    """

    def __init__(self, dct_kernel=(8,8), d_model=512, n_heads=8):
        super().__init__()
        self.dct_kernel = dct_kernel
        self.dct = dct_layer(in_c=3, h=dct_kernel[0], w=dct_kernel[1])
        self.rdct = reverse_dct_layer(out_c=3, h=dct_kernel[0], w=dct_kernel[1])

        self.conv_layer1 = nn.Conv2d(192, 512, 1, 1, 0, bias=True)
        self.feat_extractor = ResidualBlocksWithInputConv(512, 512, 3)
        self.resblocks = ResidualBlocksWithInputConv(512*2, 512, 3)
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * 512, 512, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0, bias=True))
        
        self.ftta = FTTA_layer(channel=512, d_model=d_model, n_heads=n_heads)

        self.conv_layer2 = nn.Conv2d(512, 192, 1, 1, 0, bias=True)

    def forward(self, bicubic_imgs, high_frequency_imgs, flows, padiings, to_cpu=False):
        n,t,c,h,w = bicubic_imgs.shape
        padding_h, padding_w = padiings
        flows_forward, flows_backward = flows
        #resize flows
        if flows_forward is not None:
            flows_forward = resize_flow(flows_forward.view(-1, 2, h, w), size_type='shape', sizes=(h//self.dct_kernel[0], w//self.dct_kernel[1]))
            flows_forward = flows_forward.view(n, t-1, 2, h//self.dct_kernel[0], w//self.dct_kernel[1])
        flows_backward = resize_flow(flows_backward.view(-1, 2, h, w), size_type='shape', sizes=(h//self.dct_kernel[0], w//self.dct_kernel[1]))
        flows_backward = flows_backward.view(n, t-1, 2, h//self.dct_kernel[1], w//self.dct_kernel[1])

        #to frequency domain
        dct_bic_0 = self.dct(bicubic_imgs.view(-1, c, h, w))
        dct_bic = F.normalize(dct_bic_0.view(n*t, c*8*8, -1), dim=2).view(n*t, -1, h//8, w//8)
        
        dct_hfi_0 = self.dct(high_frequency_imgs.view(-1, c, h, w))
        dct_hfi = F.normalize(dct_hfi_0.view(n*t, c*8*8, -1), dim=2).view(n*t, -1, h//8, w//8)
        dct_hfi_0 = dct_hfi_0.view(n, t, -1, h//self.dct_kernel[0], w//self.dct_kernel[1])


        dct_bic_fea = self.feat_extractor(self.conv_layer1(dct_bic)).view(n, t, 512, h//self.dct_kernel[0], w//self.dct_kernel[1])
        dct_hfi_fea = self.feat_extractor(self.conv_layer1(dct_hfi)).view(n, t, 512, h//self.dct_kernel[0], w//self.dct_kernel[1])

        n,t,c,h,w = dct_hfi_fea.shape


        hfi_backward_list = []
        hfi_prop = dct_hfi.new_zeros(n, c, h, w)
        #backward
        for i in range(t-1, -1, -1):
            bic =  dct_bic_fea[:, i, :, :, :]
            hfi = dct_hfi_fea[:, i, :, :, :]
            if i < t-1:
                flow = flows_backward[:, i, :, :, :]
                hfi_prop = flow_warp(hfi_prop, flow.permute(0, 2, 3, 1), padding_mode='border')

                hfi_ = self.ftta(bic, hfi, hfi)
                hfi_prop = self.ftta(hfi_, hfi_prop, hfi_prop)

            hfi_prop = torch.cat([hfi, hfi_prop], dim=1)
            hfi_prop = self.resblocks(hfi_prop)
            hfi_backward_list.append(hfi_prop) #(b,c,h,w)
        #forward
        out_fea = hfi_backward_list[::-1]

        final_out = []
        hfi_prop = torch.zeros_like(hfi_prop)
        for i in range(t):
            bic =  dct_bic_fea[:, i, :, :, :]
            hfi = dct_hfi_fea[:, i, :, :, :]
            if i > 0:
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                # flow = flows_forward[:, i-1, :, :, :]
                hfi_prop = flow_warp(hfi_prop, flow.permute(0, 2, 3, 1), padding_mode='border')

                # hfi_prop = self.ftta(bic, hfi, hfi_prop)
                hfi_ = self.ftta(bic, hfi, hfi)
                hfi_prop = self.ftta(hfi_, hfi_prop, hfi_prop)
                
            hfi_prop = torch.cat([hfi, hfi_prop], dim=1)
            hfi_prop = self.resblocks(hfi_prop)
            
            out = torch.cat([out_fea[i], hfi, hfi_prop], dim=1)
            out = self.conv_layer2(self.fusion(out)) + dct_hfi_0[:, i, :, :, :]
            out = self.rdct(out) + high_frequency_imgs[:, i, :, :, :]

            out = remove_image_padding(out, padding_h, padding_w)
            if to_cpu: 
                final_out.append(out.cpu())
            else: 
                final_out.append(out)
        return torch.stack(final_out, dim=1)

class FTT_encoder(nn.Module):
    def __init__(self, channel=192, d_model=512, n_heads=8, num_layer=3):
        super().__init__()
        self.num_layer = num_layer

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(
                FTTA_layer(channel, d_model, n_heads)
            )
    def forward(self, q, k, v):
        v = self.layers[0](q, k, v)
        for i in range(1, self.num_layer):
            v = self.layers[i](k, v, v)
        return v



class FTTA_layer(nn.Module):

    def __init__(self, channel=192, d_model=512, n_heads=8, patch_k=(8,8), patch_stride=8):
        super().__init__()
        self.patch_k = patch_k
        self.patch_stride = patch_stride
        inplances = (channel // 64) * patch_k[0] * patch_k[1]


        self.layer_q = nn.Linear(inplances, d_model)
        self.layer_k = nn.Linear(inplances, d_model)
        self.layer_v = nn.Linear(inplances, d_model)

        self.MultiheadAttention = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_model)
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(d_model, inplances)

    def forward_ffn(self, x):
        x2 = self.linear1(x)
        x2= self.activation(x2)
        x = x2 + x
        x = self.norm2(x)

        x = self.linear2(x)

        return x


    def forward(self, q, k, v):
        '''
        q, k, v, (n, 512, h, w)
        frequency attention
        '''
        
        N,C,H,W = q.shape

        qs = q.view(N*64, -1, H, W)
        ks = k.view(N*64, -1, H, W)
        vs = v.view(N*64, -1, H, W)

        qs = torch.nn.functional.unfold(qs, self.patch_k , dilation=1, padding=0, stride=self.patch_stride) #(N*64, 3*8*8, num)
        ks = torch.nn.functional.unfold(ks, self.patch_k , dilation=1, padding=0, stride=self.patch_stride) #(N*64, 3*8*8, num)
        vs = torch.nn.functional.unfold(vs, self.patch_k , dilation=1, padding=0, stride=self.patch_stride) #(N*64, 3*8*8, num)

        BF, D, num = qs.shape
        qs = qs.view(N, 64, D, num).permute(0, 1, 3, 2).reshape(N, -1, D) #(Batch, F*num, dim=3*8*8)
        ks = ks.view(N, 64, D, num).permute(0, 1, 3, 2).reshape(N, -1, D)
        vs = vs.view(N, 64, D, num).permute(0, 1, 3, 2).reshape(N, -1, D)

        qs = self.layer_q(qs) #(batch, F*num, d_model)
        ks = self.layer_k(ks)
        vs = self.layer_v(vs)

        qs = qs.permute(1, 0, 2) #L,N,E
        ks = ks.permute(1, 0, 2)
        vs = vs.permute(1, 0, 2)

        ttn_output, attn_output_weights  = self.MultiheadAttention(qs, ks, vs)
        out = ttn_output + vs
        out = self.norm1(out) #LNE

        out = out.permute(1, 0, 2) #NLE, (batch, F*num, dim=d_model)

        out = self.forward_ffn(out) #N,L,E,
        out = out.view(N, 64, num, D).permute(0, 1, 3, 2).reshape(-1, D, num) #(batch*64, 3*8*8, num)
        out = torch.nn.functional.fold(out, (H,W), self.patch_k, dilation=1, padding=0, stride=self.patch_stride) #(batch*64, 3, H, W)
        out = out.view(N, -1, H, W)

        return out


def check_and_padding_imgs(imgs, dct_kernel=(8,8)):
    n,t,c,h,w = imgs.size()

    if h % dct_kernel[0] != 0:
        k_t = h // dct_kernel[0]
        new_h = (k_t + 1) * dct_kernel[0]
    else:
        new_h = h
    
    if w % dct_kernel[1] != 0:
        k_t = w // dct_kernel[1]
        new_w = (k_t + 1) * dct_kernel[1]
    else:
        new_w = w 
    
    new_imgs = imgs.new_zeros(n, t, c, new_h, new_w)
    padding_h = new_h - h
    padding_w = new_w - w 

    new_imgs[:, :, :, :h, :w] = imgs
    new_imgs[:, :, :, -padding_h:, -padding_w:] = imgs[:, :, :, -padding_h:, -padding_w:]

    return new_imgs, padding_h, padding_w 


@ARCH_REGISTRY.register()
class FTVSR(nn.Module):
    """BasicVSR network structure for video super-resolution.

    Frequency-temporal transformer

    Support only x4 upsampling.
    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 30.
        keyframe_stride (int): Number determining the keyframes. If stride=5,
            then the (0, 5, 10, 15, ...)-th frame will be the keyframes.
            Default: 5.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self, 
                 mid_channels=64, 
                 num_blocks=60, 
                 stride=4, 
                 keyframe_stride=3, 
                 spynet_pretrained=None, 
                 dct_kernel=(8,8), 
                 d_model=512, 
                 n_heads=8):

        super().__init__()

        self.dct_kernel = dct_kernel
        self.mid_channels = mid_channels
        self.keyframe_stride = keyframe_stride
        self.stride = stride
        
        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)
        self.feat_extractor = ResidualBlocksWithInputConv(3, mid_channels, 5)
        self.LTAM = LTAM(stride = self.stride)
        
        # propagation branches
        self.resblocks = ResidualBlocksWithInputConv(2 * mid_channels, mid_channels, num_blocks)
        
        # upsample
        self.fusion = nn.Conv2d(3 * mid_channels, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.FTT = FTT(dct_kernel=dct_kernel, d_model=d_model, n_heads=n_heads)
    
    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True
    
    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        # if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
        #     flows_forward = None
        # else:
        #     flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward
    
    def forward(self, lrs, to_cpu=False):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lrs.size()

        # check whether the input is an extended sequence
        # self.check_if_mirror_extended(lrs)

        # compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)
        outputs = self.feat_extractor(lrs.view(-1,c,h,w)).view(n,t,-1,h,w)
        outputs = torch.unbind(outputs,dim=1)
        outputs = list(outputs)
        keyframe_idx_forward = list(range(0, t, self.keyframe_stride))
        keyframe_idx_backward = list(range(t-1, 0, 0-self.keyframe_stride))
        # backward-time propgation
        feat_buffers = []
        sparse_feat_buffers_s1 = []
        sparse_feat_buffers_s2 = []
        sparse_feat_buffers_s3 = []
        index_feat_buffers_s1 = []
        # index_feat_buffers_s2 = []
        # index_feat_buffers_s3 = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h//self.stride), torch.arange(0, w//self.stride))
        location_update = torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)
        for i in range(t - 1, -1, -1):
            lr_curr = lrs[:, i, :, :, :]
            lr_curr_feat = outputs[i]
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1),padding_mode='border')
                
                # refresh the location map
                flow = F.adaptive_avg_pool2d(flow,(h//self.stride,w//self.stride))/self.stride
                location_update = flow_warp(location_update, flow.permute(0, 2, 3, 1),padding_mode='border',interpolation="nearest")# n , 2t , h , w

                # set the real feature
                feat_buffer = torch.stack(feat_buffers, dim=1)
                sparse_feat_buffer_s1 = torch.stack(sparse_feat_buffers_s1, dim=1)
                sparse_feat_buffer_s2 = torch.stack(sparse_feat_buffers_s2, dim=1)
                sparse_feat_buffer_s3 = torch.stack(sparse_feat_buffers_s3, dim=1)
                index_feat_buffer_s1 = torch.stack(index_feat_buffers_s1, dim=1)
                feat_prop = self.LTAM(lr_curr_feat,index_feat_buffer_s1,feat_prop,sparse_feat_buffer_s1,sparse_feat_buffer_s2,sparse_feat_buffer_s3,location_update)

                # add the location map
                if i in keyframe_idx_backward:
                    location_update = torch.cat([location_update,torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)],dim=1) # n , 2t , h , w
            feat_prop = torch.cat([lr_curr_feat,feat_prop], dim=1)
            feat_prop = self.resblocks(feat_prop)
            feat_buffers.append(feat_prop)
            if i in keyframe_idx_backward:
                # cross-scale feature * 4
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s1 = F.unfold(feat_prop, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s1 = F.fold(sparse_feat_prop_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1) 
                sparse_feat_buffers_s1.append(sparse_feat_prop_s1)

                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                index_feat_prop_s1 = F.unfold(lr_curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                index_feat_prop_s1 = F.fold(index_feat_prop_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1) 
                index_feat_buffers_s1.append(index_feat_prop_s1)


                # cross-scale feature * 8
                # bs * c * h * w --> # bs * (c*8*8) * (h//4*w//4)
                sparse_feat_prop_s2 = F.unfold(feat_prop, kernel_size=(int(1.5*self.stride),int(1.5*self.stride)), padding=int(0.25*self.stride), stride=self.stride) 
                # bs * (c*8*8) * (h//4*w//4) -->  bs * c * (h*2) * (w*2)
                sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(int(1.5*h),int(1.5*w)), kernel_size=(int(1.5*self.stride),int(1.5*self.stride)), padding=0, stride=int(1.5*self.stride))
                # bs * c * (h*2) * (w*2) -->  bs * c * h * w
                sparse_feat_prop_s2 = F.adaptive_avg_pool2d(sparse_feat_prop_s2,(h,w))
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s2 = F.unfold(sparse_feat_prop_s2, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1) 
                sparse_feat_buffers_s2.append(sparse_feat_prop_s2)

                # cross-scale feature * 12
                # bs * c * h * w --> # bs * (c*12*12) * (h//4*w//4)
                sparse_feat_prop_s3 = F.unfold(feat_prop, kernel_size=(int(2*self.stride),int(2*self.stride)), padding=int(0.5*self.stride), stride=self.stride) 
                # bs * (c*12*12) * (h//4*w//4) -->  bs * c * (h*3) * (w*3)
                sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(int(2*h),int(2*w)), kernel_size=(int(2*self.stride),int(2*self.stride)), padding=0, stride=int(2*self.stride))
                # bs * c * (h*3) * (w*3) -->  bs * c * h * w
                sparse_feat_prop_s3 = F.adaptive_avg_pool2d(sparse_feat_prop_s3,(h,w))
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s3 = F.unfold(sparse_feat_prop_s3, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1) 
                sparse_feat_buffers_s3.append(sparse_feat_prop_s3)

            
        outputs_back = feat_buffers[::-1]
        del location_update
        del feat_buffers
        del sparse_feat_buffers_s1
        del sparse_feat_buffers_s2
        del sparse_feat_buffers_s3
        del index_feat_buffers_s1

        # forward-time propagation and upsampling
        fina_out = []
        bicubic_imgs = []
        feat_buffers = []
        sparse_feat_buffers_s1 = []
        sparse_feat_buffers_s2 = []
        sparse_feat_buffers_s3 = []
        index_feat_buffers_s1 = []

        feat_prop = torch.zeros_like(feat_prop)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h//self.stride), torch.arange(0, w//self.stride))
        location_update = torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            lr_curr_feat = outputs[i]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1),padding_mode='border')

                # refresh the location map
                flow = F.adaptive_avg_pool2d(flow,(h//self.stride,w//self.stride))/self.stride
                location_update = flow_warp(location_update, flow.permute(0, 2, 3, 1),padding_mode='border',interpolation="nearest")# n , 2t , h , w

                # set the real feature
                feat_buffer = torch.stack(feat_buffers, dim=1)
                sparse_feat_buffer_s1 = torch.stack(sparse_feat_buffers_s1, dim=1)
                sparse_feat_buffer_s2 = torch.stack(sparse_feat_buffers_s2, dim=1)
                sparse_feat_buffer_s3 = torch.stack(sparse_feat_buffers_s3, dim=1)

                index_feat_buffer_s1 = torch.stack(index_feat_buffers_s1, dim=1)
                feat_prop = self.LTAM(lr_curr_feat,index_feat_buffer_s1,feat_prop,sparse_feat_buffer_s1,sparse_feat_buffer_s2,sparse_feat_buffer_s3,location_update)

                # add the location map
                if i in keyframe_idx_forward:
                    location_update = torch.cat([location_update,torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)],dim=1)
            feat_prop = torch.cat([outputs[i], feat_prop], dim=1)
            feat_prop = self.resblocks(feat_prop)
            feat_buffers.append(feat_prop)

            if i in keyframe_idx_forward:
                # cross-scale feature * 4
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s1 = F.unfold(feat_prop, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s1 = F.fold(sparse_feat_prop_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1) 
                sparse_feat_buffers_s1.append(sparse_feat_prop_s1)

                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                index_feat_prop_s1 = F.unfold(lr_curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                index_feat_prop_s1 = F.fold(index_feat_prop_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1) 
                index_feat_buffers_s1.append(index_feat_prop_s1)


                # cross-scale feature * 8
                # bs * c * h * w --> # bs * (c*8*8) * (h//4*w//4)
                sparse_feat_prop_s2 = F.unfold(feat_prop, kernel_size=(int(1.5*self.stride),int(1.5*self.stride)), padding=int(0.25*self.stride), stride=self.stride) 
                # bs * (c*8*8) * (h//4*w//4) -->  bs * c * (h*2) * (w*2)
                sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(int(1.5*h),int(1.5*w)), kernel_size=(int(1.5*self.stride),int(1.5*self.stride)), padding=0, stride=int(1.5*self.stride))
                # bs * c * (h*2) * (w*2) -->  bs * c * h * w
                sparse_feat_prop_s2 = F.adaptive_avg_pool2d(sparse_feat_prop_s2,(h,w))
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s2 = F.unfold(sparse_feat_prop_s2, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1) 
                sparse_feat_buffers_s2.append(sparse_feat_prop_s2)


                # cross-scale feature * 12
                # bs * c * h * w --> # bs * (c*12*12) * (h//4*w//4)
                sparse_feat_prop_s3 = F.unfold(feat_prop, kernel_size=(int(2*self.stride),int(2*self.stride)), padding=int(0.5*self.stride), stride=self.stride) 
                # bs * (c*12*12) * (h//4*w//4) -->  bs * c * (h*3) * (w*3)
                sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(int(2*h),int(2*w)), kernel_size=(int(2*self.stride),int(2*self.stride)), padding=0, stride=int(2*self.stride))
                # bs * c * (h*3) * (w*3) -->  bs * c * h * w
                sparse_feat_prop_s3 = F.adaptive_avg_pool2d(sparse_feat_prop_s3,(h,w))
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s3 = F.unfold(sparse_feat_prop_s3, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1) 
                sparse_feat_buffers_s3.append(sparse_feat_prop_s3)

            # upsampling given the backward and forward features
            out = torch.cat([outputs_back[i],lr_curr_feat,feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            
            base = self.img_upsample(lr_curr)
            bicubic_imgs.append(base)
            out += base

            fina_out.append(out)
        del location_update
        del sparse_feat_buffers_s1
        del sparse_feat_buffers_s2
        del sparse_feat_buffers_s3
        del index_feat_buffers_s1


        #frequency attention
        high_frequency_out = torch.stack(fina_out, dim=1) #n,t,c,h,w
        bicubic_imgs = torch.stack(bicubic_imgs, dim=1) # n,t,c,h,w

        #padding images
        bicubic_imgs, padding_h, padding_w = check_and_padding_imgs(bicubic_imgs, self.dct_kernel)
        high_frequency_imgs, _, _ = check_and_padding_imgs(high_frequency_out, self.dct_kernel)

        n,t,c,h,w = bicubic_imgs.shape
        flows_forward, flows_backward = self.compute_flow(high_frequency_imgs)

        final_out = self.FTT(
            bicubic_imgs, high_frequency_imgs, 
            [flows_forward, flows_backward], 
            [padding_h, padding_w], to_cpu)

        return final_out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    model = FTVSR().to(device)
    model.eval()

    print(
        "Model have {:.3f}M parameters in total".format(sum(x.numel() for x in model.parameters()) / 1000000.0))

    input = torch.rand(1, 30, 3, 64, 64).to(device)

    with torch.no_grad():
        # print(flop_count_table(FlopCountAnalysis(model, input), activations=ActivationCountAnalysis(model, input)))
        out = model(input)

    print(out.shape)


"""
Model have 45.808M parameters in total
| module                                       | #parameters or shape   | #flops     | #activations   |
|:---------------------------------------------|:-----------------------|:-----------|:---------------|
| model                                        | 45.808M                | 5.261T     | 5.196G         |
|  spynet.basic_module                         |  1.44M                 |  1.292T    |  0.786G        |
|   spynet.basic_module.0.basic_module         |   0.24M                |   0.946G   |   0.576M       |
|    spynet.basic_module.0.basic_module.0.conv |    12.576K             |    49.474M |    0.126M      |
|    spynet.basic_module.0.basic_module.1.conv |    0.1M                |    0.396G  |    0.252M      |
|    spynet.basic_module.0.basic_module.2.conv |    0.1M                |    0.396G  |    0.126M      |
|    spynet.basic_module.0.basic_module.3.conv |    25.104K             |    98.947M |    63.104K     |
|    spynet.basic_module.0.basic_module.4.conv |    1.57K               |    6.184M  |    7.888K      |
|   spynet.basic_module.1.basic_module         |   0.24M                |   3.785G   |   2.303M       |
|    spynet.basic_module.1.basic_module.0.conv |    12.576K             |    0.198G  |    0.505M      |
|    spynet.basic_module.1.basic_module.1.conv |    0.1M                |    1.583G  |    1.01M       |
|    spynet.basic_module.1.basic_module.2.conv |    0.1M                |    1.583G  |    0.505M      |
|    spynet.basic_module.1.basic_module.3.conv |    25.104K             |    0.396G  |    0.252M      |
|    spynet.basic_module.1.basic_module.4.conv |    1.57K               |    24.737M |    31.552K     |
|   spynet.basic_module.2.basic_module         |   0.24M                |   15.139G  |   9.213M       |
|    spynet.basic_module.2.basic_module.0.conv |    12.576K             |    0.792G  |    2.019M      |
|    spynet.basic_module.2.basic_module.1.conv |    0.1M                |    6.333G  |    4.039M      |
|    spynet.basic_module.2.basic_module.2.conv |    0.1M                |    6.333G  |    2.019M      |
|    spynet.basic_module.2.basic_module.3.conv |    25.104K             |    1.583G  |    1.01M       |
|    spynet.basic_module.2.basic_module.4.conv |    1.57K               |    98.947M |    0.126M      |
|   spynet.basic_module.3.basic_module         |   0.24M                |   60.556G  |   36.853M      |
|    spynet.basic_module.3.basic_module.0.conv |    12.576K             |    3.166G  |    8.077M      |
|    spynet.basic_module.3.basic_module.1.conv |    0.1M                |    25.33G  |    16.155M     |
|    spynet.basic_module.3.basic_module.2.conv |    0.1M                |    25.33G  |    8.077M      |
|    spynet.basic_module.3.basic_module.3.conv |    25.104K             |    6.333G  |    4.039M      |
|    spynet.basic_module.3.basic_module.4.conv |    1.57K               |    0.396G  |    0.505M      |
|   spynet.basic_module.4.basic_module         |   0.24M                |   0.242T   |   0.147G       |
|    spynet.basic_module.4.basic_module.0.conv |    12.576K             |    12.665G |    32.309M     |
|    spynet.basic_module.4.basic_module.1.conv |    0.1M                |    0.101T  |    64.618M     |
|    spynet.basic_module.4.basic_module.2.conv |    0.1M                |    0.101T  |    32.309M     |
|    spynet.basic_module.4.basic_module.3.conv |    25.104K             |    25.33G  |    16.155M     |
|    spynet.basic_module.4.basic_module.4.conv |    1.57K               |    1.583G  |    2.019M      |
|   spynet.basic_module.5.basic_module         |   0.24M                |   0.969T   |   0.59G        |
|    spynet.basic_module.5.basic_module.0.conv |    12.576K             |    50.661G |    0.129G      |
|    spynet.basic_module.5.basic_module.1.conv |    0.1M                |    0.405T  |    0.258G      |
|    spynet.basic_module.5.basic_module.2.conv |    0.1M                |    0.405T  |    0.129G      |
|    spynet.basic_module.5.basic_module.3.conv |    25.104K             |    0.101T  |    64.618M     |
|    spynet.basic_module.5.basic_module.4.conv |    1.57K               |    6.333G  |    8.077M      |
|  feat_extractor.main                         |  0.371M                |  45.511G   |  86.508M       |
|   feat_extractor.main.0                      |   1.792K               |   0.212G   |   7.864M       |
|    feat_extractor.main.0.weight              |    (64, 3, 3, 3)       |            |                |
|    feat_extractor.main.0.bias                |    (64,)               |            |                |
|   feat_extractor.main.2                      |   0.369M               |   45.298G  |   78.643M      |
|    feat_extractor.main.2.0                   |    73.856K             |    9.06G   |    15.729M     |
|    feat_extractor.main.2.1                   |    73.856K             |    9.06G   |    15.729M     |
|    feat_extractor.main.2.2                   |    73.856K             |    9.06G   |    15.729M     |
|    feat_extractor.main.2.3                   |    73.856K             |    9.06G   |    15.729M     |
|    feat_extractor.main.2.4                   |    73.856K             |    9.06G   |    15.729M     |
|  LTAM                                        |  0.111M                |  27.655G   |  15.284M       |
|   LTAM.fusion                                |   0.111M               |   26.273G  |   15.204M      |
|    LTAM.fusion.weight                        |    (64, 192, 3, 3)     |            |                |
|    LTAM.fusion.bias                          |    (64,)               |            |                |
|  resblocks.main                              |  4.505M                |  1.105T    |  1.903G        |
|   resblocks.main.0                           |   73.792K              |   18.119G  |   15.729M      |
|    resblocks.main.0.weight                   |    (64, 128, 3, 3)     |            |                |
|    resblocks.main.0.bias                     |    (64,)               |            |                |
|   resblocks.main.2                           |   4.431M               |   1.087T   |   1.887G       |
|    resblocks.main.2.0                        |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.1                        |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.2                        |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.3                        |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.4                        |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.5                        |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.6                        |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.7                        |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.8                        |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.9                        |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.10                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.11                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.12                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.13                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.14                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.15                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.16                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.17                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.18                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.19                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.20                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.21                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.22                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.23                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.24                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.25                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.26                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.27                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.28                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.29                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.30                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.31                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.32                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.33                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.34                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.35                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.36                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.37                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.38                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.39                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.40                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.41                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.42                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.43                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.44                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.45                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.46                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.47                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.48                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.49                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.50                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.51                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.52                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.53                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.54                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.55                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.56                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.57                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.58                       |    73.856K             |    18.119G |    31.457M     |
|    resblocks.main.2.59                       |    73.856K             |    18.119G |    31.457M     |
|  fusion                                      |  12.352K               |  1.51G     |  7.864M        |
|   fusion.weight                              |   (64, 192, 1, 1)      |            |                |
|   fusion.bias                                |   (64,)                |            |                |
|  upsample1.upsample_conv                     |  0.148M                |  18.119G   |  31.457M       |
|   upsample1.upsample_conv.weight             |   (256, 64, 3, 3)      |            |                |
|   upsample1.upsample_conv.bias               |   (256,)               |            |                |
|  upsample2.upsample_conv                     |  0.148M                |  72.478G   |  0.126G        |
|   upsample2.upsample_conv.weight             |   (256, 64, 3, 3)      |            |                |
|   upsample2.upsample_conv.bias               |   (256,)               |            |                |
|  conv_hr                                     |  36.928K               |  72.478G   |  0.126G        |
|   conv_hr.weight                             |   (64, 64, 3, 3)       |            |                |
|   conv_hr.bias                               |   (64,)                |            |                |
|  conv_last                                   |  1.731K                |  3.397G    |  5.898M        |
|   conv_last.weight                           |   (3, 64, 3, 3)        |            |                |
|   conv_last.bias                             |   (3,)                 |            |                |
|  FTT                                         |  39.034M               |  2.622T    |  2.108G        |
|   FTT.dct.dct_conv                           |   12.288K              |   0.755G   |   11.796M      |
|    FTT.dct.dct_conv.weight                   |    (192, 1, 8, 8)      |            |                |
|   FTT.rdct.reverse_dct_conv                  |   12.288K              |   0.377G   |   5.898M       |
|    FTT.rdct.reverse_dct_conv.weight          |    (192, 1, 8, 8)      |            |                |
|   FTT.conv_layer1                            |   98.816K              |   6.04G    |   31.457M      |
|    FTT.conv_layer1.weight                    |    (512, 192, 1, 1)    |            |                |
|    FTT.conv_layer1.bias                      |    (512,)              |            |                |
|   FTT.feat_extractor.main                    |   16.519M              |   1.015T   |   0.22G        |
|    FTT.feat_extractor.main.0                 |    2.36M               |    0.145T  |    31.457M     |
|    FTT.feat_extractor.main.2                 |    14.159M             |    0.87T   |    0.189G      |
|   FTT.resblocks.main                         |   18.878M              |   1.16T    |   0.22G        |
|    FTT.resblocks.main.0                      |    4.719M              |    0.29T   |    31.457M     |
|    FTT.resblocks.main.2                      |    14.159M             |    0.87T   |    0.189G      |
|   FTT.fusion                                 |   1.05M                |   32.212G  |   31.457M      |
|    FTT.fusion.0                              |    0.787M              |    24.159G |    15.729M     |
|    FTT.fusion.2                              |    0.263M              |    8.053G  |    15.729M     |
|   FTT.ftta                                   |   2.366M               |   0.405T   |   1.581G       |
|    FTT.ftta.layer_q                          |    0.263M              |    31.139G |    60.817M     |
|    FTT.ftta.layer_k                          |    0.263M              |    31.139G |    60.817M     |
|    FTT.ftta.layer_v                          |    0.263M              |    31.139G |    60.817M     |
|    FTT.ftta.MultiheadAttention               |    1.051M              |    0.249T  |    1.277G      |
|    FTT.ftta.norm1                            |    1.024K              |    0.304G  |    0           |
|    FTT.ftta.linear1                          |    0.263M              |    31.139G |    60.817M     |
|    FTT.ftta.norm2                            |    1.024K              |    0.304G  |    0           |
|    FTT.ftta.linear2                          |    0.263M              |    31.139G |    60.817M     |
|   FTT.conv_layer2                            |   98.496K              |   3.02G    |   5.898M       |
|    FTT.conv_layer2.weight                    |    (192, 512, 1, 1)    |            |                |
|    FTT.conv_layer2.bias                      |    (192,)              |            |                |
|  img_upsample                                |                        |  23.593M   |  0             |
torch.Size([1, 30, 3, 256, 256])
"""
