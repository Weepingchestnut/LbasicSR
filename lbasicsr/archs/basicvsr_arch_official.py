import torch
from torch import nn as nn
from torch.nn import functional as F

from lbasicsr.utils.registry import ARCH_REGISTRY
from lbasicsr.archs.arch_util import ResidualBlockNoBN, flow_warp, make_layer
from lbasicsr.archs.edvr_arch import PCDAlignment, TSAFusion
from lbasicsr.archs.spynet_arch import SpyNet


# @ARCH_REGISTRY.register()
class BasicVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)


# @ARCH_REGISTRY.register()
class IconVSR(nn.Module):
    """IconVSR, proposed also in the BasicVSR paper.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15.
        keyframe_stride (int): Keyframe stride. Default: 5.
        temporal_padding (int): Temporal padding. Default: 2.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        edvr_path (str): Path to the pretrained EDVR model. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_block=15,
                 keyframe_stride=5,
                 temporal_padding=2,
                 spynet_path=None,
                 edvr_path=None):
        super().__init__()

        self.num_feat = num_feat
        self.temporal_padding = temporal_padding
        self.keyframe_stride = keyframe_stride

        # keyframe_branch
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat, edvr_path)
        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk = ConvResidualBlocks(2 * num_feat + 3, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def pad_spatial(self, x):
        """Apply padding spatially.

        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.

        Args:
            x (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = x.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        return x.view(n, t, c, h + pad_h, w + pad_w)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def get_keyframe_feature(self, x, keyframe_idx):
        if self.temporal_padding == 2:
            x = [x[:, [4, 3]], x, x[:, [-4, -5]]]
        elif self.temporal_padding == 3:
            x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]
        x = torch.cat(x, dim=1)

        num_frames = 2 * self.temporal_padding + 1
        feats_keyframe = {}
        for i in keyframe_idx:
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe

    def forward(self, x):
        b, n, _, h_input, w_input = x.size()

        x = self.pad_spatial(x)
        h, w = x.shape[3:]

        keyframe_idx = list(range(0, n, self.keyframe_stride))
        if keyframe_idx[-1] != n - 1:
            keyframe_idx.append(n - 1)  # last frame is a keyframe

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)
        feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.backward_fusion(feat_prop)
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)

            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)[..., :4 * h_input, :4 * w_input]


class EDVRFeatureExtractor(nn.Module):
    """EDVR feature extractor used in IconVSR.

    Args:
        num_input_frame (int): Number of input frames.
        num_feat (int): Number of feature channels
        load_path (str): Path to the pretrained weights of EDVR. Default: None.
    """

    def __init__(self, num_input_frame, num_feat, load_path):

        super(EDVRFeatureExtractor, self).__init__()

        self.center_frame_idx = num_input_frame // 2

        # extract pyramid features
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=8)
        self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_input_frame, center_frame_idx=self.center_frame_idx)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

    def forward(self, x):
        b, n, c, h, w = x.size()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, n, -1, h, w)
        feat_l2 = feat_l2.view(b, n, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, n, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(n):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        # TSA fusion
        return self.fusion(aligned_feat)


if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    x = torch.randn(1, 1, 3, 320, 180)
    # x = torch.randn(1, 3, 256, 256)

    model = BasicVSR(
        num_feat=64,
        num_block=30
    )

    # print(model)
    print(
        "Model have {:.3f}M parameters in total".format(
            sum(x.numel() for x in model.parameters()) / 1000000.0))
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))

    output = model(x)
    print(output.shape)


"""
Model have 6.291M parameters in total
params: 6291311
| module                                  | #parameters or shape   | #flops    | #activations   |
|:----------------------------------------|:-----------------------|:----------|:---------------|
| model                                   | 6.291M                 | 0.338T    | 0.589G         |
|  spynet.basic_module                    |  1.44M                 |           |                |
|   spynet.basic_module.0.basic_module    |   0.24M                |           |                |
|    spynet.basic_module.0.basic_module.0 |    12.576K             |           |                |
|    spynet.basic_module.0.basic_module.2 |    0.1M                |           |                |
|    spynet.basic_module.0.basic_module.4 |    0.1M                |           |                |
|    spynet.basic_module.0.basic_module.6 |    25.104K             |           |                |
|    spynet.basic_module.0.basic_module.8 |    1.57K               |           |                |
|   spynet.basic_module.1.basic_module    |   0.24M                |           |                |
|    spynet.basic_module.1.basic_module.0 |    12.576K             |           |                |
|    spynet.basic_module.1.basic_module.2 |    0.1M                |           |                |
|    spynet.basic_module.1.basic_module.4 |    0.1M                |           |                |
|    spynet.basic_module.1.basic_module.6 |    25.104K             |           |                |
|    spynet.basic_module.1.basic_module.8 |    1.57K               |           |                |
|   spynet.basic_module.2.basic_module    |   0.24M                |           |                |
|    spynet.basic_module.2.basic_module.0 |    12.576K             |           |                |
|    spynet.basic_module.2.basic_module.2 |    0.1M                |           |                |
|    spynet.basic_module.2.basic_module.4 |    0.1M                |           |                |
|    spynet.basic_module.2.basic_module.6 |    25.104K             |           |                |
|    spynet.basic_module.2.basic_module.8 |    1.57K               |           |                |
|   spynet.basic_module.3.basic_module    |   0.24M                |           |                |
|    spynet.basic_module.3.basic_module.0 |    12.576K             |           |                |
|    spynet.basic_module.3.basic_module.2 |    0.1M                |           |                |
|    spynet.basic_module.3.basic_module.4 |    0.1M                |           |                |
|    spynet.basic_module.3.basic_module.6 |    25.104K             |           |                |
|    spynet.basic_module.3.basic_module.8 |    1.57K               |           |                |
|   spynet.basic_module.4.basic_module    |   0.24M                |           |                |
|    spynet.basic_module.4.basic_module.0 |    12.576K             |           |                |
|    spynet.basic_module.4.basic_module.2 |    0.1M                |           |                |
|    spynet.basic_module.4.basic_module.4 |    0.1M                |           |                |
|    spynet.basic_module.4.basic_module.6 |    25.104K             |           |                |
|    spynet.basic_module.4.basic_module.8 |    1.57K               |           |                |
|   spynet.basic_module.5.basic_module    |   0.24M                |           |                |
|    spynet.basic_module.5.basic_module.0 |    12.576K             |           |                |
|    spynet.basic_module.5.basic_module.2 |    0.1M                |           |                |
|    spynet.basic_module.5.basic_module.4 |    0.1M                |           |                |
|    spynet.basic_module.5.basic_module.6 |    25.104K             |           |                |
|    spynet.basic_module.5.basic_module.8 |    1.57K               |           |                |
|  backward_trunk.main                    |  2.254M                |  0.13T    |  0.225G        |
|   backward_trunk.main.0                 |   38.656K              |   2.223G  |   3.686M       |
|    backward_trunk.main.0.weight         |    (64, 67, 3, 3)      |           |                |
|    backward_trunk.main.0.bias           |    (64,)               |           |                |
|   backward_trunk.main.2                 |   2.216M               |   0.127T  |   0.221G       |
|    backward_trunk.main.2.0              |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.1              |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.2              |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.3              |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.4              |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.5              |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.6              |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.7              |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.8              |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.9              |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.10             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.11             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.12             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.13             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.14             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.15             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.16             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.17             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.18             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.19             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.20             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.21             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.22             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.23             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.24             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.25             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.26             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.27             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.28             |    73.856K             |    4.247G |    7.373M      |
|    backward_trunk.main.2.29             |    73.856K             |    4.247G |    7.373M      |
|  forward_trunk.main                     |  2.254M                |  0.13T    |  0.225G        |
|   forward_trunk.main.0                  |   38.656K              |   2.223G  |   3.686M       |
|    forward_trunk.main.0.weight          |    (64, 67, 3, 3)      |           |                |
|    forward_trunk.main.0.bias            |    (64,)               |           |                |
|   forward_trunk.main.2                  |   2.216M               |   0.127T  |   0.221G       |
|    forward_trunk.main.2.0               |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.1               |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.2               |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.3               |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.4               |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.5               |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.6               |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.7               |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.8               |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.9               |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.10              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.11              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.12              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.13              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.14              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.15              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.16              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.17              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.18              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.19              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.20              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.21              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.22              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.23              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.24              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.25              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.26              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.27              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.28              |    73.856K             |    4.247G |    7.373M      |
|    forward_trunk.main.2.29              |    73.856K             |    4.247G |    7.373M      |
|  fusion                                 |  8.256K                |  0.472G   |  3.686M        |
|   fusion.weight                         |   (64, 128, 1, 1)      |           |                |
|   fusion.bias                           |   (64,)                |           |                |
|  upconv1                                |  0.148M                |  8.493G   |  14.746M       |
|   upconv1.weight                        |   (256, 64, 3, 3)      |           |                |
|   upconv1.bias                          |   (256,)               |           |                |
|  upconv2                                |  0.148M                |  33.974G  |  58.982M       |
|   upconv2.weight                        |   (256, 64, 3, 3)      |           |                |
|   upconv2.bias                          |   (256,)               |           |                |
|  conv_hr                                |  36.928K               |  33.974G  |  58.982M       |
|   conv_hr.weight                        |   (64, 64, 3, 3)       |           |                |
|   conv_hr.bias                          |   (64,)                |           |                |
|  conv_last                              |  1.731K                |  1.593G   |  2.765M        |
|   conv_last.weight                      |   (3, 64, 3, 3)        |           |                |
|   conv_last.bias                        |   (3,)                 |           |                |
torch.Size([1, 1, 3, 1280, 720])
"""
