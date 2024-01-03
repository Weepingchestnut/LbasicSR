import torch
from torch import nn as nn
from torch.nn import functional as F

from lbasicsr.metrics.runtime import VSR_runtime_test
from lbasicsr.utils.registry import ARCH_REGISTRY
from lbasicsr.archs.arch_util import ResidualBlockNoBN, flow_warp, make_layer
from lbasicsr.archs.edvr_arch import PCDAlignment, TSAFusion
from lbasicsr.archs.spynet_arch import SpyNet


@ARCH_REGISTRY.register()
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
        b, n, c, h, w = x.size()                            # [b, t, 3, h, w]       e.g. 0 1 2 3 4 5 6

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)       # [b*(t-1), 3, h, w]    e.g. 0 1 2 3 4 5
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)        # [b*(t-1), 3, h, w]    e.g. 1 2 3 4 5 6

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)              # ref <-- supp, 0 <-- 1
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)               # ref <-- supp, 0 --> 1

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


@ARCH_REGISTRY.register()
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
    from torch.profiler import profile, record_function, ProfilerActivity
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    scale = (4, 4)
    model = BasicVSR(
        num_feat=64,
        num_block=30
    ).to(device)
    # -------------------
    # model = IconVSR(
    #     num_feat=64,
    #     num_block=30,
    #     keyframe_stride=5,
    #     temporal_padding=2
    # ).to(device)
    model.eval()
    
    input = torch.randn(1, 7, 3, 180, 320).to(device)
    
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
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    
    # ------ FLOPs --------------------------------
    with torch.no_grad():
        print('Input:', input.shape)
        print(flop_count_table(FlopCountAnalysis(model, input), activations=ActivationCountAnalysis(model, input)))
        out = model(input)
        print('Output:', out.shape)


"""
# BaiscVSR

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        model_inference         3.72%      58.775ms        99.42%        1.571s        1.571s       0.000us         0.00%     336.607ms     336.607ms       4.07 Kb     -58.57 Kb      16.41 Gb     -14.43 Gb             1  
                                           aten::conv2d         0.23%       3.577ms        77.11%        1.219s       1.284ms       0.000us         0.00%     246.209ms     259.440us           0 b           0 b      15.91 Gb           0 b           949  
                                      aten::convolution         0.25%       4.015ms        76.88%        1.215s       1.280ms       0.000us         0.00%     246.209ms     259.440us           0 b           0 b      15.91 Gb           0 b           949  
                                     aten::_convolution         0.70%      11.141ms        76.63%        1.211s       1.276ms       0.000us         0.00%     246.209ms     259.440us           0 b           0 b      15.91 Gb           0 b           949  
                                aten::cudnn_convolution         7.08%     111.838ms        74.47%        1.177s       1.240ms     194.741ms        57.85%     194.741ms     205.207us           0 b           0 b      15.91 Gb      14.03 Gb           949  
ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile14...         0.00%       0.000us         0.00%       0.000us       0.000us     135.970ms        40.39%     135.970ms     157.921us           0 b           0 b           0 b           0 b           861  
                                             aten::add_         0.63%       9.914ms         1.04%      16.359ms      17.112us      51.823ms        15.40%      51.823ms      54.208us           0 b           0 b           0 b           0 b           956  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      51.630ms        15.34%      51.630ms      52.845us           0 b           0 b           0 b           0 b           977  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      28.588ms         8.49%      28.588ms      65.121us           0 b           0 b           0 b           0 b           439  
                                              aten::add         0.68%      10.723ms         5.90%      93.206ms     204.399us      28.329ms         8.42%      28.329ms      62.125us           0 b           0 b       5.79 Gb       5.79 Gb           456  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.580s
Self CUDA time total: 336.607ms

Warm up ...

Testing ...

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [01:48<00:00,  2.76it/s]

Average Runtime: 361.87100290934245 ms

Model have 6.291M parameters in total
params: 6291311
Input: torch.Size([1, 7, 3, 180, 320])
| module                                  | #parameters or shape   | #flops     | #activations   |
|:----------------------------------------|:-----------------------|:-----------|:---------------|
| model                                   | 6.291M                 | 2.6T       | 4.266G         |
|  spynet.basic_module                    |  1.44M                 |  0.236T    |  0.143G        |
|   spynet.basic_module.0.basic_module    |   0.24M                |   0.173G   |   0.105M       |
|    spynet.basic_module.0.basic_module.0 |    12.576K             |    9.032M  |    23.04K      |
|    spynet.basic_module.0.basic_module.2 |    0.1M                |    72.253M |    46.08K      |
|    spynet.basic_module.0.basic_module.4 |    0.1M                |    72.253M |    23.04K      |
|    spynet.basic_module.0.basic_module.6 |    25.104K             |    18.063M |    11.52K      |
|    spynet.basic_module.0.basic_module.8 |    1.57K               |    1.129M  |    1.44K       |
|   spynet.basic_module.1.basic_module    |   0.24M                |   0.691G   |   0.42M        |
|    spynet.basic_module.1.basic_module.0 |    12.576K             |    36.127M |    92.16K      |
|    spynet.basic_module.1.basic_module.2 |    0.1M                |    0.289G  |    0.184M      |
|    spynet.basic_module.1.basic_module.4 |    0.1M                |    0.289G  |    92.16K      |
|    spynet.basic_module.1.basic_module.6 |    25.104K             |    72.253M |    46.08K      |
|    spynet.basic_module.1.basic_module.8 |    1.57K               |    4.516M  |    5.76K       |
|   spynet.basic_module.2.basic_module    |   0.24M                |   2.764G   |   1.682M       |
|    spynet.basic_module.2.basic_module.0 |    12.576K             |    0.145G  |    0.369M      |
|    spynet.basic_module.2.basic_module.2 |    0.1M                |    1.156G  |    0.737M      |
|    spynet.basic_module.2.basic_module.4 |    0.1M                |    1.156G  |    0.369M      |
|    spynet.basic_module.2.basic_module.6 |    25.104K             |    0.289G  |    0.184M      |
|    spynet.basic_module.2.basic_module.8 |    1.57K               |    18.063M |    23.04K      |
|   spynet.basic_module.3.basic_module    |   0.24M                |   11.055G  |   6.728M       |
|    spynet.basic_module.3.basic_module.0 |    12.576K             |    0.578G  |    1.475M      |
|    spynet.basic_module.3.basic_module.2 |    0.1M                |    4.624G  |    2.949M      |
|    spynet.basic_module.3.basic_module.4 |    0.1M                |    4.624G  |    1.475M      |
|    spynet.basic_module.3.basic_module.6 |    25.104K             |    1.156G  |    0.737M      |
|    spynet.basic_module.3.basic_module.8 |    1.57K               |    72.253M |    92.16K      |
|   spynet.basic_module.4.basic_module    |   0.24M                |   44.219G  |   26.911M      |
|    spynet.basic_module.4.basic_module.0 |    12.576K             |    2.312G  |    5.898M      |
|    spynet.basic_module.4.basic_module.2 |    0.1M                |    18.497G |    11.796M     |
|    spynet.basic_module.4.basic_module.4 |    0.1M                |    18.497G |    5.898M      |
|    spynet.basic_module.4.basic_module.6 |    25.104K             |    4.624G  |    2.949M      |
|    spynet.basic_module.4.basic_module.8 |    1.57K               |    0.289G  |    0.369M      |
|   spynet.basic_module.5.basic_module    |   0.24M                |   0.177T   |   0.108G       |
|    spynet.basic_module.5.basic_module.0 |    12.576K             |    9.248G  |    23.593M     |
|    spynet.basic_module.5.basic_module.2 |    0.1M                |    73.988G |    47.186M     |
|    spynet.basic_module.5.basic_module.4 |    0.1M                |    73.988G |    23.593M     |
|    spynet.basic_module.5.basic_module.6 |    25.104K             |    18.497G |    11.796M     |
|    spynet.basic_module.5.basic_module.8 |    1.57K               |    1.156G  |    1.475M      |
|  backward_trunk.main                    |  2.254M                |  0.907T    |  1.574G        |
|   backward_trunk.main.0                 |   38.656K              |   15.56G   |   25.805M      |
|    backward_trunk.main.0.weight         |    (64, 67, 3, 3)      |            |                |
|    backward_trunk.main.0.bias           |    (64,)               |            |                |
|   backward_trunk.main.2                 |   2.216M               |   0.892T   |   1.548G       |
|    backward_trunk.main.2.0              |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.1              |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.2              |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.3              |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.4              |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.5              |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.6              |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.7              |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.8              |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.9              |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.10             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.11             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.12             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.13             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.14             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.15             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.16             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.17             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.18             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.19             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.20             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.21             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.22             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.23             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.24             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.25             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.26             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.27             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.28             |    73.856K             |    29.727G |    51.61M      |
|    backward_trunk.main.2.29             |    73.856K             |    29.727G |    51.61M      |
|  forward_trunk.main                     |  2.254M                |  0.907T    |  1.574G        |
|   forward_trunk.main.0                  |   38.656K              |   15.56G   |   25.805M      |
|    forward_trunk.main.0.weight          |    (64, 67, 3, 3)      |            |                |
|    forward_trunk.main.0.bias            |    (64,)               |            |                |
|   forward_trunk.main.2                  |   2.216M               |   0.892T   |   1.548G       |
|    forward_trunk.main.2.0               |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.1               |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.2               |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.3               |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.4               |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.5               |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.6               |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.7               |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.8               |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.9               |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.10              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.11              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.12              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.13              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.14              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.15              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.16              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.17              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.18              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.19              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.20              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.21              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.22              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.23              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.24              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.25              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.26              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.27              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.28              |    73.856K             |    29.727G |    51.61M      |
|    forward_trunk.main.2.29              |    73.856K             |    29.727G |    51.61M      |
|  fusion                                 |  8.256K                |  3.303G    |  25.805M       |
|   fusion.weight                         |   (64, 128, 1, 1)      |            |                |
|   fusion.bias                           |   (64,)                |            |                |
|  upconv1                                |  0.148M                |  59.454G   |  0.103G        |
|   upconv1.weight                        |   (256, 64, 3, 3)      |            |                |
|   upconv1.bias                          |   (256,)               |            |                |
|  upconv2                                |  0.148M                |  0.238T    |  0.413G        |
|   upconv2.weight                        |   (256, 64, 3, 3)      |            |                |
|   upconv2.bias                          |   (256,)               |            |                |
|  conv_hr                                |  36.928K               |  0.238T    |  0.413G        |
|   conv_hr.weight                        |   (64, 64, 3, 3)       |            |                |
|   conv_hr.bias                          |   (64,)                |            |                |
|  conv_last                              |  1.731K                |  11.148G   |  19.354M       |
|   conv_last.weight                      |   (3, 64, 3, 3)        |            |                |
|   conv_last.bias                        |   (3,)                 |            |                |
Output: torch.Size([1, 7, 3, 720, 1280])

# IconVSR

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        model_inference        75.11%        6.295s        99.89%        8.372s        8.372s       0.000us         0.00%     523.838ms     523.838ms       4.45 Kb     -70.58 Kb      25.64 Gb     -20.78 Gb             1  
                                           aten::conv2d         0.06%       5.321ms        16.96%        1.421s       1.117ms       0.000us         0.00%     335.720ms     263.931us           0 b           0 b      21.88 Gb           0 b          1272  
                                      aten::convolution         0.07%       5.938ms        16.89%        1.416s       1.113ms       0.000us         0.00%     335.720ms     263.931us           0 b           0 b      21.88 Gb           0 b          1272  
                                     aten::_convolution         0.20%      16.518ms        16.82%        1.410s       1.108ms       0.000us         0.00%     335.720ms     263.931us           0 b           0 b      21.88 Gb           0 b          1272  
                                aten::cudnn_convolution         1.95%     163.127ms        16.24%        1.361s       1.070ms     265.398ms        50.66%     265.398ms     208.646us           0 b           0 b      21.88 Gb      18.60 Gb          1272  
ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile14...         0.00%       0.000us         0.00%       0.000us       0.000us     184.921ms        35.30%     184.921ms     169.964us           0 b           0 b           0 b           0 b          1088  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      72.112ms        13.77%      72.112ms      53.024us           0 b           0 b           0 b           0 b          1360  
                                             aten::add_         0.17%      14.254ms         0.27%      22.242ms      17.390us      70.672ms        13.49%      70.672ms      55.256us           0 b           0 b           0 b           0 b          1279  
                             torchvision::deform_conv2d         0.08%       6.770ms         0.44%      36.898ms     614.967us      26.761ms         5.11%      46.363ms     772.717us           0 b           0 b     487.79 Mb      -5.24 Gb            60  
                                              aten::add         0.16%      13.754ms         1.33%     111.087ms     206.866us      35.067ms         6.69%      35.067ms      65.302us           0 b           0 b       7.35 Gb       7.35 Gb           537  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 8.381s
Self CUDA time total: 523.838ms

Warm up ...

Testing ...

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [02:47<00:00,  1.79it/s]

Average Runtime: 558.6389986165365 ms

Model have 8.695M parameters in total
params: 8694991
Input: torch.Size([1, 7, 3, 180, 320])
"""
