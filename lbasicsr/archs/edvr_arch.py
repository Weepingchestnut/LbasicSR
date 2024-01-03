import torch
from torch import nn as nn
from torch.nn import functional as F

from lbasicsr.metrics.flops import get_flops
from lbasicsr.metrics.runtime import VSR_runtime_test
from lbasicsr.utils.registry import ARCH_REGISTRY
from lbasicsr.archs.arch_util import DCNv2Pack, ResidualBlockNoBN, make_layer


class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    ``Paper: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks``

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat


class PredeblurModule(nn.Module):
    """Pre-dublur module.

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        hr_in (bool): Whether the input has high resolution. Default: False.
    """

    def __init__(self, num_in_ch=3, num_feat=64, hr_in=False):
        super(PredeblurModule, self).__init__()
        self.hr_in = hr_in

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        if self.hr_in:
            # downsample x4 by stride conv
            self.stride_conv_hr1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
            self.stride_conv_hr2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        # generate feature pyramid
        self.stride_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.stride_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        self.resblock_l3 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_2 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l1 = nn.ModuleList([ResidualBlockNoBN(num_feat=num_feat) for i in range(5)])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        feat_l1 = self.lrelu(self.conv_first(x))
        if self.hr_in:
            feat_l1 = self.lrelu(self.stride_conv_hr1(feat_l1))
            feat_l1 = self.lrelu(self.stride_conv_hr2(feat_l1))

        # generate feature pyramid
        feat_l2 = self.lrelu(self.stride_conv_l2(feat_l1))
        feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))

        feat_l3 = self.upsample(self.resblock_l3(feat_l3))
        feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
        feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))

        for i in range(2):
            feat_l1 = self.resblock_l1[i](feat_l1)
        feat_l1 = feat_l1 + feat_l2
        for i in range(2, 5):
            feat_l1 = self.resblock_l1[i](feat_l1)
        return feat_l1


# @ARCH_REGISTRY.register()
class EDVR(nn.Module):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.

    ``Paper: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks``

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        num_reconstruct_block (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: Middle of input frames.
        hr_in (bool): Whether the input has high resolution. Default: False.
        with_predeblur (bool): Whether has predeblur module.
            Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=None,
                 hr_in=False,
                 with_predeblur=False,
                 with_tsa=True):
        super(EDVR, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa

        # extract features for each frame
        if self.with_predeblur:
            self.predeblur = PredeblurModule(num_feat=num_feat, hr_in=self.hr_in)
            self.conv_1x1 = nn.Conv2d(num_feat, num_feat, 1, 1)
        else:
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        if self.with_tsa:
            self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_frame, center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        b, t, c, h, w = x.size()
        origin_h, origin_w = h, w
        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, ('The height and width must be multiple of 16.')
        else:
            # assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')
            # ------ for arbitrary-scale VSR ----------------------------------
            if w % 4 != 0:
                pad_w = int(w // 4 + 1) * 4 - w
                x = F.pad(x, (0, pad_w), mode='constant', value=0)
            if h % 4 != 0:
                pad_h = int(h // 4 + 1) * 4 - h
                x = F.pad(x, (0, 0, 0, pad_h), mode='constant', value=0)
        b, t, c, h, w = x.size()

        # extract features for each frame
        # L1
        if self.with_predeblur:
            feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w)))
            if self.hr_in:
                h, w = h // 4, w // 4
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)
        # ------ for arbitrary-scale VSR ----------------------------------
        feat = feat[..., 0:origin_h, 0:origin_w]

        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        if self.hr_in:
            base = x_center
        else:
            base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        return out


if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    from torch.profiler import profile, record_function, ProfilerActivity
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    
    # EDVR-M -----------------
    # net = EDVR().to(device)
    # EDVR-L -----------------
    scale = (4, 4)
    model = EDVR(
        num_feat=128, 
        num_frame=7, 
        num_reconstruct_block=40
    ).to(device)
    model.eval()
    
    input = torch.rand(1, 7, 3, 180, 320).to(device)
    
    # ------ torch profile -------------------------
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,], 
        record_shapes=True,
        profile_memory=True,
        # use_cuda=True
    ) as prof:
        with record_function("model_inference"):
            for _ in range(input.shape[1]):
                out = model(input)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    
    # ------ Runtime ----------------------
    VSR_runtime_test(model, input, scale)

    # ------ Parameter --------------------
    print(
        "Model have {:.3f}M parameters in total".format(sum(x.numel() for x in model.parameters()) / 1000000.0))

    # ------ FLOPs ------------------------
    with torch.no_grad():
        print('Input:', input.shape)
        print(flop_count_table(FlopCountAnalysis(model, input), activations=ActivationCountAnalysis(model, input)))
        out = model(input)
        print('Output:', out.shape)


"""
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        model_inference         5.37%     140.068ms        98.03%        2.557s        2.557s       0.000us         0.00%        1.636s        1.636s         476 b         460 b      10.53 Gb    -112.40 Gb             1  
                                           aten::conv2d         0.24%       6.327ms        45.36%        1.183s     754.457us       0.000us         0.00%     980.546ms     625.348us           0 b           0 b      52.58 Gb           0 b          1568  
                                      aten::convolution         0.28%       7.330ms        45.11%        1.177s     750.422us       0.000us         0.00%     980.546ms     625.348us           0 b           0 b      52.58 Gb           0 b          1568  
                                     aten::_convolution         0.78%      20.405ms        44.83%        1.169s     745.747us       0.000us         0.00%     980.546ms     625.348us           0 b           0 b      52.58 Gb           0 b          1568  
                                aten::cudnn_convolution         7.51%     195.913ms        42.63%        1.112s     709.071us     812.658ms        49.66%     812.658ms     518.277us           0 b           0 b      52.58 Gb      -2.24 Gb          1568  
sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nh...         0.00%       0.000us         0.00%       0.000us       0.000us     576.642ms        35.24%     576.642ms     416.048us           0 b           0 b           0 b           0 b          1386  
                             torchvision::deform_conv2d         0.92%      24.035ms         3.44%      89.621ms     457.250us     172.938ms        10.57%     301.212ms       1.537ms           0 b           0 b       3.13 Gb     -34.33 Gb           196  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us     178.286ms        10.90%     178.286ms     101.069us           0 b           0 b           0 b           0 b          1764  
void cudnn::ops::nchwToNhwcKernel<float, float, floa...         0.00%       0.000us         0.00%       0.000us       0.000us     178.162ms        10.89%     178.162ms      58.645us           0 b           0 b           0 b           0 b          3038  
void vision::ops::(anonymous namespace)::deformable_...         0.00%       0.000us         0.00%       0.000us       0.000us     172.938ms        10.57%     172.938ms     882.337us           0 b           0 b           0 b           0 b           196  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.608s
Self CUDA time total: 1.636s

Warm up ...

Testing ...

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [01:12<00:00,  4.14it/s]

Average Runtime: 240.708 ms

Model have 20.699M parameters in total
Input: torch.Size([1, 7, 3, 180, 320])
| module                               | #parameters or shape   | #flops     | #activations   |
|:-------------------------------------|:-----------------------|:-----------|:---------------|
| model                                | 20.699M                | 2.489T     | 2.013G         |
|  conv_first                          |  3.584K                |  1.393G    |  51.61M        |
|   conv_first.weight                  |   (128, 3, 3, 3)       |            |                |
|   conv_first.bias                    |   (128,)               |            |                |
|  feature_extraction                  |  1.476M                |  0.595T    |  0.516G        |
|   feature_extraction.0               |   0.295M               |   0.119T   |   0.103G       |
|    feature_extraction.0.conv1        |    0.148M              |    59.454G |    51.61M      |
|    feature_extraction.0.conv2        |    0.148M              |    59.454G |    51.61M      |
|   feature_extraction.1               |   0.295M               |   0.119T   |   0.103G       |
|    feature_extraction.1.conv1        |    0.148M              |    59.454G |    51.61M      |
|    feature_extraction.1.conv2        |    0.148M              |    59.454G |    51.61M      |
|   feature_extraction.2               |   0.295M               |   0.119T   |   0.103G       |
|    feature_extraction.2.conv1        |    0.148M              |    59.454G |    51.61M      |
|    feature_extraction.2.conv2        |    0.148M              |    59.454G |    51.61M      |
|   feature_extraction.3               |   0.295M               |   0.119T   |   0.103G       |
|    feature_extraction.3.conv1        |    0.148M              |    59.454G |    51.61M      |
|    feature_extraction.3.conv2        |    0.148M              |    59.454G |    51.61M      |
|   feature_extraction.4               |   0.295M               |   0.119T   |   0.103G       |
|    feature_extraction.4.conv1        |    0.148M              |    59.454G |    51.61M      |
|    feature_extraction.4.conv2        |    0.148M              |    59.454G |    51.61M      |
|  conv_l2_1                           |  0.148M                |  14.864G   |  12.902M       |
|   conv_l2_1.weight                   |   (128, 128, 3, 3)     |            |                |
|   conv_l2_1.bias                     |   (128,)               |            |                |
|  conv_l2_2                           |  0.148M                |  14.864G   |  12.902M       |
|   conv_l2_2.weight                   |   (128, 128, 3, 3)     |            |                |
|   conv_l2_2.bias                     |   (128,)               |            |                |
|  conv_l3_1                           |  0.148M                |  3.716G    |  3.226M        |
|   conv_l3_1.weight                   |   (128, 128, 3, 3)     |            |                |
|   conv_l3_1.bias                     |   (128,)               |            |                |
|  conv_l3_2                           |  0.148M                |  3.716G    |  3.226M        |
|   conv_l3_2.weight                   |   (128, 128, 3, 3)     |            |                |
|   conv_l3_2.bias                     |   (128,)               |            |                |
|  pcd_align                           |  4.537M                |  0.942T    |  0.569G        |
|   pcd_align.offset_conv1             |   0.885M               |   0.156T   |   67.738M      |
|    pcd_align.offset_conv1.l3         |    0.295M              |    7.432G  |    3.226M      |
|    pcd_align.offset_conv1.l2         |    0.295M              |    29.727G |    12.902M     |
|    pcd_align.offset_conv1.l1         |    0.295M              |    0.119T  |    51.61M      |
|   pcd_align.offset_conv2             |   0.738M               |   0.152T   |   67.738M      |
|    pcd_align.offset_conv2.l3         |    0.148M              |    3.716G  |    3.226M      |
|    pcd_align.offset_conv2.l2         |    0.295M              |    29.727G |    12.902M     |
|    pcd_align.offset_conv2.l1         |    0.295M              |    0.119T  |    51.61M      |
|   pcd_align.offset_conv3             |   0.295M               |   74.318G  |   64.512M      |
|    pcd_align.offset_conv3.l2         |    0.148M              |    14.864G |    12.902M     |
|    pcd_align.offset_conv3.l1         |    0.148M              |    59.454G |    51.61M      |
|   pcd_align.dcn_pack                 |   1.19M                |   0.132T   |   0.114G       |
|    pcd_align.dcn_pack.l3             |    0.397M              |    6.271G  |    5.443M      |
|    pcd_align.dcn_pack.l2             |    0.397M              |    25.082G |    21.773M     |
|    pcd_align.dcn_pack.l1             |    0.397M              |    0.1T    |    87.091M     |
|   pcd_align.feat_conv                |   0.59M                |   0.149T   |   64.512M      |
|    pcd_align.feat_conv.l2            |    0.295M              |    29.727G |    12.902M     |
|    pcd_align.feat_conv.l1            |    0.295M              |    0.119T  |    51.61M      |
|   pcd_align.cas_offset_conv1         |   0.295M               |   0.119T   |   51.61M       |
|    pcd_align.cas_offset_conv1.weight |    (128, 256, 3, 3)    |            |                |
|    pcd_align.cas_offset_conv1.bias   |    (128,)              |            |                |
|   pcd_align.cas_offset_conv2         |   0.148M               |   59.454G  |   51.61M       |
|    pcd_align.cas_offset_conv2.weight |    (128, 128, 3, 3)    |            |                |
|    pcd_align.cas_offset_conv2.bias   |    (128,)              |            |                |
|   pcd_align.cas_dcnpack              |   0.397M               |   0.1T     |   87.091M      |
|    pcd_align.cas_dcnpack.weight      |    (128, 128, 3, 3)    |            |                |
|    pcd_align.cas_dcnpack.bias        |    (128,)              |            |                |
|    pcd_align.cas_dcnpack.conv_offset |    0.249M              |    0.1T    |    87.091M     |
|   pcd_align.upsample                 |                        |   0.516G   |   0            |
|  fusion                              |  1.362M                |  96.237G   |  0.104G        |
|   fusion.temporal_attn1              |   0.148M               |   8.493G   |   7.373M       |
|    fusion.temporal_attn1.weight      |    (128, 128, 3, 3)    |            |                |
|    fusion.temporal_attn1.bias        |    (128,)              |            |                |
|   fusion.temporal_attn2              |   0.148M               |   59.454G  |   51.61M       |
|    fusion.temporal_attn2.weight      |    (128, 128, 3, 3)    |            |                |
|    fusion.temporal_attn2.bias        |    (128,)              |            |                |
|   fusion.feat_fusion                 |   0.115M               |   6.606G   |   7.373M       |
|    fusion.feat_fusion.weight         |    (128, 896, 1, 1)    |            |                |
|    fusion.feat_fusion.bias           |    (128,)              |            |                |
|   fusion.spatial_attn1               |   0.115M               |   6.606G   |   7.373M       |
|    fusion.spatial_attn1.weight       |    (128, 896, 1, 1)    |            |                |
|    fusion.spatial_attn1.bias         |    (128,)              |            |                |
|   fusion.spatial_attn2               |   32.896K              |   0.472G   |   1.843M       |
|    fusion.spatial_attn2.weight       |    (128, 256, 1, 1)    |            |                |
|    fusion.spatial_attn2.bias         |    (128,)              |            |                |
|   fusion.spatial_attn3               |   0.148M               |   2.123G   |   1.843M       |
|    fusion.spatial_attn3.weight       |    (128, 128, 3, 3)    |            |                |
|    fusion.spatial_attn3.bias         |    (128,)              |            |                |
|   fusion.spatial_attn4               |   16.512K              |   0.236G   |   1.843M       |
|    fusion.spatial_attn4.weight       |    (128, 128, 1, 1)    |            |                |
|    fusion.spatial_attn4.bias         |    (128,)              |            |                |
|   fusion.spatial_attn5               |   0.148M               |   8.493G   |   7.373M       |
|    fusion.spatial_attn5.weight       |    (128, 128, 3, 3)    |            |                |
|    fusion.spatial_attn5.bias         |    (128,)              |            |                |
|   fusion.spatial_attn_l1             |   16.512K              |   0.236G   |   1.843M       |
|    fusion.spatial_attn_l1.weight     |    (128, 128, 1, 1)    |            |                |
|    fusion.spatial_attn_l1.bias       |    (128,)              |            |                |
|   fusion.spatial_attn_l2             |   0.295M               |   1.062G   |   0.461M       |
|    fusion.spatial_attn_l2.weight     |    (128, 256, 3, 3)    |            |                |
|    fusion.spatial_attn_l2.bias       |    (128,)              |            |                |
|   fusion.spatial_attn_l3             |   0.148M               |   0.531G   |   0.461M       |
|    fusion.spatial_attn_l3.weight     |    (128, 128, 3, 3)    |            |                |
|    fusion.spatial_attn_l3.bias       |    (128,)              |            |                |
|   fusion.spatial_attn_add1           |   16.512K              |   0.944G   |   7.373M       |
|    fusion.spatial_attn_add1.weight   |    (128, 128, 1, 1)    |            |                |
|    fusion.spatial_attn_add1.bias     |    (128,)              |            |                |
|   fusion.spatial_attn_add2           |   16.512K              |   0.944G   |   7.373M       |
|    fusion.spatial_attn_add2.weight   |    (128, 128, 1, 1)    |            |                |
|    fusion.spatial_attn_add2.bias     |    (128,)              |            |                |
|   fusion.upsample                    |                        |   36.864M  |   0            |
|  reconstruction                      |  11.807M               |  0.679T    |  0.59G         |
|   reconstruction.0                   |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.0.conv1            |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.0.conv2            |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.1                   |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.1.conv1            |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.1.conv2            |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.2                   |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.2.conv1            |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.2.conv2            |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.3                   |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.3.conv1            |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.3.conv2            |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.4                   |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.4.conv1            |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.4.conv2            |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.5                   |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.5.conv1            |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.5.conv2            |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.6                   |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.6.conv1            |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.6.conv2            |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.7                   |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.7.conv1            |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.7.conv2            |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.8                   |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.8.conv1            |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.8.conv2            |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.9                   |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.9.conv1            |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.9.conv2            |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.10                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.10.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.10.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.11                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.11.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.11.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.12                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.12.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.12.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.13                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.13.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.13.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.14                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.14.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.14.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.15                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.15.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.15.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.16                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.16.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.16.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.17                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.17.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.17.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.18                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.18.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.18.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.19                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.19.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.19.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.20                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.20.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.20.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.21                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.21.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.21.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.22                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.22.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.22.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.23                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.23.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.23.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.24                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.24.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.24.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.25                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.25.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.25.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.26                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.26.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.26.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.27                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.27.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.27.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.28                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.28.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.28.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.29                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.29.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.29.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.30                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.30.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.30.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.31                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.31.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.31.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.32                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.32.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.32.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.33                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.33.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.33.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.34                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.34.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.34.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.35                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.35.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.35.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.36                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.36.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.36.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.37                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.37.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.37.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.38                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.38.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.38.conv2           |    0.148M              |    8.493G  |    7.373M      |
|   reconstruction.39                  |   0.295M               |   16.987G  |   14.746M      |
|    reconstruction.39.conv1           |    0.148M              |    8.493G  |    7.373M      |
|    reconstruction.39.conv2           |    0.148M              |    8.493G  |    7.373M      |
|  upconv1                             |  0.59M                 |  33.974G   |  29.491M       |
|   upconv1.weight                     |   (512, 128, 3, 3)     |            |                |
|   upconv1.bias                       |   (512,)               |            |                |
|  upconv2                             |  0.295M                |  67.948G   |  58.982M       |
|   upconv2.weight                     |   (256, 128, 3, 3)     |            |                |
|   upconv2.bias                       |   (256,)               |            |                |
|  conv_hr                             |  36.928K               |  33.974G   |  58.982M       |
|   conv_hr.weight                     |   (64, 64, 3, 3)       |            |                |
|   conv_hr.bias                       |   (64,)                |            |                |
|  conv_last                           |  1.731K                |  1.593G    |  2.765M        |
|   conv_last.weight                   |   (3, 64, 3, 3)        |            |                |
|   conv_last.bias                     |   (3,)                 |            |                |
Output: torch.Size([1, 3, 720, 1280])
"""
