import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings

from lbasicsr.archs.arch_util import flow_warp
from lbasicsr.archs.basicvsr_arch import ConvResidualBlocks
from lbasicsr.archs.spynet_arch import SpyNet
from lbasicsr.metrics.runtime import VSR_runtime_test
from lbasicsr.ops.dcn import ModulatedDeformConvPack
from lbasicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class BasicVSRPlusPlus(nn.Module):
    """BasicVSR++ network structure.

    Support either x4 upsampling or same size output. Since DCN is used in this
    model, it can only be used with CUDA enabled. If CUDA is not enabled,
    feature alignment will be skipped. Besides, we adopt the official DCN
    implementation and the version of torch need to be higher than 1.9.

    ``Paper: BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment``

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_path=None,
                 cpu_cache_length=100):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        self.spynet = SpyNet(spynet_path)

        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ConvResidualBlocks(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ConvResidualBlocks(mid_channels, mid_channels, 5))

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    2 * mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deformable_groups=16,
                    max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ConvResidualBlocks((2 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = ConvResidualBlocks(5 * mid_channels, mid_channels, 5)

        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn('Deformable alignment module is not added. '
                          'Probably your CUDA is not configured correctly. DCN can only '
                          'be used with CUDA enabled. Alignment is skipped now.')

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the flows used for forward-time propagation \
                (current to previous). 'flows_backward' corresponds to the flows used for backward-time \
                propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = flows_backward.flip(1)
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated \
                features. Each key in the dictionary corresponds to a \
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)


class SecondOrderDeformableAlignment(ModulatedDeformConvPack):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)


if __name__ == '__main__':
    from torch.profiler import profile, record_function, ProfilerActivity
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    
    scale = (4, 4)
    model = BasicVSRPlusPlus(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_path='experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth'
    ).to(device)
    
    input = torch.rand(1, 7, 3, 180, 320).to(device)
    
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
    with torch.no_grad():
        print('Input:', input.shape)
        print(flop_count_table(FlopCountAnalysis(model, input), activations=ActivationCountAnalysis(model, input)))
        out = model(input)
        print('Output:', out.shape)


"""
test on A6000

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        model_inference         4.51%      74.453ms        96.02%        1.585s        1.585s       0.000us         0.00%     432.055ms     432.055ms       4.45 Kb    -261.32 Kb      20.24 Gb     -21.50 Gb             1  
                                           aten::conv2d         0.21%       3.545ms        74.68%        1.233s       1.782ms       0.000us         0.00%     251.474ms     363.402us           0 b           0 b      15.18 Gb           0 b           692  
                                      aten::convolution         0.21%       3.460ms        74.47%        1.229s       1.777ms       0.000us         0.00%     251.474ms     363.402us           0 b           0 b      15.18 Gb           0 b           692  
                                     aten::_convolution         0.64%      10.531ms        74.26%        1.226s       1.772ms       0.000us         0.00%     251.474ms     363.402us           0 b           0 b      15.18 Gb           0 b           692  
                                aten::cudnn_convolution         5.94%      98.046ms        72.48%        1.197s       1.729ms     202.637ms        46.90%     202.637ms     292.828us           0 b           0 b      15.18 Gb       9.68 Gb           692  
ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile14...         0.00%       0.000us         0.00%       0.000us       0.000us      99.376ms        23.00%      99.376ms     186.097us           0 b           0 b           0 b           0 b           534  
                             torchvision::deform_conv2d         0.22%       3.564ms         1.38%      22.710ms     946.250us      37.145ms         8.60%      59.589ms       2.483ms           0 b           0 b     337.50 Mb      -6.61 Gb            24  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      50.314ms        11.65%      50.314ms      63.209us           0 b           0 b           0 b           0 b           796  
                                             aten::add_         0.50%       8.298ms         0.79%      12.966ms      18.549us      49.189ms        11.38%      49.189ms      70.371us           0 b           0 b           0 b           0 b           699  
void vision::ops::(anonymous namespace)::deformable_...         0.00%       0.000us         0.00%       0.000us       0.000us      37.145ms         8.60%      37.145ms       1.548ms           0 b           0 b           0 b           0 b            24  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.651s
Self CUDA time total: 432.055ms

Warm up ...

Testing ...

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [02:18<00:00,  2.16it/s]

Average Runtime: 461.0952089436849 ms

Model have 7.323M parameters in total
Input: torch.Size([1, 7, 3, 180, 320])
| module                                  | #parameters or shape   | #flops     | #activations   |
|:----------------------------------------|:-----------------------|:-----------|:---------------|
| model                                   | 7.323M                 | 2.798T     | 4.07G          |
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
|  feat_extract.main                      |  0.371M                |  0.149T    |  0.284G        |
|   feat_extract.main.0                   |   1.792K               |   0.697G   |   25.805M      |
|    feat_extract.main.0.weight           |    (64, 3, 3, 3)       |            |                |
|    feat_extract.main.0.bias             |    (64,)               |            |                |
|   feat_extract.main.2                   |   0.369M               |   0.149T   |   0.258G       |
|    feat_extract.main.2.0                |    73.856K             |    29.727G |    51.61M      |
|    feat_extract.main.2.1                |    73.856K             |    29.727G |    51.61M      |
|    feat_extract.main.2.2                |    73.856K             |    29.727G |    51.61M      |
|    feat_extract.main.2.3                |    73.856K             |    29.727G |    51.61M      |
|    feat_extract.main.2.4                |    73.856K             |    29.727G |    51.61M      |
|  deform_align                           |  2.039M                |  0.602T    |  0.863G        |
|   deform_align.backward_1               |   0.51M                |   0.15T    |   0.216G       |
|    deform_align.backward_1.weight       |    (64, 128, 3, 3)     |            |                |
|    deform_align.backward_1.bias         |    (64,)               |            |                |
|    deform_align.backward_1.conv_offset  |    0.436M              |    0.15T   |    0.216G      |
|   deform_align.forward_1                |   0.51M                |   0.15T    |   0.216G       |
|    deform_align.forward_1.weight        |    (64, 128, 3, 3)     |            |                |
|    deform_align.forward_1.bias          |    (64,)               |            |                |
|    deform_align.forward_1.conv_offset   |    0.436M              |    0.15T   |    0.216G      |
|   deform_align.backward_2               |   0.51M                |   0.15T    |   0.216G       |
|    deform_align.backward_2.weight       |    (64, 128, 3, 3)     |            |                |
|    deform_align.backward_2.bias         |    (64,)               |            |                |
|    deform_align.backward_2.conv_offset  |    0.436M              |    0.15T   |    0.216G      |
|   deform_align.forward_2                |   0.51M                |   0.15T    |   0.216G       |
|    deform_align.forward_2.weight        |    (64, 128, 3, 3)     |            |                |
|    deform_align.forward_2.bias          |    (64,)               |            |                |
|    deform_align.forward_2.conv_offset   |    0.436M              |    0.15T   |    0.216G      |
|  backbone                               |  2.584M                |  1.04T     |  1.548G        |
|   backbone.backward_1.main              |   0.591M               |   0.238T   |   0.387G       |
|    backbone.backward_1.main.0           |    73.792K             |    29.727G |    25.805M     |
|    backbone.backward_1.main.2           |    0.517M              |    0.208T  |    0.361G      |
|   backbone.forward_1.main               |   0.628M               |   0.253T   |   0.387G       |
|    backbone.forward_1.main.0            |    0.111M              |    44.591G |    25.805M     |
|    backbone.forward_1.main.2            |    0.517M              |    0.208T  |    0.361G      |
|   backbone.backward_2.main              |   0.665M               |   0.268T   |   0.387G       |
|    backbone.backward_2.main.0           |    0.148M              |    59.454G |    25.805M     |
|    backbone.backward_2.main.2           |    0.517M              |    0.208T  |    0.361G      |
|   backbone.forward_2.main               |   0.701M               |   0.282T   |   0.387G       |
|    backbone.forward_2.main.0            |    0.184M              |    74.318G |    25.805M     |
|    backbone.forward_2.main.2            |    0.517M              |    0.208T  |    0.361G      |
|  reconstruction.main                    |  0.554M                |  0.223T    |  0.284G        |
|   reconstruction.main.0                 |   0.184M               |   74.318G  |   25.805M      |
|    reconstruction.main.0.weight         |    (64, 320, 3, 3)     |            |                |
|    reconstruction.main.0.bias           |    (64,)               |            |                |
|   reconstruction.main.2                 |   0.369M               |   0.149T   |   0.258G       |
|    reconstruction.main.2.0              |    73.856K             |    29.727G |    51.61M      |
|    reconstruction.main.2.1              |    73.856K             |    29.727G |    51.61M      |
|    reconstruction.main.2.2              |    73.856K             |    29.727G |    51.61M      |
|    reconstruction.main.2.3              |    73.856K             |    29.727G |    51.61M      |
|    reconstruction.main.2.4              |    73.856K             |    29.727G |    51.61M      |
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
|  img_upsample                           |                        |  77.414M   |  0             |
Output: torch.Size([1, 7, 3, 720, 1280])
"""