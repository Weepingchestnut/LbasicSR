import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from lbasicsr.metrics.runtime import VSR_runtime_test

from lbasicsr.utils.registry import ARCH_REGISTRY


class DenseBlocksTemporalReduce(nn.Module):
    """A concatenation of 3 dense blocks with reduction in temporal dimension.

    Note that the output temporal dimension is 6 fewer the input temporal dimension, since there are 3 blocks.

    Args:
        num_feat (int): Number of channels in the blocks. Default: 64.
        num_grow_ch (int): Growing factor of the dense blocks. Default: 32
        adapt_official_weights (bool): Whether to adapt the weights translated from the official implementation.
            Set to false if you want to train from scratch. Default: False.
    """

    def __init__(self, num_feat=64, num_grow_ch=32, adapt_official_weights=False):
        super(DenseBlocksTemporalReduce, self).__init__()
        if adapt_official_weights:
            eps = 1e-3
            momentum = 1e-3
        else:  # pytorch default values
            eps = 1e-05
            momentum = 0.1

        self.temporal_reduce1 = nn.Sequential(
            nn.BatchNorm3d(num_feat, eps=eps, momentum=momentum), nn.ReLU(inplace=True),
            nn.Conv3d(num_feat, num_feat, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.BatchNorm3d(num_feat, eps=eps, momentum=momentum), nn.ReLU(inplace=True),
            nn.Conv3d(num_feat, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))

        self.temporal_reduce2 = nn.Sequential(
            nn.BatchNorm3d(num_feat + num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True),
            nn.Conv3d(
                num_feat + num_grow_ch,
                num_feat + num_grow_ch, (1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=True), nn.BatchNorm3d(num_feat + num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True),
            nn.Conv3d(num_feat + num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))

        self.temporal_reduce3 = nn.Sequential(
            nn.BatchNorm3d(num_feat + 2 * num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True),
            nn.Conv3d(
                num_feat + 2 * num_grow_ch,
                num_feat + 2 * num_grow_ch, (1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=True), nn.BatchNorm3d(num_feat + 2 * num_grow_ch, eps=eps, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                num_feat + 2 * num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (b, num_feat, t, h, w).

        Returns:
            Tensor: Output with shape (b, num_feat + num_grow_ch * 3, 1, h, w).
        """
        x1 = self.temporal_reduce1(x)
        x1 = torch.cat((x[:, :, 1:-1, :, :], x1), 1)

        x2 = self.temporal_reduce2(x1)
        x2 = torch.cat((x1[:, :, 1:-1, :, :], x2), 1)

        x3 = self.temporal_reduce3(x2)
        x3 = torch.cat((x2[:, :, 1:-1, :, :], x3), 1)

        return x3


class DenseBlocks(nn.Module):
    """ A concatenation of N dense blocks.

    Args:
        num_feat (int): Number of channels in the blocks. Default: 64.
        num_grow_ch (int): Growing factor of the dense blocks. Default: 32.
        num_block (int): Number of dense blocks. The values are:
            DUF-S (16 layers): 3
            DUF-M (18 layers): 9
            DUF-L (52 layers): 21
        adapt_official_weights (bool): Whether to adapt the weights translated from the official implementation.
            Set to false if you want to train from scratch. Default: False.
    """

    def __init__(self, num_block, num_feat=64, num_grow_ch=16, adapt_official_weights=False):
        super(DenseBlocks, self).__init__()
        if adapt_official_weights:
            eps = 1e-3
            momentum = 1e-3
        else:  # pytorch default values
            eps = 1e-05
            momentum = 0.1

        self.dense_blocks = nn.ModuleList()
        for i in range(0, num_block):
            self.dense_blocks.append(
                nn.Sequential(
                    nn.BatchNorm3d(num_feat + i * num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True),
                    nn.Conv3d(
                        num_feat + i * num_grow_ch,
                        num_feat + i * num_grow_ch, (1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                        bias=True), nn.BatchNorm3d(num_feat + i * num_grow_ch, eps=eps, momentum=momentum),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(
                        num_feat + i * num_grow_ch,
                        num_grow_ch, (3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                        bias=True)))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (b, num_feat, t, h, w).

        Returns:
            Tensor: Output with shape (b, num_feat + num_block * num_grow_ch, t, h, w).
        """
        for i in range(0, len(self.dense_blocks)):
            y = self.dense_blocks[i](x)
            x = torch.cat((x, y), 1)
        return x


class DynamicUpsamplingFilter(nn.Module):
    """Dynamic upsampling filter used in DUF.

    Reference: https://github.com/yhjo09/VSR-DUF

    It only supports input with 3 channels. And it applies the same filters to 3 channels.

    Args:
        filter_size (tuple): Filter size of generated filters. The shape is (kh, kw). Default: (5, 5).
    """

    def __init__(self, filter_size=(5, 5)):
        super(DynamicUpsamplingFilter, self).__init__()
        if not isinstance(filter_size, tuple):
            raise TypeError(f'The type of filter_size must be tuple, but got type{filter_size}')
        if len(filter_size) != 2:
            raise ValueError(f'The length of filter size must be 2, but got {len(filter_size)}.')
        # generate a local expansion filter, similar to im2col
        self.filter_size = filter_size
        filter_prod = np.prod(filter_size)
        expansion_filter = torch.eye(int(filter_prod)).view(filter_prod, 1, *filter_size)  # (kh*kw, 1, kh, kw)
        self.expansion_filter = expansion_filter.repeat(3, 1, 1, 1)  # repeat for all the 3 channels

    def forward(self, x, filters):
        """Forward function for DynamicUpsamplingFilter.

        Args:
            x (Tensor): Input image with 3 channels. The shape is (n, 3, h, w).
            filters (Tensor): Generated dynamic filters. The shape is (n, filter_prod, upsampling_square, h, w).
                filter_prod: prod of filter kernel size, e.g., 1*5*5=25.
                upsampling_square: similar to pixel shuffle, upsampling_square = upsampling * upsampling.
                e.g., for x 4 upsampling, upsampling_square= 4*4 = 16

        Returns:
            Tensor: Filtered image with shape (n, 3*upsampling_square, h, w)
        """
        n, filter_prod, upsampling_square, h, w = filters.size()
        kh, kw = self.filter_size
        expanded_input = F.conv2d(
            x, self.expansion_filter.to(x), padding=(kh // 2, kw // 2), groups=3)  # (n, 3*filter_prod, h, w)
        expanded_input = expanded_input.view(n, 3, filter_prod, h, w).permute(0, 3, 4, 1,
                                                                              2)  # (n, h, w, 3, filter_prod)
        filters = filters.permute(0, 3, 4, 1, 2)  # (n, h, w, filter_prod, upsampling_square]
        out = torch.matmul(expanded_input, filters)  # (n, h, w, 3, upsampling_square)
        return out.permute(0, 3, 4, 1, 2).view(n, 3 * upsampling_square, h, w)


# @ARCH_REGISTRY.register()
class DUF(nn.Module):
    """Network architecture for DUF

    ``Paper: Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation``

    Reference: https://github.com/yhjo09/VSR-DUF

    For all the models below, 'adapt_official_weights' is only necessary when
    loading the weights converted from the official TensorFlow weights.
    Please set it to False if you are training the model from scratch.

    There are three models with different model size: DUF16Layers, DUF28Layers,
    and DUF52Layers. This class is the base class for these models.

    Args:
        scale (int): The upsampling factor. Default: 4.
        num_layer (int): The number of layers. Default: 52.
        adapt_official_weights_weights (bool): Whether to adapt the weights
            translated from the official implementation. Set to false if you
            want to train from scratch. Default: False.
    """

    def __init__(self, scale=4, num_layer=52, adapt_official_weights=False):
        super(DUF, self).__init__()
        self.scale = scale
        if adapt_official_weights:
            eps = 1e-3
            momentum = 1e-3
        else:  # pytorch default values
            eps = 1e-05
            momentum = 0.1

        self.conv3d1 = nn.Conv3d(3, 64, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.dynamic_filter = DynamicUpsamplingFilter((5, 5))

        if num_layer == 16:
            num_block = 3
            num_grow_ch = 32
        elif num_layer == 28:
            num_block = 9
            num_grow_ch = 16
        elif num_layer == 52:
            num_block = 21
            num_grow_ch = 16
        else:
            raise ValueError(f'Only supported (16, 28, 52) layers, but got {num_layer}.')

        self.dense_block1 = DenseBlocks(
            num_block=num_block, num_feat=64, num_grow_ch=num_grow_ch,
            adapt_official_weights=adapt_official_weights)  # T = 7
        self.dense_block2 = DenseBlocksTemporalReduce(
            64 + num_grow_ch * num_block, num_grow_ch, adapt_official_weights=adapt_official_weights)  # T = 1
        channels = 64 + num_grow_ch * num_block + num_grow_ch * 3
        self.bn3d2 = nn.BatchNorm3d(channels, eps=eps, momentum=momentum)
        self.conv3d2 = nn.Conv3d(channels, 256, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)

        self.conv3d_r1 = nn.Conv3d(256, 256, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.conv3d_r2 = nn.Conv3d(256, 3 * (scale**2), (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)

        self.conv3d_f1 = nn.Conv3d(256, 512, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.conv3d_f2 = nn.Conv3d(
            512, 1 * 5 * 5 * (scale**2), (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input with shape (b, 7, c, h, w)

        Returns:
            Tensor: Output with shape (b, c, h * scale, w * scale)
        """
        num_batches, num_imgs, _, h, w = x.size()

        x = x.permute(0, 2, 1, 3, 4)  # (b, c, 7, h, w) for Conv3D
        x_center = x[:, :, num_imgs // 2, :, :]

        x = self.conv3d1(x)
        x = self.dense_block1(x)
        x = self.dense_block2(x)
        x = F.relu(self.bn3d2(x), inplace=True)
        x = F.relu(self.conv3d2(x), inplace=True)

        # residual image
        res = self.conv3d_r2(F.relu(self.conv3d_r1(x), inplace=True))

        # filter
        filter_ = self.conv3d_f2(F.relu(self.conv3d_f1(x), inplace=True))
        filter_ = F.softmax(filter_.view(num_batches, 25, self.scale**2, h, w), dim=1)

        # dynamic filter
        out = self.dynamic_filter(x_center, filter_)
        out += res.squeeze_(2)
        out = F.pixel_shuffle(out, self.scale)

        return out


if __name__ == '__main__':
    from torch.profiler import profile, record_function, ProfilerActivity
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    
    scale = (4, 4)
    model = DUF(
        scale=4, 
        num_layer=52, 
        adapt_official_weights=True
    ).to(device)
    model.eval()
    
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
            for _ in range(input.shape[1]):
                out = model(input)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # ------ Runtime ------------------------------
    VSR_runtime_test(model, input, scale)
    
    # ------ Parameter ----------------------------
    print(
        "Model have {:.3f}M parameters in total".format(sum(x.numel() for x in model.parameters()) / 1000000.0))

    with torch.no_grad():
        print('Input:', input.shape)
        print(flop_count_table(FlopCountAnalysis(model, input), activations=ActivationCountAnalysis(model, input)))
        out = model(input)
        print('Output:', out.shape)


"""
A6000 OOM, test on A800

STAGE:2023-11-20 04:16:58 353945:353945 ActivityProfilerController.cpp:311] Completed Stage: Warm Up
STAGE:2023-11-20 04:17:06 353945:353945 ActivityProfilerController.cpp:317] Completed Stage: Collection
STAGE:2023-11-20 04:17:06 353945:353945 ActivityProfilerController.cpp:321] Completed Stage: Post Processing
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        model_inference         0.58%      46.857ms       100.00%        8.139s        8.139s       0.000us         0.00%        2.180s        2.180s           0 b           0 b      34.12 Gb    -218.12 Gb             1  
                                      aten::convolution         0.04%       3.360ms        80.13%        6.522s      16.940ms       0.000us         0.00%        1.591s       4.132ms           0 b           0 b      65.54 Gb           0 b           385  
                                     aten::_convolution         0.06%       4.508ms        80.09%        6.518s      16.931ms       0.000us         0.00%        1.591s       4.132ms           0 b           0 b      65.54 Gb     -39.29 Mb           385  
                                           aten::conv3d         0.04%       3.073ms        80.12%        6.522s      17.253ms       0.000us         0.00%        1.527s       4.040ms           0 b           0 b      65.42 Gb       1.08 Gb           378  
                                aten::cudnn_convolution         2.60%     211.939ms        79.87%        6.501s      17.198ms        1.307s        69.48%        1.491s       3.943ms           0 b           0 b      65.42 Gb      65.42 Gb           378  
sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f3...         0.00%       0.000us         0.00%       0.000us       0.000us        1.079s        57.36%        1.079s       6.424ms           0 b           0 b           0 b           0 b           168  
                                 aten::cudnn_batch_norm         0.17%      13.467ms         1.28%     103.893ms     302.895us     245.116ms        13.03%     290.110ms     845.802us           0 b           0 b     118.22 Gb           0 b           343  
                           aten::_batch_norm_impl_index         0.02%       1.358ms         1.29%     105.077ms     306.347us       0.000us         0.00%     288.025ms     839.723us           0 b           0 b     118.22 Gb    1010.53 Mb           343  
                                       aten::batch_norm         0.05%       3.910ms         1.31%     106.318ms     309.965us       0.000us         0.00%     273.973ms     798.755us           0 b           0 b     118.22 Gb       6.88 Gb           343  
void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float,...         0.00%       0.000us         0.00%       0.000us       0.000us     245.116ms        13.03%     245.116ms     714.624us           0 b           0 b           0 b           0 b           343  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 8.139s
Self CUDA time total: 1.882s

Warm up ...

Testing ...

100%|███████████████████████████████████████████████████████████████████████████████████████████| 300/300 [01:20<00:00,  3.73it/s]

Average Runtime: 267.19112579345705 ms

Model have 5.822M parameters in total
Input: torch.Size([1, 7, 3, 180, 320])
| module                             | #parameters or shape   | #flops     | #activations   |
|:-----------------------------------|:-----------------------|:-----------|:---------------|
| model                              | 5.822M                 | 1.655T     | 2.514G         |
|  conv3d1                           |  1.792K                |  0.697G    |  25.805M       |
|   conv3d1.weight                   |   (64, 3, 1, 3, 3)     |            |                |
|   conv3d1.bias                     |   (64,)                |            |                |
|  dense_block1.dense_blocks         |  3.307M                |  1.331T    |  2.032G        |
|   dense_block1.dense_blocks.0      |   32.08K               |   12.902G  |   32.256M      |
|    dense_block1.dense_blocks.0.0   |    0.128K              |    51.61M  |    0           |
|    dense_block1.dense_blocks.0.2   |    4.16K               |    1.652G  |    25.805M     |
|    dense_block1.dense_blocks.0.3   |    0.128K              |    51.61M  |    0           |
|    dense_block1.dense_blocks.0.5   |    27.664K             |    11.148G |    6.451M      |
|   dense_block1.dense_blocks.1      |   41.376K              |   16.644G  |   38.707M      |
|    dense_block1.dense_blocks.1.0   |    0.16K               |    64.512M |    0           |
|    dense_block1.dense_blocks.1.2   |    6.48K               |    2.58G   |    32.256M     |
|    dense_block1.dense_blocks.1.3   |    0.16K               |    64.512M |    0           |
|    dense_block1.dense_blocks.1.5   |    34.576K             |    13.935G |    6.451M      |
|   dense_block1.dense_blocks.2      |   51.184K              |   20.592G  |   45.158M      |
|    dense_block1.dense_blocks.2.0   |    0.192K              |    77.414M |    0           |
|    dense_block1.dense_blocks.2.2   |    9.312K              |    3.716G  |    38.707M     |
|    dense_block1.dense_blocks.2.3   |    0.192K              |    77.414M |    0           |
|    dense_block1.dense_blocks.2.5   |    41.488K             |    16.722G |    6.451M      |
|   dense_block1.dense_blocks.3      |   61.504K              |   24.747G  |   51.61M       |
|    dense_block1.dense_blocks.3.0   |    0.224K              |    90.317M |    0           |
|    dense_block1.dense_blocks.3.2   |    12.656K             |    5.058G  |    45.158M     |
|    dense_block1.dense_blocks.3.3   |    0.224K              |    90.317M |    0           |
|    dense_block1.dense_blocks.3.5   |    48.4K               |    19.508G |    6.451M      |
|   dense_block1.dense_blocks.4      |   72.336K              |   29.108G  |   58.061M      |
|    dense_block1.dense_blocks.4.0   |    0.256K              |    0.103G  |    0           |
|    dense_block1.dense_blocks.4.2   |    16.512K             |    6.606G  |    51.61M      |
|    dense_block1.dense_blocks.4.3   |    0.256K              |    0.103G  |    0           |
|    dense_block1.dense_blocks.4.5   |    55.312K             |    22.295G |    6.451M      |
|   dense_block1.dense_blocks.5      |   83.68K               |   33.675G  |   64.512M      |
|    dense_block1.dense_blocks.5.0   |    0.288K              |    0.116G  |    0           |
|    dense_block1.dense_blocks.5.2   |    20.88K              |    8.361G  |    58.061M     |
|    dense_block1.dense_blocks.5.3   |    0.288K              |    0.116G  |    0           |
|    dense_block1.dense_blocks.5.5   |    62.224K             |    25.082G |    6.451M      |
|   dense_block1.dense_blocks.6      |   95.536K              |   38.449G  |   70.963M      |
|    dense_block1.dense_blocks.6.0   |    0.32K               |    0.129G  |    0           |
|    dense_block1.dense_blocks.6.2   |    25.76K              |    10.322G |    64.512M     |
|    dense_block1.dense_blocks.6.3   |    0.32K               |    0.129G  |    0           |
|    dense_block1.dense_blocks.6.5   |    69.136K             |    27.869G |    6.451M      |
|   dense_block1.dense_blocks.7      |   0.108M               |   43.429G  |   77.414M      |
|    dense_block1.dense_blocks.7.0   |    0.352K              |    0.142G  |    0           |
|    dense_block1.dense_blocks.7.2   |    31.152K             |    12.49G  |    70.963M     |
|    dense_block1.dense_blocks.7.3   |    0.352K              |    0.142G  |    0           |
|    dense_block1.dense_blocks.7.5   |    76.048K             |    30.656G |    6.451M      |
|   dense_block1.dense_blocks.8      |   0.121M               |   48.616G  |   83.866M      |
|    dense_block1.dense_blocks.8.0   |    0.384K              |    0.155G  |    0           |
|    dense_block1.dense_blocks.8.2   |    37.056K             |    14.864G |    77.414M     |
|    dense_block1.dense_blocks.8.3   |    0.384K              |    0.155G  |    0           |
|    dense_block1.dense_blocks.8.5   |    82.96K              |    33.443G |    6.451M      |
|   dense_block1.dense_blocks.9      |   0.134M               |   54.009G  |   90.317M      |
|    dense_block1.dense_blocks.9.0   |    0.416K              |    0.168G  |    0           |
|    dense_block1.dense_blocks.9.2   |    43.472K             |    17.444G |    83.866M     |
|    dense_block1.dense_blocks.9.3   |    0.416K              |    0.168G  |    0           |
|    dense_block1.dense_blocks.9.5   |    89.872K             |    36.23G  |    6.451M      |
|   dense_block1.dense_blocks.10     |   0.148M               |   59.609G  |   96.768M      |
|    dense_block1.dense_blocks.10.0  |    0.448K              |    0.181G  |    0           |
|    dense_block1.dense_blocks.10.2  |    50.4K               |    20.231G |    90.317M     |
|    dense_block1.dense_blocks.10.3  |    0.448K              |    0.181G  |    0           |
|    dense_block1.dense_blocks.10.5  |    96.784K             |    39.017G |    6.451M      |
|   dense_block1.dense_blocks.11     |   0.162M               |   65.415G  |   0.103G       |
|    dense_block1.dense_blocks.11.0  |    0.48K               |    0.194G  |    0           |
|    dense_block1.dense_blocks.11.2  |    57.84K              |    23.224G |    96.768M     |
|    dense_block1.dense_blocks.11.3  |    0.48K               |    0.194G  |    0           |
|    dense_block1.dense_blocks.11.5  |    0.104M              |    41.804G |    6.451M      |
|   dense_block1.dense_blocks.12     |   0.177M               |   71.428G  |   0.11G        |
|    dense_block1.dense_blocks.12.0  |    0.512K              |    0.206G  |    0           |
|    dense_block1.dense_blocks.12.2  |    65.792K             |    26.424G |    0.103G      |
|    dense_block1.dense_blocks.12.3  |    0.512K              |    0.206G  |    0           |
|    dense_block1.dense_blocks.12.5  |    0.111M              |    44.591G |    6.451M      |
|   dense_block1.dense_blocks.13     |   0.193M               |   77.647G  |   0.116G       |
|    dense_block1.dense_blocks.13.0  |    0.544K              |    0.219G  |    0           |
|    dense_block1.dense_blocks.13.2  |    74.256K             |    29.83G  |    0.11G       |
|    dense_block1.dense_blocks.13.3  |    0.544K              |    0.219G  |    0           |
|    dense_block1.dense_blocks.13.5  |    0.118M              |    47.378G |    6.451M      |
|   dense_block1.dense_blocks.14     |   0.209M               |   84.072G  |   0.123G       |
|    dense_block1.dense_blocks.14.0  |    0.576K              |    0.232G  |    0           |
|    dense_block1.dense_blocks.14.2  |    83.232K             |    33.443G |    0.116G      |
|    dense_block1.dense_blocks.14.3  |    0.576K              |    0.232G  |    0           |
|    dense_block1.dense_blocks.14.5  |    0.124M              |    50.165G |    6.451M      |
|   dense_block1.dense_blocks.15     |   0.225M               |   90.704G  |   0.129G       |
|    dense_block1.dense_blocks.15.0  |    0.608K              |    0.245G  |    0           |
|    dense_block1.dense_blocks.15.2  |    92.72K              |    37.262G |    0.123G      |
|    dense_block1.dense_blocks.15.3  |    0.608K              |    0.245G  |    0           |
|    dense_block1.dense_blocks.15.5  |    0.131M              |    52.951G |    6.451M      |
|   dense_block1.dense_blocks.16     |   0.242M               |   97.542G  |   0.135G       |
|    dense_block1.dense_blocks.16.0  |    0.64K               |    0.258G  |    0           |
|    dense_block1.dense_blocks.16.2  |    0.103M              |    41.288G |    0.129G      |
|    dense_block1.dense_blocks.16.3  |    0.64K               |    0.258G  |    0           |
|    dense_block1.dense_blocks.16.5  |    0.138M              |    55.738G |    6.451M      |
|   dense_block1.dense_blocks.17     |   0.26M                |   0.105T   |   0.142G       |
|    dense_block1.dense_blocks.17.0  |    0.672K              |    0.271G  |    0           |
|    dense_block1.dense_blocks.17.2  |    0.113M              |    45.52G  |    0.135G      |
|    dense_block1.dense_blocks.17.3  |    0.672K              |    0.271G  |    0           |
|    dense_block1.dense_blocks.17.5  |    0.145M              |    58.525G |    6.451M      |
|   dense_block1.dense_blocks.18     |   0.278M               |   0.112T   |   0.148G       |
|    dense_block1.dense_blocks.18.0  |    0.704K              |    0.284G  |    0           |
|    dense_block1.dense_blocks.18.2  |    0.124M              |    49.958G |    0.142G      |
|    dense_block1.dense_blocks.18.3  |    0.704K              |    0.284G  |    0           |
|    dense_block1.dense_blocks.18.5  |    0.152M              |    61.312G |    6.451M      |
|   dense_block1.dense_blocks.19     |   0.296M               |   0.119T   |   0.155G       |
|    dense_block1.dense_blocks.19.0  |    0.736K              |    0.297G  |    0           |
|    dense_block1.dense_blocks.19.2  |    0.136M              |    54.603G |    0.148G      |
|    dense_block1.dense_blocks.19.3  |    0.736K              |    0.297G  |    0           |
|    dense_block1.dense_blocks.19.5  |    0.159M              |    64.099G |    6.451M      |
|   dense_block1.dense_blocks.20     |   0.315M               |   0.127T   |   0.161G       |
|    dense_block1.dense_blocks.20.0  |    0.768K              |    0.31G   |    0           |
|    dense_block1.dense_blocks.20.2  |    0.148M              |    59.454G |    0.155G      |
|    dense_block1.dense_blocks.20.3  |    0.768K              |    0.31G   |    0           |
|    dense_block1.dense_blocks.20.5  |    0.166M              |    66.886G |    6.451M      |
|  dense_block2                      |  1.065M                |  0.24T     |  0.364G        |
|   dense_block2.temporal_reduce1    |   0.335M               |   0.115T   |   0.166G       |
|    dense_block2.temporal_reduce1.0 |    0.8K                |    0.323G  |    0           |
|    dense_block2.temporal_reduce1.2 |    0.16M               |    64.512G |    0.161G      |
|    dense_block2.temporal_reduce1.3 |    0.8K                |    0.323G  |    0           |
|    dense_block2.temporal_reduce1.5 |    0.173M              |    49.766G |    4.608M      |
|   dense_block2.temporal_reduce2    |   0.355M               |   81.374G  |   0.123G       |
|    dense_block2.temporal_reduce2.0 |    0.832K              |    0.24G   |    0           |
|    dense_block2.temporal_reduce2.2 |    0.173M              |    49.84G  |    0.12G       |
|    dense_block2.temporal_reduce2.3 |    0.832K              |    0.24G   |    0           |
|    dense_block2.temporal_reduce2.5 |    0.18M               |    31.054G |    2.765M      |
|   dense_block2.temporal_reduce3    |   0.375M               |   43.297G  |   75.571M      |
|    dense_block2.temporal_reduce3.0 |    0.864K              |    0.149G  |    0           |
|    dense_block2.temporal_reduce3.2 |    0.187M              |    32.249G |    74.65M      |
|    dense_block2.temporal_reduce3.3 |    0.864K              |    0.149G  |    0           |
|    dense_block2.temporal_reduce3.5 |    0.187M              |    10.75G  |    0.922M      |
|  bn3d2                             |  0.896K                |  51.61M    |  0             |
|   bn3d2.weight                     |   (448,)               |            |                |
|   bn3d2.bias                       |   (448,)               |            |                |
|  conv3d2                           |  1.032M                |  59.454G   |  14.746M       |
|   conv3d2.weight                   |   (256, 448, 1, 3, 3)  |            |                |
|   conv3d2.bias                     |   (256,)               |            |                |
|  conv3d_r1                         |  65.792K               |  3.775G    |  14.746M       |
|   conv3d_r1.weight                 |   (256, 256, 1, 1, 1)  |            |                |
|   conv3d_r1.bias                   |   (256,)               |            |                |
|  conv3d_r2                         |  12.336K               |  0.708G    |  2.765M        |
|   conv3d_r2.weight                 |   (48, 256, 1, 1, 1)   |            |                |
|   conv3d_r2.bias                   |   (48,)                |            |                |
|  conv3d_f1                         |  0.132M                |  7.55G     |  29.491M       |
|   conv3d_f1.weight                 |   (512, 256, 1, 1, 1)  |            |                |
|   conv3d_f1.bias                   |   (512,)               |            |                |
|  conv3d_f2                         |  0.205M                |  11.796G   |  23.04M        |
|   conv3d_f2.weight                 |   (400, 512, 1, 1, 1)  |            |                |
|   conv3d_f2.bias                   |   (400,)               |            |                |
|  dynamic_filter                    |                        |  0.177G    |  7.085M        |
Output: torch.Size([1, 3, 720, 1280])
"""
