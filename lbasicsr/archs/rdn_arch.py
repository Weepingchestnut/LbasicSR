import torch
from torch import nn as nn

from lbasicsr.archs.arch_util import Upsample
from lbasicsr.utils.registry import ARCH_REGISTRY


class DenseLayer(nn.Module):
    """Dense layer.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c_in, h, w).

        Returns:
            Tensor: Forward results, tensor with shape (n, c_in+c_out, h, w).
        """
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    """Residual Dense Block of Residual Dense Network.

    Args:
        in_channels (int): Channel number of inputs.
        channel_growth (int): Channels growth in each layer.
        num_layers (int): Layer number in the Residual Dense Block.
    """

    def __init__(self, in_channels, channel_growth, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[
            DenseLayer(in_channels + channel_growth * i, channel_growth)
            for i in range(num_layers)
        ])

        # local feature fusion
        self.lff = nn.Conv2d(
            in_channels + channel_growth * num_layers,
            in_channels,
            kernel_size=1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return x + self.lff(self.layers(x))  # local residual learning


@ARCH_REGISTRY.register()
class RDN(nn.Module):
    """RDN model for single image super-resolution.

    Paper: Residual Dense Network for Image Super-Resolution

    Adapted from 'https://github.com/yjn870/RDN-pytorch.git'
    'RDN-pytorch/blob/master/models.py'
    Copyright (c) 2021, JaeYun Yeo, under MIT License.

    Most of the implementation follows the implementation in:
    'https://github.com/sanghyun-son/EDSR-PyTorch.git'
    'EDSR-PyTorch/blob/master/src/model/rdn.py'
    Copyright (c) 2017, sanghyun-son, under MIT license.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_blocks (int): Block number in the trunk network. Default: 16.
        upscale_factor (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        num_layer (int): Layer number in the Residual Dense Block.
            Default: 8.
        channel_growth(int): Channels growth in each layer of RDB.
            Default: 64.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 num_layer=8,
                 channel_growth=64):
        super(RDN, self).__init__()
        self.num_feat = num_feat
        self.channel_growth = channel_growth
        self.num_block = num_block
        self.num_layer = num_layer
        
        # shallow feature extraction
        self.sfe1 = nn.Conv2d(
            num_in_ch, num_feat, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(
            num_feat, num_feat, kernel_size=3, padding=3 // 2)
        
        # residual dense blocks
        self.rdbs = nn.ModuleList()
        for _ in range(self.num_block):
            self.rdbs.append(
                RDB(self.num_feat, self.channel_growth, self.num_layer))
        
        
        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(
                self.num_feat * self.num_block,
                self.num_feat,
                kernel_size=1),
            nn.Conv2d(
                self.num_feat,
                self.num_feat,
                kernel_size=3,
                padding=3 // 2))

        # up-sampling
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.num_block):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1
        # global residual learning
        x = self.upsample(x)
        x = self.conv_last(x)
        return x


if __name__ == '__main__':
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    x = torch.randn(1, 3, 320, 180)
    # x = torch.randn(1, 3, 256, 256)

    model = RDN()
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)


"""
params: 22271107
| module               | #parameters or shape   | #flops     | #activations   |
|:---------------------|:-----------------------|:-----------|:---------------|
| model                | 22.271M                | 1.309T     | 0.622G         |
|  sfe1                |  1.792K                |  99.533M   |  3.686M        |
|   sfe1.weight        |   (64, 3, 3, 3)        |            |                |
|   sfe1.bias          |   (64,)                |            |                |
|  sfe2                |  36.928K               |  2.123G    |  3.686M        |
|   sfe2.weight        |   (64, 64, 3, 3)       |            |                |
|   sfe2.bias          |   (64,)                |            |                |
|  rdbs                |  21.833M               |  1.257T    |  0.531G        |
|   rdbs.0             |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.0.layers     |    1.328M              |    76.441G |    29.491M     |
|    rdbs.0.lff        |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.1             |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.1.layers     |    1.328M              |    76.441G |    29.491M     |
|    rdbs.1.lff        |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.2             |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.2.layers     |    1.328M              |    76.441G |    29.491M     |
|    rdbs.2.lff        |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.3             |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.3.layers     |    1.328M              |    76.441G |    29.491M     |
|    rdbs.3.lff        |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.4             |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.4.layers     |    1.328M              |    76.441G |    29.491M     |
|    rdbs.4.lff        |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.5             |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.5.layers     |    1.328M              |    76.441G |    29.491M     |
|    rdbs.5.lff        |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.6             |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.6.layers     |    1.328M              |    76.441G |    29.491M     |
|    rdbs.6.lff        |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.7             |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.7.layers     |    1.328M              |    76.441G |    29.491M     |
|    rdbs.7.lff        |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.8             |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.8.layers     |    1.328M              |    76.441G |    29.491M     |
|    rdbs.8.lff        |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.9             |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.9.layers     |    1.328M              |    76.441G |    29.491M     |
|    rdbs.9.lff        |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.10            |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.10.layers    |    1.328M              |    76.441G |    29.491M     |
|    rdbs.10.lff       |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.11            |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.11.layers    |    1.328M              |    76.441G |    29.491M     |
|    rdbs.11.lff       |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.12            |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.12.layers    |    1.328M              |    76.441G |    29.491M     |
|    rdbs.12.lff       |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.13            |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.13.layers    |    1.328M              |    76.441G |    29.491M     |
|    rdbs.13.lff       |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.14            |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.14.layers    |    1.328M              |    76.441G |    29.491M     |
|    rdbs.14.lff       |    36.928K             |    2.123G  |    3.686M      |
|   rdbs.15            |   1.365M               |   78.565G  |   33.178M      |
|    rdbs.15.layers    |    1.328M              |    76.441G |    29.491M     |
|    rdbs.15.lff       |    36.928K             |    2.123G  |    3.686M      |
|  gff                 |  0.103M                |  5.898G    |  7.373M        |
|   gff.0              |   65.6K                |   3.775G   |   3.686M       |
|    gff.0.weight      |    (64, 1024, 1, 1)    |            |                |
|    gff.0.bias        |    (64,)               |            |                |
|   gff.1              |   36.928K              |   2.123G   |   3.686M       |
|    gff.1.weight      |    (64, 64, 3, 3)      |            |                |
|    gff.1.bias        |    (64,)               |            |                |
|  upsample            |  0.295M                |  42.467G   |  73.728M       |
|   upsample.0         |   0.148M               |   8.493G   |   14.746M      |
|    upsample.0.weight |    (256, 64, 3, 3)     |            |                |
|    upsample.0.bias   |    (256,)              |            |                |
|   upsample.2         |   0.148M               |   33.974G  |   58.982M      |
|    upsample.2.weight |    (256, 64, 3, 3)     |            |                |
|    upsample.2.bias   |    (256,)              |            |                |
|  conv_last           |  1.731K                |  1.593G    |  2.765M        |
|   conv_last.weight   |   (3, 64, 3, 3)        |            |                |
|   conv_last.bias     |   (3,)                 |            |                |
torch.Size([1, 3, 1280, 720])
"""
