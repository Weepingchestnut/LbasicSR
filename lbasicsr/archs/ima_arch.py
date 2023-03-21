import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
import torch.nn.init as init

from lbasicsr.archs.arch_util import make_layer


class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, 90, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(90, nf, 3, 1, 1, bias=True)
        self.scale = torch.nn.Parameter(torch.FloatTensor([0.5]))

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=False)
        out = self.conv2(out)
        return identity + self.scale * out


class encoder(nn.Module):
    def __init__(self, nf=64, N_RB=5):
        super(encoder, self).__init__()
        RB_f = functools.partial(ResidualBlock_noBN, nf=nf)

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.rbs = make_layer(RB_f, N_RB)

        self.d2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.d2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.d4_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.d4_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.py_conv = nn.Conv2d(nf * 3, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        fea_lr = self.rbs(fea)

        fea_d2 = self.lrelu(self.d2_conv2(self.lrelu(self.d2_conv1(fea_lr))))
        fea_d4 = self.lrelu(self.d4_conv2(self.lrelu(self.d4_conv1(fea_d2))))

        fea_d2 = F.interpolate(fea_d2, size=(x.size()[-2], x.size()[-1]), mode='bilinear', align_corners=False)
        fea_d4 = F.interpolate(fea_d4, size=(x.size()[-2], x.size()[-1]), mode='bilinear', align_corners=False)

        out = self.lrelu(self.py_conv(torch.cat([fea_lr, fea_d2, fea_d4], 1)))



