from torch import nn as nn

from lbasicsr.data.core import imresize
from lbasicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class BicubicVSR(nn.Module):
    """

    """
    def __init__(self, scale=4):
        super(BicubicVSR, self).__init__()
        self.scale = scale

    def forward(self, x):
        b, t, c, h, w = x.size()

        x = x.view(-1, c, h, w)
        x = imresize(x, scale=self.scale)
        x = x.view(b, t, c, x.size(-2), x.size(-1))

        return x
