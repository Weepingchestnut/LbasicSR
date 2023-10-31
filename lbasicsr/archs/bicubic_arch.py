import torchvision.transforms
from torch import nn as nn
from torchvision.transforms import InterpolationMode

from lbasicsr.data.core import imresize
from lbasicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class BicubicVSR(nn.Module):
    """Bicubic

    """
    def __init__(self, scale=4, bicubic_mode='core'):
        super(BicubicVSR, self).__init__()
        self.scale = scale
        self.bicubic_mode = bicubic_mode
    
    def set_scale(self, scale):
        self.scale = scale

    def forward(self, x):
        b, t, c, h, w = x.size()

        x = x.view(-1, c, h, w)
        if self.bicubic_mode == 'core':
            x = imresize(x, sizes=(round(h*self.scale[0]), round(w*self.scale[1])))
        elif self.bicubic_mode == 'torch':
            x = torchvision.transforms.Resize(size=(round(h*self.scale[0]), round(w*self.scale[1])),
                                              interpolation=InterpolationMode.BICUBIC,
                                              antialias=True)(x)
        x = x.view(b, t, c, x.size(-2), x.size(-1))

        return x
