import torch

from .video_base_model import VideoBaseModel
from ..utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ASVSRModel(VideoBaseModel):
    """ASVSR Model

    """

    def __init__(self, opt):
        super(ASVSRModel, self).__init__(opt)

    def optimize_parameters(self, current_iter):
        if hasattr(self, 'scale'):
            self.net_g.set_scale(self.scale)
            print('current iteration scale: {}'.format(self.scale))

        super(ASVSRModel, self).optimize_parameters(current_iter)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.set_scale(self.opt['scale'])
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            if self.opt['is_train']:
                if self.opt['datasets']['val'].__contains__('downsampling_scale'):
                    self.net_g.set_scale(self.opt['datasets']['val']['downsampling_scale'])
            else:
                self.net_g.set_scale(self.opt['scale'])
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)   # network influence
            self.net_g.train()


