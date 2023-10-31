import torch
from collections import OrderedDict

from lbasicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class VideoINRModel(VideoBaseModel):
    
    def __init__(self, opt):
        super(VideoINRModel, self).__init__(opt)
        # self.net_base = opt['network_g']['type']
    
    def feed_data(self, data, need_GT=True):
        self.lq = data['lq'].to(self.device)
        # if ('time' in data.keys()) and self.net_base == 'LIIF':
        if ('time' in data.keys()):
            self.times = [t_.to(self.device) for t_ in data['time']]
        else:
            self.times = None
        if 'scale' in data.keys():
            self.scale = data['scale']
        else:
            self.scale = None
        if 'test' in data.keys():
            self.testmode = data['test']
        else:
            self.testmode = False
        if need_GT:
            self.gt = data['gt'].to(self.device)
    
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        if self.times is None:      # network forward
            self.output = self.net_g(self.lq)
        # elif self.net_base == 'LIIF':
        #     self.output = self.net_g(self.lq, self.times, self.scale)
        else:
            self.output = self.net_g(self.lq, self.times, self.scale)

        # loss
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            if self.times is None:
                l_pix = self.cri_pix(self.output, self.gt)
            else:
                l_pix = 0
                for idx in range(len(self.times)):
                    l_pix += self.cri_pix(self.output[idx], self.gt[:, idx])
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        
    def test(self, output=False):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                if self.times is None:
                    self.output = self.net_g(self.lq)   # network influence
                # elif self.net_base == 'LIIF':
                else:
                    self.output = self.net_g(self.lq, self.times, self.scale, self.testmode)
            self.net_g.train()
            if output is True:
                return self.output
    
    