from typing import Tuple
import torch
from collections import OrderedDict

from lbasicsr.models.video_recurrent_model import VideoRecurrentModel
from lbasicsr.utils.logger import get_root_logger
from lbasicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class VideoINRModel(VideoRecurrentModel):
    
    def __init__(self, opt):
        super(VideoINRModel, self).__init__(opt)
        # self.net_base = opt['network_g']['type']
        if self.is_train:
            self.stage_div_iter = opt['train'].get('stage_division')
    
    def feed_data(self, data, need_GT=True):
        self.lq = data['lq'].to(self.device)                            # [b, 2, 3, h, w]
        # if ('time' in data.keys()) and self.net_base == 'LIIF':
        if 'time' in data.keys():
            self.times = [t_.to(self.device) for t_ in data['time']]    # list[tensor[0.3750]]
        else:
            self.times = None
        if 'gt_size' in data.keys():
            if isinstance(data['gt_size'], Tuple):
                self.gt_size = data['gt_size']
            else:
                # [tensor([128, 128, ... bs]), tensor([128, 128, ... bs])], for Adobe240Dataset 
                self.gt_size = (int(data['gt_size'][0][0]), int(data['gt_size'][1][0]))
        else:
            self.gt_size = None
        if 'test' in data.keys():
            self.testmode = data['test']
        else:
            self.testmode = False
        if 'scale' in data.keys():
            if isinstance(data['scale'], Tuple):
                self.scale = data['scale']
            elif isinstance(data['scale'], int) or isinstance(data['scale'], float):
                self.scale = (data['scale'], data['scale'])
            else:
                self.scale = (int(data['scale'][0]), int(data['scale'][0]))                      # for Adobe240Dataset
        if need_GT:
            self.gt = data['gt'].to(self.device)                        # [b, 1, 3, H, W]
    
    def optimize_parameters(self, current_iter):
        if self.stage_div_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Stage 1: fix scale={self.scale} for {self.stage_div_iter} iters.')
            elif current_iter == self.stage_div_iter:
                logger.info(f'Stage 2: arbitrary scale after {self.stage_div_iter} iters.')

        if hasattr(self, 'scale') and current_iter % 5000 == 0:
            print('current iteration scale: {}'.format(self.scale))
        # if hasattr(self, 'times'):
        #     print('current iteration times: {}'.format(self.times))
        
        self.optimizer_g.zero_grad()
        
        if self.times is None:      # network forward
            self.output = self.net_g(self.lq)
        # elif self.net_base == 'LIIF':
        #     self.output = self.net_g(self.lq, self.times, self.scale)
        else:
            self.output = self.net_g(self.lq, self.times, self.scale, self.gt_size)   # self.lq: [bs, 2, 3, h, w] --> self.output: [[bs, 3, H, W], ...] len=3

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
    
    def test(self):
        logger = get_root_logger()
        n = self.lq.size(1)     # the frame numbers of current video sequence
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            # set network's current scale
            # if hasattr(self.net_g, 'set_scale') and hasattr(self, 'scale'):
            #     self.net_g.set_scale(self.scale)
            # self.output = self.net_g(self.lq)
            logger.info(f'Arbitrary-scale STVSR model --> Arbitrary-scale VSR task.')
            self.output = self.vsr_test()

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()
    
    def vsr_test(self):
        time_scale = 1
        b, t, c, h, w = self.lq.size()
        
        outputs = []
        for i in range(t):
            img1 = self.lq[:, i, ...]                   # [bs, 3, h, w]
            img2 = self.lq[:, i, ...]
            imgs = torch.stack((img1, img2), dim=1)     # [bs, 2, 3, h, w]
            
            time_Tensors = [torch.tensor([i / time_scale])[None].to(imgs.device) for i in range(time_scale)]        # time_scale=1 ==> time_Tensors: [tensor([[0.]], device='cuda:0')]
            output = self.net_g(imgs, time_Tensors, self.scale, test=True)[0]
            
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)

    # TODO: STVSR test
    def stvsr_test(self):
        pass
    
    