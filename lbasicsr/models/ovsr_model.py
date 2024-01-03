from collections import OrderedDict, Counter
from os import path as osp

import math
import torch
from torch.cuda.amp import autocast, GradScaler

from lbasicsr.utils.registry import MODEL_REGISTRY
# from .video_base_model import VideoBaseModel
from .video_recurrent_model import VideoRecurrentModel


@MODEL_REGISTRY.register()
class OVSRModel(VideoRecurrentModel):
    """OVSR Model.

    """

    def __init__(self, opt):
        super(OVSRModel, self).__init__(opt)
        # if self.is_train:
        #     self.train_tsa_iter = opt['train'].get('tsa_iter')
        # self.output_last_fea = None
        if 'train' in self.opt:
            self.loss_frame_seq = list(range(self.opt['train']['sub_frame'], 
                                             self.opt['datasets']['train']['num_frame'] - self.opt['train']['sub_frame']))
            self.alpha = self.opt['train']['alpha']
            
            # Mixed Precision Training
            self.scaler = GradScaler()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        with autocast():
            self.output = self.net_g(self.lq, self.opt['train']['sub_frame'])

            l_total = 0
            loss_dict = OrderedDict()
            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(self.output[0], self.gt[:, self.loss_frame_seq, ...]) + \
                        self.alpha * self.cri_pix(self.output[1], self.gt[:, self.loss_frame_seq, ...])
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

                l_total_v = l_total.detach()
                if l_total_v > 5 or l_total_v < 0 or math.isnan(l_total_v):
                    raise RuntimeWarning(f'loss error {l_total_v}')

            # -------------------------------
            # Mix Precision Training
            # l_total.backward()
            # self.optimizer_g.step()
            # -->
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
            # -------------------------------
            
            self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    # def optimize_parameters(self, current_iter):
    #     self.optimizer_g.zero_grad()
        
    #     self.output = self.net_g(self.lq, self.opt['train']['sub_frame'])

    #     l_total = 0
    #     loss_dict = OrderedDict()
    #     # pixel loss
    #     if self.cri_pix:
    #         l_pix = self.cri_pix(self.output[0], self.gt[:, self.loss_frame_seq, ...]) + \
    #                 self.alpha * self.cri_pix(self.output[1], self.gt[:, self.loss_frame_seq, ...])
    #         l_total += l_pix
    #         loss_dict['l_pix'] = l_pix

    #         l_total_v = l_total.detach()
    #         if l_total_v > 5 or l_total_v < 0 or math.isnan(l_total_v):
    #             raise RuntimeWarning(f'loss error {l_total_v}')

    #     l_total.backward()
    #     self.optimizer_g.step()
        
    #     self.log_dict = self.reduce_loss_dict(loss_dict)

    #     if self.ema_decay > 0:
    #         self.model_ema(decay=self.ema_decay)

    def test(self):
        n = self.lq.size(1)     # 当前视频序列帧数
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            self.output = self.net_g(self.lq)
            if isinstance(self.output, tuple):
                self.output = self.output[0]

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()
