from collections import OrderedDict
from os import path as osp
import math
import torch
from torch.nn import functional as F
from tqdm import tqdm
from lbasicsr.metrics import calculate_metric
from lbasicsr.utils import tensor2img, imwrite
from lbasicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class LIIFModel(SRModel):
    """LIIF model for single image super-resolution.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function
    """
    
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.coord = data['coord'].to(self.device)
        self.cell = data['cell'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'scale' in data:
            self.scale = data['scale']

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.coord, self.cell)

        # loss
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
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
    
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()
        
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
    
    def test(self):
        """Forward inference. Returns predictions of validation, testing, and
        simple inference.
        
        """
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.inference_mode():
                # NOTE: feats: shape [bz, N, 3]
                feats = self.net_g_ema(self.lq, self.coord, self.cell)
        else:
            self.net_g.eval()
            with torch.inference_mode():
                # NOTE: feats: shape [bz, N, 3]
                feats = self.net_g(self.lq, self.coord, self.cell)
            self.net_g.train()

        # reshape for eval, [bz, N, 3] -> [bz, 3, H, W]
        ih, iw = self.lq.shape[-2:]
        # metainfo in stacked data sample is a list, fetch by indexing
        coord_count = self.coord.shape[1]
        s = math.sqrt(coord_count / (ih * iw))
        shape = [1, round(ih * s), round(iw * s), 3]
        self.output = feats.view(shape).permute(0, 3, 1, 2).contiguous()
    
