from collections import OrderedDict, Counter
from os import path as osp

import torch
from torch import distributed as dist
from tqdm import tqdm

from lbasicsr.metrics import calculate_metric
from lbasicsr.utils import imwrite, tensor2img
from lbasicsr.utils.dist_util import get_dist_info
from lbasicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel
from ..data.data_util import generate_frame_indices


@MODEL_REGISTRY.register()
class STANModel(VideoBaseModel):
    """STAN Model.

    """

    def __init__(self, opt):
        super(STANModel, self).__init__(opt)
        if self.is_train:
            self.train_tsa_iter = opt['train'].get('tsa_iter')
        self.output_last_fea = None

    def optimize_parameters(self, current_iter):
        # if current_iter == 1:
        #     output_last_fea = None
        self.optimizer_g.zero_grad()
        l_total = 0
        loss_dict = OrderedDict()

        if hasattr(self, 'scale'):
            self.net_g.set_scale(self.scale)
            print('current iteration scale: {}'.format(self.scale))
        output_last_fea = None
        # ================================================================
        for idx in range(7):
            select_idx = generate_frame_indices(idx, 7, 7, padding='reflection')
            imgs_lq = [self.lq[:, i, ...] for i in select_idx]
            imgs_lq = torch.stack(imgs_lq, dim=1)
            img_gt = self.gt[:, idx, ...]

            self.output, out_fea = self.net_g(imgs_lq, output_last_fea)
            output_last_fea = out_fea.detach()

            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(self.output, img_gt)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix
            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, img_gt)
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
        if hasattr(self, 'net_g_ema'):      # self.lq: torch.Size([1, 7, 3, 144, 180]);
            self.net_g_ema.set_scale(self.opt['scale'])
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, out_fea = self.net_g_ema(self.lq, self.output_last_fea)
                self.output_last_fea = out_fea.detach()
        else:
            if self.opt['is_train']:
                if self.opt['datasets']['val'].__contains__('downsampling_scale'):
                    self.net_g.set_scale(self.opt['datasets']['val']['downsampling_scale'])
            else:
                self.net_g.set_scale(self.opt['scale'])
            self.net_g.eval()
            with torch.no_grad():
                self.output, out_fea = self.net_g(self.lq, self.output_last_fea)   # network influence
                self.output_last_fea = out_fea.detach()
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        if with_metrics:
            if not hasattr(self, 'metric_results') or self.metric_results:  # only execute in the first run X
                self.metric_results = {}
                num_frame_each_folder = Counter(dataset.data_info['folder'])
                for folder, num_frame in num_frame_each_folder.items():
                    self.metric_results[folder] = torch.zeros(
                        num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')     # 记录每一帧的指标
            # initialize the best metric results
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        metric_data = dict()
        # record all frames (border and center frames)
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='frame')
        for idx in range(rank, len(dataset), world_size):
            val_data = dataset[idx]     # dict{'lq': Tensor TCHW, 'gt': Tensor CHW, 'folder': 'calendar', 'idx': '0/41', border: 1, 'lq_path': str}
            val_data['lq'].unsqueeze_(0)
            # print("\nlq:({}, {})".format(val_data['lq'].size(-2), val_data['lq'].size(-1)))
            val_data['gt'].unsqueeze_(0)
            # print("gt:({}, {})".format(val_data['gt'].size(-2), val_data['gt'].size(-1)))
            folder = val_data['folder']
            frame_idx, max_idx = val_data['idx'].split('/')
            if frame_idx == '0':
                self.output_last_fea = None
            lq_path = val_data['lq_path']

            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            result_img = tensor2img([visuals['result']])
            metric_data['img'] = result_img
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
                    raise NotImplementedError('saving image is not supported during training.')
                else:
                    if 'vimeo' in dataset_name.lower():  # vimeo90k dataset
                        split_result = lq_path.split('/')
                        img_name = f'{split_result[-3]}_{split_result[-2]}_{split_result[-1].split(".")[0]}'
                    else:  # other datasets, e.g., REDS, Vid4
                        img_name = osp.splitext(osp.basename(lq_path))[0]

                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(result_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                    result = calculate_metric(metric_data, opt_)
                    self.metric_results[folder][int(frame_idx), metric_idx] += result

            # progress bar
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {folder}: {int(frame_idx) + world_size}/{max_idx}')
        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()
            else:
                pass  # assume use one gpu in non-dist testing

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
