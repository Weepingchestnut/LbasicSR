from collections import OrderedDict, Counter
from os import path as osp

import math
import torch
from torch import distributed as dist
from tqdm import tqdm

from lbasicsr.metrics import calculate_metric
from lbasicsr.utils import imwrite, tensor2img
from lbasicsr.utils.dist_util import get_dist_info
from lbasicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class OVSRModel(VideoBaseModel):
    """OVSR Model.

    """

    def __init__(self, opt):
        super(OVSRModel, self).__init__(opt)
        # if self.is_train:
        #     self.train_tsa_iter = opt['train'].get('tsa_iter')
        # self.output_last_fea = None
        if 'train' in self.opt:
            self.loss_frame_seq = list(range(self.opt['train']['sub_frame'], 7 - self.opt['train']['sub_frame']))
            self.alpha = self.opt['train']['alpha']

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

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

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # def test(self):
    #     if hasattr(self, 'net_g_ema'):
    #         self.net_g_ema.eval()
    #         with torch.no_grad():
    #             self.output = self.net_g_ema(self.lq)
    #             if isinstance(self.output, tuple):
    #                 self.output = self.output[0]
    #     else:
    #         self.net_g.eval()
    #         with torch.no_grad():
    #             self.output = self.net_g(self.lq)  # network influence
    #             if isinstance(self.output, tuple):
    #                 self.output = self.output[0]
    #         self.net_g.train()

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
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {}
                num_frame_each_folder = Counter(dataset.data_info['folder'])
                for folder, num_frame in num_frame_each_folder.items():
                    self.metric_results[folder] = torch.zeros(
                        num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
            # initialize the best metric results
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        metric_data = dict()
        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')
        # Will evaluate (num_folders + num_pad) times, but only the first num_folders results will be recorded.
        # (To avoid wait-dead)
        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']

            # compute outputs
            val_data['lq'].unsqueeze_(0)    # torch.Size([1, 41, 3, 144, 180])
            val_data['gt'].unsqueeze_(0)    # torch.Size([1, 41, 3, 576, 720])
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)      # torch.Size([41, 3, 144, 180])
            val_data['gt'].squeeze_(0)      # torch.Size([41, 3, 576, 720])

            self.test()
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            if self.center_frame_only:
                visuals['result'] = visuals['result'].unsqueeze(1)
                if 'gt' in visuals:
                    visuals['gt'] = visuals['gt'].unsqueeze(1)

            # evaluate
            if i < num_folders:
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    result_img = tensor2img([result])  # uint8, bgr
                    metric_data['img'] = result_img
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        gt_img = tensor2img([gt])  # uint8, bgr
                        metric_data['img2'] = gt_img

                    if save_img:
                        if self.opt['is_train']:
                            raise NotImplementedError('saving image is not supported during training.')
                        else:
                            if self.center_frame_only:  # vimeo-90k
                                clip_ = val_data['lq_path'].split('/')[-3]
                                seq_ = val_data['lq_path'].split('/')[-2]
                                name_ = f'{clip_}_{seq_}'
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                    f"{name_}_{self.opt['name']}.png")
                            else:  # others
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                    f"{idx:08d}_{self.opt['name']}.png")
                            # image name only for REDS dataset
                        imwrite(result_img, img_path)

                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            result = calculate_metric(metric_data, opt_)
                            self.metric_results[folder][idx, metric_idx] += result

                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
