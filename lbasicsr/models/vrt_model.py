import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from collections import Counter, OrderedDict
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm

from lbasicsr.metrics import calculate_metric
from lbasicsr.utils import get_root_logger, imwrite, tensor2img
from lbasicsr.utils.dist_util import get_dist_info
from lbasicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class VRTModel(VideoBaseModel):

    def __init__(self, opt):
        super(VRTModel, self).__init__(opt)
        if self.is_train:
            self.fix_iter = opt['train'].get('fix_iter', 0)
            self.fix_keys = opt['train'].get('fix_keys', [])
            self.fix_unflagged = True

    def setup_optimizers(self):
        train_opt = self.opt['train']
        if self.is_train:
            self.fix_iter = train_opt.get('fix_iter', 0)
            self.fix_keys = train_opt.get('fix_keys', [])
            self.fix_unflagged = True
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if any([key in name for key in self.fix_keys]):
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.fix_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix keys: {self.fix_keys} for {self.fix_iter} iters.')
                self.fix_unflagged = False
                for name, param in self.net_g.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        param.requires_grad_(False)
            elif current_iter == self.fix_iter:
                logger.warning(f'Train all the parameters from {self.fix_iter} iters.')
                self.net_g.requires_grad_(True)

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)   # network forward
        
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

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        if self.opt.get('scale', 0) == 0:
            # only support symmetric scale
            if isinstance(dataset.opt['downsampling_scale'], tuple):
                self.opt['scale'] = dataset.opt['downsampling_scale'][0]
            else:
                self.opt['scale'] = dataset.opt['downsampling_scale']
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
    
    def test(self):
        n = self.lq.size(1)     # the number of frames in current video clip
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            self.output = self._test_video(self.lq)

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()
    
    def _test_video(self, lq):
        '''test the video as a whole or as clips (divided temporally). '''

        num_frame_testing = self.opt['val'].get('num_frame_testing', 0)

        if num_frame_testing:
            # test as multiple clips if out-of-memory
            # sf = self.opt['scale']
            if isinstance(self.opt['scale'], tuple):
                sf = self.opt['scale'][0]
            else:
                sf = self.opt['scale']
            psf = self.opt['val'].get('pre_scale')      # pretrained model upscale
            num_frame_overlapping = self.opt['val'].get('num_frame_overlapping', 2)
            not_overlap_border = False
            b, d, c, h, w = lq.size()
            c = c - 1 if self.opt['network_g'].get('nonblind_denoising', False) else c
            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
            if psf:   # specified scale pretrained model
                E = torch.zeros(b, d, c, h*psf, w*psf)
            elif sf in [2, 3, 4]:
                E = torch.zeros(b, d, c, h*sf, w*sf)
            else:   # if you have x2, x3 pretrained model
                E = torch.zeros(b, d, c, h * (int(sf)+1), w * (int(sf)+1))
            W = torch.zeros(b, d, 1, 1, 1)

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
                out_clip = self._test_clip(lq_clip)
                out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

                if not_overlap_border:
                    if d_idx < d_idx_list[-1]:
                        out_clip[:, -num_frame_overlapping//2:, ...] *= 0
                        out_clip_mask[:, -num_frame_overlapping//2:, ...] *= 0
                    if d_idx > d_idx_list[0]:
                        out_clip[:, :num_frame_overlapping//2, ...] *= 0
                        out_clip_mask[:, :num_frame_overlapping//2, ...] *= 0

                E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
                W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)
            output = E.div_(W)
        else:
            # test as one clip (the whole video) if you have enough memory
            window_size = self.opt['network_g'].get('window_size', [6,8,8])
            d_old = lq.size(1)
            d_pad = (d_old // window_size[0] + 1) * window_size[0] - d_old
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1)
            output = self._test_clip(lq)
            output = output[:, :d_old, :, :, :]

        return output

    def _test_clip(self, lq):
        ''' test the clip as a whole or as patches. '''

        if isinstance(self.opt['scale'], tuple):
            sf = self.opt['scale'][0]
        else:
            sf = self.opt['scale']
        psf = self.opt['val'].get('pre_scale')      # pretrained model upscale
        window_size = self.opt['network_g'].get('window_size', [6, 8, 8])
        size_patch_testing = self.opt['val'].get('size_patch_testing', 0)
        assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

        if size_patch_testing:
            # divide the clip to patches (spatially only, tested patch by patch)
            overlap_size = self.opt['val'].get('overlap_size', 20)
            not_overlap_border = True

            # test patch by patch
            b, d, c, h, w = lq.size()
            c = c - 1 if self.opt['network_g'].get('nonblind_denoising', False) else c
            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
            w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
            
            if psf:    # specified scale pretrained model
                E = torch.zeros(b, d, c, h*psf, w*psf)
            elif sf in [2, 3, 4]:
                E = torch.zeros(b, d, c, h*sf, w*sf)
            else:
                E = torch.zeros(b, d, c, h * (int(sf)+1), w * (int(sf)+1))
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                    if hasattr(self, 'netE'):
                        out_patch = self.netE(in_patch).detach().cpu()
                    else:
                        out_patch = self.net_g(in_patch).detach().cpu()     # torch.Size([1, 32, 3, 512, 512])

                    out_patch_mask = torch.ones_like(out_patch)

                    if not_overlap_border:
                        if h_idx < h_idx_list[-1]:
                            out_patch[..., -overlap_size//2:, :] *= 0
                            out_patch_mask[..., -overlap_size//2:, :] *= 0
                        if w_idx < w_idx_list[-1]:
                            out_patch[..., :, -overlap_size//2:] *= 0
                            out_patch_mask[..., :, -overlap_size//2:] *= 0
                        if h_idx > h_idx_list[0]:
                            out_patch[..., :overlap_size//2, :] *= 0
                            out_patch_mask[..., :overlap_size//2, :] *= 0
                        if w_idx > w_idx_list[0]:
                            out_patch[..., :, :overlap_size//2] *= 0
                            out_patch_mask[..., :, :overlap_size//2] *= 0
                    if psf:    # specified scale pretrained model
                        E[..., h_idx*psf:(h_idx+size_patch_testing)*psf, w_idx*psf:(w_idx+size_patch_testing)*psf].add_(out_patch)
                        W[..., h_idx*psf:(h_idx+size_patch_testing)*psf, w_idx*psf:(w_idx+size_patch_testing)*psf].add_(out_patch_mask)
                    elif sf in [2, 3, 4]:
                        E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch)
                        W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch_mask)
                    else:
                        E[..., h_idx * (int(sf)+1):(h_idx+size_patch_testing) * (int(sf)+1), 
                               w_idx * (int(sf)+1):(w_idx+size_patch_testing) * (int(sf)+1)].add_(out_patch)
                        W[..., h_idx * (int(sf)+1):(h_idx+size_patch_testing) * (int(sf)+1), 
                               w_idx * (int(sf)+1):(w_idx+size_patch_testing) * (int(sf)+1)].add_(out_patch_mask)
            output = E.div_(W)

        else:
            _, _, _, h_old, w_old = lq.size()
            # h_pad = (h_old// window_size[1]+1)*window_size[1] - h_old   # (144 // 8 + 1) * 8 - 144 = 8
            # w_pad = (w_old// window_size[2]+1)*window_size[2] - w_old   # (180 // 8 + 1) * 8 - 180 = 4
            # ref RVRT test
            h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]      # (8 - 144 % 8) % 8 = 0
            w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]      # (8 - 180 % 8) % 8 = 4

            lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
            lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq
            
            output = self.net_g(lq)
            # for arbitrary-scale VSR, use BI post-process
            if psf:
                output = output[:, :, :, :h_old*psf, :w_old*psf]
            elif sf in [2, 3, 4]:
                output = output[:, :, :, :h_old*sf, :w_old*sf]
            else:
                output = output[:, :, :, :h_old * (int(sf)+1), :w_old * (int(sf)+1)]
                

        return output
