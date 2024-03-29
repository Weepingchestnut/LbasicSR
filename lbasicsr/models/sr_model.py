from collections import OrderedDict
from os import path as osp

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from lbasicsr.archs import build_network
from lbasicsr.data.core import imresize
from lbasicsr.losses import build_loss
from lbasicsr.metrics import calculate_metric
from lbasicsr.models.base_model import BaseModel
from lbasicsr.utils import get_root_logger, tensor2img, imwrite
from lbasicsr.utils.registry import MODEL_REGISTRY
from .sam import SAM


@MODEL_REGISTRY.register()  # 注册SRModel
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    # ========================================
    # 初始化 SRModel类，如定义网络和 load weight
    # ========================================
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # 初始化训练相关的设置
        if self.is_train:
            self.init_training_settings()

    # =========================================================
    # 初始化与训练相关的配置，如 loss，设置 optimizers 和 schedulers
    # =========================================================
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            # =================================================================
            # 根据配置文件yml中的loss类型和参数，实例化loss
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            # =================================================================
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    # =============================================================
    # 具体设置 optimizer，可根据实际需求，对params设置多组不同的optimizer
    # =============================================================
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    # =========================================
    # 提供数据，是与dataloder（dataset）的接口
    # =========================================
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        # print('self.lq shape:', self.lq.shape)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        # for arbitrary-scale
        if 'scale' in data:
            self.scale = data['scale']
            # self.scale = (data['scale'][0].to(self.device), data['scale'][1].to(self.device))

    # =======================================================================
    # 优化参数，即一个完整的 train step，包括forward，loss计算，backward，参数优化等
    # =======================================================================
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()    # 使得 optimizer 中的梯度归零
        self.output = self.net_g(self.lq)   # network forward

        # ===========================================================================
        # loss 的计算
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
        # ===========================================================================

        l_total.backward()
        self.optimizer_g.step()     # 优化器更新

        # 为了 loss 的显示，同时也同步多卡上的loss
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # ====================================
    # 测试流程
    # ====================================
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)   # network influence
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    # ==================================
    # validation 的流程（多卡 dist）
    # ==================================
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    # ==================================
    # validation 的流程（单卡 non-dist）
    # ==================================
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

        # =========================================================================
        for idx, val_data in enumerate(dataloader):
            if val_data.get('lq_path', None):
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            else:
                img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            # 喂测试数据
            self.feed_data(val_data)
            # 测试
            self.test()

            # 得到测试结果
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
                    # 根据配置文件yml中metrics的配置，调用相应的函数
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()
        # =========================================================================

        # ========================================================================================================
        # 显示 metrics 的结果
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        # ========================================================================================================

    # =============================================
    # 控制如何打印 validation 的结果
    # =============================================
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    # ==================================================================
    # 得到网络的输出结果。该函数会在 validation 中用到（实际可以简化掉）
    # ==================================================================
    def get_current_visuals(self):
        logger = get_root_logger()
        # only h, w
        # logger.info("gt: ({}, {})".format(self.gt.size(-2), self.gt.size(-1)))
        # logger.info("lq: ({}, {})".format(self.lq.size(-2), self.lq.size(-1)))
        # logger.info("output: ({}, {})".format(self.output.size(-2), self.output.size(-1)))
        # all shape
        logger.info("gt: {}".format(self.gt.shape))
        logger.info("lq: {}".format(self.lq.shape))
        logger.info("output: {}".format(self.output.shape))

        # arbitrary-scale BI post-processing
        if self.output.ndim == 4 and self.output.shape != self.gt.shape:
            logger.info('arbitrary-scale SR, use BI post-process ......')
            self.output = T.Resize(size=(self.gt.size(-2), self.gt.size(-1)), interpolation=InterpolationMode.BICUBIC,
                         antialias=True)(self.output)
            # self.output = imresize(self.output, sizes=(self.gt.size(-2), self.gt.size(-1)))
        if self.output.ndim == 5 and self.output.shape != self.gt.shape:
            # logger.info('BI resize ......')
            logger.info('arbitrary-scale SR, use BI post-process ......')
            b, t, c, h, w = self.output.size()
            self.output = self.output.view(-1, c, h, w)
            # self.output = imresize(self.output, sizes=(self.gt.size(-2), self.gt.size(-1)))
            self.output = T.Resize(size=(self.gt.size(-2), self.gt.size(-1)), interpolation=InterpolationMode.BICUBIC,
                                   antialias=True)(self.output)
            self.output = self.output.view(b, t, c, self.output.size(-2), self.output.size(-1))
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
    
    def get_current_visuals_nologger(self):
        print("gt: ({}, {})".format(self.gt.size(-2), self.gt.size(-1)))
        print("lq: ({}, {})".format(self.lq.size(-2), self.lq.size(-1)))
        print("output: ({}, {})".format(self.output.size(-2), self.output.size(-1)))

        # arbitrary-scale BI post-processing
        if self.output.ndim == 4 and self.output.shape != self.gt.shape:
            print('arbitrary-scale SR, use BI post-process ......')
            self.output = T.Resize(size=(self.gt.size(-2), self.gt.size(-1)), interpolation=InterpolationMode.BICUBIC,
                         antialias=True)(self.output)
            # self.output = imresize(self.output, sizes=(self.gt.size(-2), self.gt.size(-1)))
        if self.output.ndim == 5 and self.output.shape != self.gt.shape:
            # logger.info('BI resize ......')
            print('arbitrary-scale SR, use BI post-process ......')
            b, t, c, h, w = self.output.size()
            self.output = self.output.view(-1, c, h, w)
            # self.output = imresize(self.output, sizes=(self.gt.size(-2), self.gt.size(-1)))
            self.output = T.Resize(size=(self.gt.size(-2), self.gt.size(-1)), interpolation=InterpolationMode.BICUBIC,
                                   antialias=True)(self.output)
            self.output = self.output.view(b, t, c, self.output.size(-2), self.output.size(-1))
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    # =========================================================
    # 保存网络（.pth 文件）以及训练状态（.state 文件）
    # =========================================================
    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()  # 注册SRModel
class SRModel_SAM(SRModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel_SAM, self).__init__(opt)
        
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        # self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        # SGD
        # base_optimizer = torch.optim.SGD
        # self.optimizer_g = SAM(optim_params, base_optimizer, lr=train_opt['optim_g']['lr'], momentum=train_opt['optim_g']['momentum'])
        # Adam
        base_optimizer = torch.optim.Adam
        self.optimizer_g = SAM(optim_params, base_optimizer, rho=0.05, adaptive=False, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
    
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()    # 使得 optimizer 中的梯度归零
        self.output = self.net_g(self.lq)   # network forward

        # ===========================================================================
        # loss 的计算
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
        # ===========================================================================

        # ------ for SAM ---------------------------------------------- 
        l_total.backward()
        self.optimizer_g.first_step(zero_grad=True)
        
        # l_total.backward()
        l_total1 = l_total.detach_().requires_grad_(True)
        l_total1.backward()
        self.optimizer_g.second_step(zero_grad=True)
        # -------------------------------------------------------------

        # 为了 loss 的显示，同时也同步多卡上的loss
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
