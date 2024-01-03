from abc import abstractmethod
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

# from lbasicsr.archs import build_network
# from archs import build_network
from lbasicsr.archs.arch_util import make_coord
from lbasicsr.utils.logger import get_root_logger
from lbasicsr.utils.registry import ARCH_REGISTRY


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    # ==================================================
    # 根据配置文件yml中的 arch 类型，创建相应的 network 实例
    net = ARCH_REGISTRY.get(network_type)(**opt)
    # ==================================================
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net


class LIIF(nn.Module):
    """LIIF net for single image super-resolution, CVPR, 2021.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    The subclasses should define `generator` with `encoder` and `imnet`,
        and overwrite the function `gen_feature`.
    If `encoder` does not contain `mid_channels`, `__init__` should be
        overwrite.

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """
    
    def __init__(self,
                 encoder,
                 imnet,
                 local_ensemble=True,
                 feat_unfold=True,
                 cell_decode=True,
                 eval_bsize=None) -> None:
        super().__init__()
        
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.eval_bsize = eval_bsize
        
        # model
        self.encoder = build_network(encoder)
        # imnet_in_dim = self.encoder.num_feat
        imnet_in_dim = 64
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2   # attach coordinates
        if self.cell_decode:
            imnet_in_dim += 2
        imnet['in_dim'] = imnet_in_dim
        self.imnet = build_network(imnet)
    
    def forward(self, x, coord, cell, test_mode=False):
        """Forward function.

        Args:
            x: input tensor.
            coord (Tensor): coordinates tensor.
            cell (Tensor): cell tensor.
            test_mode (bool): Whether in test mode or not. Default: False.

        Returns:
            pred (Tensor): output of model.
        """
        
        feat = self.gen_feature(x)
        if self.eval_bsize is None or not test_mode:
            pred = self.query_rgb(feat, coord, cell)
        else:
            pred = self.batched_predict(feat, coord, cell)

        return pred
    
    def query_rgb(self, feat, coord, cell=None):
        """Query RGB value of GT.

        Adapted from 'https://github.com/yinboc/liif.git'
        'liif/models/liif.py'
        Copyright (c) 2020, Yinbo Chen, under BSD 3-Clause License.

        Args:
            feat (Tensor): encoded feature.
            coord (Tensor): coord tensor, shape (BHW, 2).
            cell (Tensor | None): cell tensor. Default: None.

        Returns:
            result (Tensor): (part of) output.
        """
        
        if self.imnet is None:
            coord = coord.type(feat.type())
            result = F.grid_sample(
                feat,
                coord.flip(-1).unsqueeze(1),
                mode='nearest',
                align_corners=False)
            result = result[:, :, 0, :].permute(0, 2, 1)
            return result
        
        if self.feat_unfold:    # [bs, C, h, w] --> [bs, C*9, h, w]
            feat = F.unfold(
                feat, 3,
                padding=1).view(feat.shape[0], feat.shape[1] * 9,
                                feat.shape[2], feat.shape[3])
        
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0
        
        # field radius (global: [-1, 1])
        radius_x = 2 / feat.shape[-2] / 2
        radius_y = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])    # [h, w, 2] --> [2, h, w] --> [1, 2, h, w] --> [bs, 2, h, w]
        feat_coord = feat_coord.to(coord)
        
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:       # [-1, -1] -> [-1, 1] -> [1, -1] -> [1, 1]
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * radius_x + eps_shift
                coord_[:, :, 1] += vy * radius_y + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                coord_ = coord_.type(feat.type())
                query_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)       # [bs, C*9, 1, h*w] --> [bs, C*9, h*w] --> [bs, h*w, C*9]

                feat_coord = feat_coord.type(coord_.type())
                query_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - query_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                mid_tensor = torch.cat([query_feat, rel_coord], dim=-1)     # [bs, h*w, C*9 + 2]

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    mid_tensor = torch.cat([mid_tensor, rel_cell], dim=-1)  # [bs, h*w, C*9+2 + 2]

                bs, q = coord.shape[:2]
                pred = self.imnet(mid_tensor.view(bs * q, -1)).view(bs, q, -1)      # imnet([bs*h*w, C*9+2 + 2]): [bs*h*w, 3] --> [bs, h*w, 3]
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        total_area = torch.stack(areas).sum(dim=0)      # [bs, h*w]
        if self.local_ensemble:
            areas = areas[::-1]
        result = 0
        for pred, area in zip(preds, areas):
            result = result + pred * (area / total_area).unsqueeze(-1)

        return result

    def batched_predict(self, x, coord, cell):
        """Batched predict.

        Args:
            x (Tensor): Input tensor.
            coord (Tensor): coord tensor.
            cell (Tensor): cell tensor.

        Returns:
            pred (Tensor): output of model.
        """
        with torch.no_grad():
            n = coord.shape[1]
            left = 0
            preds = []
            while left < n:
                right = min(left + self.eval_bsize, n)
                pred = self.query_rgb(x, coord[:, left:right, :],
                                      cell[:, left:right, :])
                preds.append(pred)
                left = right
            pred = torch.cat(preds, dim=1)
        return pred
    
    @abstractmethod
    def gen_feature(self, x):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """


@ARCH_REGISTRY.register()
class LIIFEDSR(LIIF):
    """LIIF net based on EDSR.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 imnet,
                 local_ensemble=True,
                 feat_unfold=True,
                 cell_decode=True,
                 eval_bsize=None):
        super().__init__(
            encoder=encoder,
            imnet=imnet,
            local_ensemble=local_ensemble,
            feat_unfold=feat_unfold,
            cell_decode=cell_decode,
            eval_bsize=eval_bsize)

        self.conv_first = self.encoder.conv_first
        self.body = self.encoder.body
        self.conv_after_body = self.encoder.conv_after_body
        del self.encoder

    def gen_feature(self, x):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        x = self.conv_first(x)
        res = self.body(x)
        res = self.conv_after_body(res)
        res += x

        return res


# @ARCH_REGISTRY.register()
class LIIFRDN(LIIF):
    """LIIF net based on RDN.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feat unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 imnet,
                 local_ensemble=True,
                 feat_unfold=True,
                 cell_decode=True,
                 eval_bsize=None):
        super().__init__(
            encoder=encoder,
            imnet=imnet,
            local_ensemble=local_ensemble,
            feat_unfold=feat_unfold,
            cell_decode=cell_decode,
            eval_bsize=eval_bsize)

        self.sfe1 = self.encoder.sfe1
        self.sfe2 = self.encoder.sfe2
        self.rdbs = self.encoder.rdbs
        self.gff = self.encoder.gff
        self.num_block = self.encoder.num_block
        del self.encoder

    def gen_feature(self, x):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.num_block):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1

        return x


# @ARCH_REGISTRY.register()
class LIIFSwinIR(LIIF):
    """LIIF net based on SwinIR.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feat unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 imnet,
                 local_ensemble=True,
                 feat_unfold=True,
                 cell_decode=True,
                 eval_bsize=None):
        super().__init__(
            encoder=encoder,
            imnet=imnet,
            local_ensemble=local_ensemble,
            feat_unfold=feat_unfold,
            cell_decode=cell_decode,
            eval_bsize=eval_bsize)

        self.window_size = self.encoder.window_size
        self.conv_first = self.encoder.conv_first
        # -------------------------------------------------------
        # self.forward_features = self.encoder.forward_features
        self.patch_embed = self.encoder.patch_embed
        self.ape = self.encoder.ape
        # self.absolute_pos_embed = self.encoder.absolute_pos_embed
        self.pos_drop = self.encoder.pos_drop
        self.layers = self.encoder.layers
        self.norm = self.encoder.norm
        self.patch_unembed = self.encoder.patch_unembed
        # -------------------------------------------------------
        self.conv_after_body = self.encoder.conv_after_body
        self.conv_before_upsample = self.encoder.conv_before_upsample
        del self.encoder
    
    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)                     # [B, h*w, C(180)]
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def gen_feature(self, x):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        
        # padding
        _, _, h, w = x.shape
        pad_input = (h % self.window_size != 0) or (w % self.window_size != 0)
        if pad_input:
            x = F.pad(x, (0, self.window_size - w % self.window_size,   # 左右
                          0, self.window_size - h % self.window_size,   # 上下
                          0, 0))
        
        x = self.conv_first(x)                                      # [B, 180, h, w]
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)

        return x


if __name__ == '__main__':
    import tqdm
    import numpy as np
    import torchvision
    from torchvision.transforms import InterpolationMode
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    from torch.backends import cudnn
    
    cudnn.benchmark = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    repetitions = 300
    
    scale = (3.5, 3.5)
    model = LIIFSwinIR(
        encoder={'type': 'SwinIR', 
                 'img_size': 48,
                 'embed_dim': 180,
                 'depths': (6, 6, 6, 6, 6, 6),
                 'num_heads': (6, 6, 6, 6, 6, 6),
                 'window_size': 8,
                 'mlp_ratio': 2.,
                 'upsampler': 'pixelshuffle'},
        imnet={'type': 'MLPRefiner', 
               'in_dim': 64, 
               'out_dim': 3,
               'hidden_list': [256, 256, 256, 256]},
        local_ensemble=True,
        feat_unfold=True,
        cell_decode=True,
        eval_bsize=30000
    ).to(device)
    window_size = 8
    scale_max = 4
    
    input = torch.rand(1, 7, 3, 100, 100).to(device)
    
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            for i in range(input.shape[1]):
                lr_img = input[:, i, ...]
                # ------ for CNN -----------------------
                # H, W = round(lr_img.shape[-2] * scale[0]), round(lr_img.shape[-1] * scale[1])
                # coord = make_coord((H, W)).to(device)
                # cell = torch.ones_like(coord)
                # cell[:, 0] *= 2 / H
                # cell[:, 1] *= 2 / W
                # sr = model(lr_img, coord.unsqueeze(0), cell.unsqueeze(0), test_mode=True)
                # _ = sr.view(1, H, W, 3).permute(0, 3, 1, 2).contiguous()
                # ------ for Transformer ---------------
                _, _, h_old, w_old = lr_img.shape
                H_old = round(lr_img.shape[-2] * scale[0])
                W_old = round(lr_img.shape[-1] * scale[1])
                
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                lr_img = torch.cat([lr_img, torch.flip(lr_img, [2])], 2)[..., :h_old + h_pad, :]
                lr_img = torch.cat([lr_img, torch.flip(lr_img, [3])], 3)[..., :w_old + w_pad]
                
                H = round(scale[0] * (h_old + h_pad))
                W = round(scale[1] * (w_old + w_pad))
                
                coord = make_coord((H, W)).to(device)
                cell = torch.ones_like(coord)
                cell[:, 0] *= 2 / lr_img.shape[-2] / scale[0]
                cell[:, 1] *= 2 / lr_img.shape[-1] / scale[1]
                
                cell_factor = max(scale[0] / scale_max, 1)
                sr = model(lr_img, coord.unsqueeze(0), cell_factor * cell.unsqueeze(0), test_mode=True)
                sr = sr.view(1, H, W, 3).permute(0, 3, 1, 2).contiguous()
                _ = sr[..., :H_old, :W_old]
                # print(_.shape)
    
    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))
    
    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            for i in range(input.shape[1]):
                lr_img = input[:, i, ...]
                # ------ for CNN -----------------------
                # H, W = round(lr_img.shape[-2] * scale[0]), round(lr_img.shape[-1] * scale[1])
                # coord = make_coord((H, W)).to(device)
                # cell = torch.ones_like(coord)
                # cell[:, 0] *= 2 / H
                # cell[:, 1] *= 2 / W
                # sr = model(lr_img, coord.unsqueeze(0), cell.unsqueeze(0), test_mode=True)
                # _ = sr.view(1, H, W, 3).permute(0, 3, 1, 2).contiguous()
                # ------ for Transformer ---------------
                _, _, h_old, w_old = lr_img.shape
                H_old = round(lr_img.shape[-2] * scale[0])
                W_old = round(lr_img.shape[-1] * scale[1])
                
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                lr_img = torch.cat([lr_img, torch.flip(lr_img, [2])], 2)[..., :h_old + h_pad, :]
                lr_img = torch.cat([lr_img, torch.flip(lr_img, [3])], 3)[..., :w_old + w_pad]
                
                H = round(scale[0] * (h_old + h_pad))
                W = round(scale[1] * (w_old + w_pad))
                
                coord = make_coord((H, W)).to(device)
                cell = torch.ones_like(coord)
                cell[:, 0] *= 2 / lr_img.shape[-2] / scale[0]
                cell[:, 1] *= 2 / lr_img.shape[-1] / scale[1]
                
                cell_factor = max(scale[0] / scale_max, 1)
                sr = model(lr_img, coord.unsqueeze(0), cell_factor * cell.unsqueeze(0), test_mode=True)
                sr = sr.view(1, H, W, 3).permute(0, 3, 1, 2).contiguous()
                _ = sr[..., :H_old, :W_old]
            # sr = model(input)
            ender.record()
            torch.cuda.synchronize() # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time
    
    avg = timings.sum()/repetitions
    print('\navg={}\n'.format(avg))
    
    print(
        "Model have {:.3f}M parameters in total".format(sum(x.numel() for x in model.parameters()) / 1000000.0))

    with torch.no_grad():
        lr_img = input[:, 0, ...]
        # ------ for CNN -----------------------
        # H, W = round(lr_img.shape[-2] * scale[0]), round(lr_img.shape[-1] * scale[1])
        # coord = make_coord((H, W)).to(device)
        # cell = torch.ones_like(coord)
        # cell[:, 0] *= 2 / H
        # cell[:, 1] *= 2 / W
        # print(flop_count_table(FlopCountAnalysis(model, 
        #                                          (lr_img, coord.unsqueeze(0), cell.unsqueeze(0), True)), 
        #                        activations=ActivationCountAnalysis(model, (lr_img, coord.unsqueeze(0), cell.unsqueeze(0), True))))
        # out = model(lr_img, coord.unsqueeze(0), cell.unsqueeze(0), True).view(1, H, W, 3).permute(0, 3, 1, 2).contiguous()
        # ------ for Transformer ---------------
        _, _, h_old, w_old = lr_img.shape
        H_old = round(lr_img.shape[-2] * scale[0])
        W_old = round(lr_img.shape[-1] * scale[1])
        
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        lr_img = torch.cat([lr_img, torch.flip(lr_img, [2])], 2)[..., :h_old + h_pad, :]
        lr_img = torch.cat([lr_img, torch.flip(lr_img, [3])], 3)[..., :w_old + w_pad]
        
        H = round(scale[0] * (h_old + h_pad))
        W = round(scale[1] * (w_old + w_pad))
        
        coord = make_coord((H, W)).to(device)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / lr_img.shape[-2] / scale[0]
        cell[:, 1] *= 2 / lr_img.shape[-1] / scale[1]
        
        cell_factor = max(scale[0] / scale_max, 1)
        print(flop_count_table(FlopCountAnalysis(model, 
                                                 (lr_img, coord.unsqueeze(0), cell_factor * cell.unsqueeze(0), True)), 
                               activations=ActivationCountAnalysis(model, (lr_img, coord.unsqueeze(0), cell.unsqueeze(0), True))))
        out = model(lr_img, coord.unsqueeze(0), cell.unsqueeze(0), True).view(1, H, W, 3).permute(0, 3, 1, 2).contiguous()
        out = out[..., :H_old, :W_old]
    print(out.shape)


"""
warm up ...

testing ...

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [01:16<00:00,  3.90it/s]

avg=256.4162207539876

Model have 22.321M parameters in total
| module                   | #parameters or shape   | #flops     | #activations   |
|:-------------------------|:-----------------------|:-----------|:---------------|
| model                    | 22.321M                | 0.39T      | 0.598G         |
|  imnet.layers            |  0.347M                |  0.169T    |  0.503G        |
|   imnet.layers.0         |   0.149M               |   72.755G  |   0.125G       |
|    imnet.layers.0.weight |    (256, 580)          |            |                |
|    imnet.layers.0.bias   |    (256,)              |            |                |
|   imnet.layers.2         |   65.792K              |   32.113G  |   0.125G       |
|    imnet.layers.2.weight |    (256, 256)          |            |                |
|    imnet.layers.2.bias   |    (256,)              |            |                |
|   imnet.layers.4         |   65.792K              |   32.113G  |   0.125G       |
|    imnet.layers.4.weight |    (256, 256)          |            |                |
|    imnet.layers.4.bias   |    (256,)              |            |                |
|   imnet.layers.6         |   65.792K              |   32.113G  |   0.125G       |
|    imnet.layers.6.weight |    (256, 256)          |            |                |
|    imnet.layers.6.bias   |    (256,)              |            |                |
|   imnet.layers.8         |   0.771K               |   0.376G   |   1.47M        |
|    imnet.layers.8.weight |    (3, 256)            |            |                |
|    imnet.layers.8.bias   |    (3,)                |            |                |
|  sfe1                    |  1.792K                |  17.28M    |  0.64M         |
|   sfe1.weight            |   (64, 3, 3, 3)        |            |                |
|   sfe1.bias              |   (64,)                |            |                |
|  sfe2                    |  36.928K               |  0.369G    |  0.64M         |
|   sfe2.weight            |   (64, 64, 3, 3)       |            |                |
|   sfe2.bias              |   (64,)                |            |                |
|  rdbs                    |  21.833M               |  0.218T    |  92.16M        |
|   rdbs.0                 |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.0.layers         |    1.328M              |    13.271G |    5.12M       |
|    rdbs.0.lff            |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.1                 |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.1.layers         |    1.328M              |    13.271G |    5.12M       |
|    rdbs.1.lff            |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.2                 |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.2.layers         |    1.328M              |    13.271G |    5.12M       |
|    rdbs.2.lff            |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.3                 |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.3.layers         |    1.328M              |    13.271G |    5.12M       |
|    rdbs.3.lff            |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.4                 |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.4.layers         |    1.328M              |    13.271G |    5.12M       |
|    rdbs.4.lff            |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.5                 |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.5.layers         |    1.328M              |    13.271G |    5.12M       |
|    rdbs.5.lff            |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.6                 |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.6.layers         |    1.328M              |    13.271G |    5.12M       |
|    rdbs.6.lff            |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.7                 |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.7.layers         |    1.328M              |    13.271G |    5.12M       |
|    rdbs.7.lff            |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.8                 |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.8.layers         |    1.328M              |    13.271G |    5.12M       |
|    rdbs.8.lff            |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.9                 |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.9.layers         |    1.328M              |    13.271G |    5.12M       |
|    rdbs.9.lff            |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.10                |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.10.layers        |    1.328M              |    13.271G |    5.12M       |
|    rdbs.10.lff           |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.11                |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.11.layers        |    1.328M              |    13.271G |    5.12M       |
|    rdbs.11.lff           |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.12                |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.12.layers        |    1.328M              |    13.271G |    5.12M       |
|    rdbs.12.lff           |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.13                |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.13.layers        |    1.328M              |    13.271G |    5.12M       |
|    rdbs.13.lff           |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.14                |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.14.layers        |    1.328M              |    13.271G |    5.12M       |
|    rdbs.14.lff           |    36.928K             |    0.369G  |    0.64M       |
|   rdbs.15                |   1.365M               |   13.64G   |   5.76M        |
|    rdbs.15.layers        |    1.328M              |    13.271G |    5.12M       |
|    rdbs.15.lff           |    36.928K             |    0.369G  |    0.64M       |
|  gff                     |  0.103M                |  1.024G    |  1.28M         |
|   gff.0                  |   65.6K                |   0.655G   |   0.64M        |
|    gff.0.weight          |    (64, 1024, 1, 1)    |            |                |
|    gff.0.bias            |    (64,)               |            |                |
|   gff.1                  |   36.928K              |   0.369G   |   0.64M        |
|    gff.1.weight          |    (64, 64, 3, 3)      |            |                |
|    gff.1.bias            |    (64,)               |            |                |
torch.Size([1, 3, 350, 350])


SwinIR
warm up ...

testing ...

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [03:13<00:00,  1.55it/s]

avg=644.2337123616536

Model have 11.950M parameters in total
| module                            | #parameters or shape   | #flops     | #activations   |
|:----------------------------------|:-----------------------|:-----------|:---------------|
| model                             | 11.95M                 | 0.318T     | 1.271G         |
|  imnet.layers                     |  0.347M                |  0.183T    |  0.544G        |
|   imnet.layers.0                  |   0.149M               |   78.692G  |   0.136G       |
|    imnet.layers.0.weight          |    (256, 580)          |            |                |
|    imnet.layers.0.bias            |    (256,)              |            |                |
|   imnet.layers.2                  |   65.792K              |   34.733G  |   0.136G       |
|    imnet.layers.2.weight          |    (256, 256)          |            |                |
|    imnet.layers.2.bias            |    (256,)              |            |                |
|   imnet.layers.4                  |   65.792K              |   34.733G  |   0.136G       |
|    imnet.layers.4.weight          |    (256, 256)          |            |                |
|    imnet.layers.4.bias            |    (256,)              |            |                |
|   imnet.layers.6                  |   65.792K              |   34.733G  |   0.136G       |
|    imnet.layers.6.weight          |    (256, 256)          |            |                |
|    imnet.layers.6.bias            |    (256,)              |            |                |
|   imnet.layers.8                  |   0.771K               |   0.407G   |   1.59M        |
|    imnet.layers.8.weight          |    (3, 256)            |            |                |
|    imnet.layers.8.bias            |    (3,)                |            |                |
|  conv_first                       |  5.04K                 |  52.566M   |  1.947M        |
|   conv_first.weight               |   (180, 3, 3, 3)       |            |                |
|   conv_first.bias                 |   (180,)               |            |                |
|  patch_embed.norm                 |  0.36K                 |  9.734M    |  0             |
|   patch_embed.norm.weight         |   (180,)               |            |                |
|   patch_embed.norm.bias           |   (180,)               |            |                |
|  layers                           |  11.202M               |  0.13T     |  0.722G        |
|   layers.0                        |   1.867M               |   21.587G  |   0.12G        |
|    layers.0.residual_group.blocks |    1.575M              |    18.433G |    0.118G      |
|    layers.0.conv                  |    0.292M              |    3.154G  |    1.947M      |
|   layers.1                        |   1.867M               |   21.587G  |   0.12G        |
|    layers.1.residual_group.blocks |    1.575M              |    18.433G |    0.118G      |
|    layers.1.conv                  |    0.292M              |    3.154G  |    1.947M      |
|   layers.2                        |   1.867M               |   21.587G  |   0.12G        |
|    layers.2.residual_group.blocks |    1.575M              |    18.433G |    0.118G      |
|    layers.2.conv                  |    0.292M              |    3.154G  |    1.947M      |
|   layers.3                        |   1.867M               |   21.587G  |   0.12G        |
|    layers.3.residual_group.blocks |    1.575M              |    18.433G |    0.118G      |
|    layers.3.conv                  |    0.292M              |    3.154G  |    1.947M      |
|   layers.4                        |   1.867M               |   21.587G  |   0.12G        |
|    layers.4.residual_group.blocks |    1.575M              |    18.433G |    0.118G      |
|    layers.4.conv                  |    0.292M              |    3.154G  |    1.947M      |
|   layers.5                        |   1.867M               |   21.587G  |   0.12G        |
|    layers.5.residual_group.blocks |    1.575M              |    18.433G |    0.118G      |
|    layers.5.conv                  |    0.292M              |    3.154G  |    1.947M      |
|  norm                             |  0.36K                 |  9.734M    |  0             |
|   norm.weight                     |   (180,)               |            |                |
|   norm.bias                       |   (180,)               |            |                |
|  conv_after_body                  |  0.292M                |  3.154G    |  1.947M        |
|   conv_after_body.weight          |   (180, 180, 3, 3)     |            |                |
|   conv_after_body.bias            |   (180,)               |            |                |
|  conv_before_upsample.0           |  0.104M                |  1.121G    |  0.692M        |
|   conv_before_upsample.0.weight   |   (64, 180, 3, 3)      |            |                |
|   conv_before_upsample.0.bias     |   (64,)                |            |                |
torch.Size([1, 3, 350, 350])
"""

