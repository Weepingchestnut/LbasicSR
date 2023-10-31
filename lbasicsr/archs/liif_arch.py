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
        imnet_in_dim = self.encoder.num_feat
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


@ARCH_REGISTRY.register()
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

