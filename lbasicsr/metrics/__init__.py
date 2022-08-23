from copy import deepcopy

from lbasicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    # ===========================================================
    # 根据配置文件 yml 中的 metrics 设置，调用相应的函数
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    # ===========================================================
    return metric
