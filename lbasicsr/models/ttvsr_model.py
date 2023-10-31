import torch
from copy import deepcopy

from lbasicsr.utils import get_root_logger
from lbasicsr.utils.registry import MODEL_REGISTRY
from .video_recurrent_model import VideoRecurrentModel


@MODEL_REGISTRY.register()
class TTVSRModel(VideoRecurrentModel):
    """TTVSR
    
    Paper:
        Learning Trajectory-Aware Transformer for Video Super-Resolution, CVPR, 2022
    """
    
    def __init__(self, opt):
        super(TTVSRModel, self).__init__(opt)
    
    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'generator.'
        for k, v in deepcopy(load_net).items():
            if k == 'step_counter':             # 由于TTVSR采用mmediting框架训练，在保存权重时与BasicSR不同
                load_net.pop(k)
            if k.startswith('generator.'):      # 修改权重字典中的key值即可
                load_net[k[10:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)
