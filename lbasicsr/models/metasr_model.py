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
class MetaSRModel(SRModel):
    """Meta-SR model for arbitrary-scale single image super-resolution.

    """
    
    # def feed_data(self, data):
    #     self.lq = data['lq'].to(self.device)
    #     # self.coord = data['coord'].to(self.device)
    #     # self.cell = data['cell'].to(self.device)
    #     if 'gt' in data:
    #         self.gt = data['gt'].to(self.device)
    #     if 'scale' in data:
    #         self.scale = data['scale']
    
    def test(self):
        """Forward inference. 
        
        """
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.inference_mode():
                pass
        else:
            self.net_g.eval()
            with torch.inference_mode():
                self.net_g.set_scale_inf(max(self.scale[0], self.scale[1]))
                self.output = self.net_g(self.lq)
            self.net_g.train()
