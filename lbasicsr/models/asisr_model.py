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
class ASISRModel(SRModel):
    """Arbitrary-scale image super-resolution model.
        for ArbSR, MetaSR

    """
    
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
                self.net_g.set_scale(self.scale)
                self.output = self.net_g(self.lq)
            self.net_g.train()
