import torch
import torch.nn as nn
import torch.nn.functional as F

from lbasicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MLPRefiner(nn.Module):
    """Multilayer perceptrons (MLPs), refiner used in LIIF.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_list (list[int]): List of hidden dimensions.
    """

    def __init__(self, 
                 in_dim: int = 64, 
                 out_dim: int = 3, 
                 hidden_list: list = [256, 256, 256, 256]):
        super().__init__()

        layers = []
        last_channels = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(last_channels, hidden))
            layers.append(nn.ReLU())
            last_channels = hidden
        layers.append(nn.Linear(last_channels, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): The input of MLP.

        Returns:
            Tensor: The output of MLP.
        """

        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))

        return x.view(*shape, -1)
