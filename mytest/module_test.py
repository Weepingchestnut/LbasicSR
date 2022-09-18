import torch
from lbasicsr.archs.stan_arch import RouteFuncMLP, TadaConv2d


if __name__ == '__main__':
    print(torch.__version__)

    batch_size = 4
    frame_num = 7
    num_feature = 64
    x = torch.randn((batch_size, num_feature, frame_num, 32, 32))

    conv_rf = RouteFuncMLP(c_in=64, ratio=4, kernels=[3, 3])
    conv = TadaConv2d(
        in_channels=64,
        out_channels=64,
        kernel_size=[1, 3, 3],
        stride=[1, 1, 1],
        bias=False,
        cal_dim='cin'
    )

    out = conv(x, conv_rf(x))
    print(out.size())
