import numpy as np
import torch
import torch.nn as nn

from lbasicsr.metrics.flops import get_flops
from lbasicsr.metrics.runtime import VSR_runtime_test
from lbasicsr.utils.registry import ARCH_REGISTRY


def generate_it(x, t=0, nf=3, f=7):
    index = np.array([t - nf // 2 + i for i in range(nf)])
    index = np.clip(index, 0, f - 1).tolist()
    # print("index =", index)
    # it = x[:, :, index]
    it = x[:, index, ...]

    return it


class PFRB(nn.Module):
    """
    Progressive Fusion Residual Block
    Progressive Fusion Video Super-Resolution Network via Exploiting Non-Local Spatio-Temporal Correlations, ICCV 2019
    """

    def __init__(self, basic_feature=64, num_channel=3, act=torch.nn.LeakyReLU(0.2, True)):
        super(PFRB, self).__init__()
        self.bf = basic_feature
        self.nc = num_channel
        self.act = act
        self.conv0 = nn.Sequential(*[nn.Conv2d(self.bf, self.bf, 3, 1, 3 // 2) for _ in range(num_channel)])
        self.conv1 = nn.Conv2d(self.bf * num_channel, self.bf, 1, 1, 1 // 2)
        self.conv2 = nn.Sequential(*[nn.Conv2d(self.bf * 2, self.bf, 3, 1, 3 // 2) for _ in range(num_channel)])

    def forward(self, x):
        x1 = [self.act(self.conv0[i](x[i])) for i in range(self.nc)]
        merge = torch.cat(x1, 1)
        base = self.act(self.conv1(merge))
        x2 = [torch.cat([base, i], 1) for i in x1]
        x2 = [self.act(self.conv2[i](x2[i])) for i in range(self.nc)]

        return [torch.add(x[i], x2[i]) for i in range(self.nc)]


class UPSCALE(nn.Module):
    def __init__(self, basic_feature=64, scale=4, act=nn.LeakyReLU(0.2, True)):
        super(UPSCALE, self).__init__()
        body = []
        body.append(nn.Conv2d(basic_feature, 48, 3, 1, 3 // 2))
        body.append(act)
        body.append(nn.PixelShuffle(2))
        body.append(nn.Conv2d(12, 12, 3, 1, 3 // 2))
        body.append(nn.PixelShuffle(2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class UNIT(nn.Module):
    def __init__(self, kind='successor', basic_feature=64, num_frame=3, num_b=5, scale=4, act=nn.LeakyReLU(0.2, True)):
        super(UNIT, self).__init__()
        self.bf = basic_feature
        self.nf = num_frame
        self.num_b = num_b
        self.scale = scale
        self.act = act
        self.kind = kind
        if kind == 'precursor':
            self.conv_c = nn.Conv2d(3, self.bf, 3, 1, 3 // 2)
            self.conv_sup = nn.Conv2d(3 * (num_frame - 1), self.bf, 3, 1, 3 // 2)
        else:
            self.conv_c = nn.Sequential(*[nn.Conv2d((3 + self.bf), self.bf, 3, 1, 3 // 2) for i in range(num_frame)])
        self.blocks = nn.Sequential(*[PFRB(self.bf, 3, act) for i in range(num_b)])
        self.merge = nn.Conv2d(3 * self.bf, self.bf, 3, 1, 3 // 2)
        self.upscale = UPSCALE(self.bf, scale, act)
        print(kind, num_b)

    def forward(self, it, ht_past, ht_now=None, ht_future=None):
        # B, C, T, H, W = it.shape    # torch.Size([B, 3, 3, 128, 240])
        B, T, C, H, W = it.size()

        if self.kind == 'precursor':
            # it_c = it[:, :, T // 2]     # torch.Size([B, 3(C), 128, 240])  get center frame
            it_c = it[:, T // 2]
            index_sup = list(range(T))
            index_sup.pop(T // 2)       # [0, 2] the index of support frame
            # it_sup = it[:, :, index_sup]    # torch.Size([B, 3(C), 2(T), 128, 240])
            it_sup = it[:, index_sup]
            # it_sup = it_sup.view(B, C * (T - 1), H, W)      # torch.Size([B, 6, 128, 240])
            it_sup = it_sup.view(B, (T - 1) * C, H, W)
            hsup = self.act(self.conv_sup(it_sup))          # torch.Size([B, 80, 128, 240])     the hidden state of support frame
            hc = self.act(self.conv_c(it_c))                # torch.Size([b, 80, 128, 240])     the hidden state of center frame
            inp = [hc, hsup, ht_past]
        else:
            ht = [ht_past, ht_now, ht_future]
            # it_c = [torch.cat([it[:, :, i, :, :], ht[i]], 1) for i in range(3)]
            it_c = [torch.cat([it[:, i, :, :, :], ht[i]], 1) for i in range(3)]
            inp = [self.act(self.conv_c[i](it_c[i])) for i in range(3)]

        inp = self.blocks(inp)      # after some residual block

        ht = self.merge(torch.cat(inp, 1))      # Conv 3x3 to merge feature maps (Note that there no LReLU)
        it_sr = self.upscale(ht)

        return it_sr, ht


# @ARCH_REGISTRY.register()
class OVSR(nn.Module):
    def __init__(self, num_feat=56, num_pb=4, num_sb=2, scale=4, num_frame=3, kind='global'):
        super(OVSR, self).__init__()
        self.bf = num_feat
        self.num_pb = num_pb
        self.num_sb = num_sb
        self.scale = scale
        self.nf = num_frame
        self.kind = kind  # local or global
        self.act = nn.LeakyReLU(0.2, True)
        self.precursor = UNIT('precursor', self.bf, self.nf, self.num_pb, self.scale, self.act)
        self.successor = UNIT('successor', self.bf, self.nf, self.num_sb, self.scale, self.act)
        # print(self.kind, '{}+{}'.format(self.num_pb, self.num_sb))
        #
        # params = list(self.parameters())
        # pnum = 0
        # for p in params:
        #     l = 1
        #     for j in p.shape:
        #         l *= j
        #     pnum += l
        # print('Number of parameters {}'.format(pnum))

    def forward(self, x, start=0):
        # B, C, T, H, W = x.shape
        B, T, C, H, W = x.size()
        start = max(0, start)
        end = T - start

        sr_all = []
        pre_sr_all = []
        pre_ht_all = []
        ht_past = torch.zeros((B, self.bf, H, W), dtype=torch.float, device=x.device)

        # precursor
        for idx in range(T):
            t = idx if self.kind == 'local' else T - idx - 1
            insert_idx = T + 1 if self.kind == 'local' else 0

            it = generate_it(x, t, self.nf, T)
            it_sr_pre, ht_past = self.precursor(it, ht_past, None, None)
            pre_ht_all.insert(insert_idx, ht_past)
            pre_sr_all.insert(insert_idx, it_sr_pre)

        # successor
        ht_past = torch.zeros((B, self.bf, H, W), dtype=torch.float, device=x.device)
        for t in range(end):
            it = generate_it(x, t, self.nf, T)
            ht_future = pre_ht_all[t] if t == T - 1 else pre_ht_all[t + 1]
            it_sr, ht_past = self.successor(it, ht_past, pre_ht_all[t], ht_future)
            sr_all.append(it_sr + pre_sr_all[t])

        # sr_all = torch.stack(sr_all, 2)[:, :, start:]
        sr_all = torch.stack(sr_all, 1)[:, start:]
        pre_sr_all = torch.stack(pre_sr_all, 1)[:, start:end]

        # return sr_all, pre_sr_all
        # for influence test
        return sr_all


if __name__ == '__main__':
    from torch.profiler import profile, record_function, ProfilerActivity
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # GOVSR-8+4-56 -------------------
    # net = OVSR(
    #     num_feat=56,
    #     num_pb=8,
    #     num_sb=4,
    #     scale=4,
    #     num_frame=5).to(device)
    # GOVSR-8+4-80 -------------------
    scale = (4, 4)
    model = OVSR(
        num_feat=80,
        num_pb=8,
        num_sb=4,
        scale=4,
        num_frame=3).to(device)
    model.eval()
    
    input = torch.rand(1, 7, 3, 180, 320).to(device)
    # input = torch.rand(1, 9, 3, 64, 64).to(device)
    
    # ------ torch profile -------------------------
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("model_inference"):
            out = model(input)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # ------ Runtime ------------------------------
    VSR_runtime_test(model, input, scale)
    
    # ------ Parameter ----------------------------
    print(
        "Model have {:.3f}M parameters in total".format(sum(x.numel() for x in model.parameters()) / 1000000.0))

    # ------ FLOPs --------------------------------
    with torch.no_grad():
        print('Input:', input.shape)
        print(flop_count_table(FlopCountAnalysis(model, input), activations=ActivationCountAnalysis(model, input)))
        out = model(input)
        print('Output:', out.shape)


"""
# GOVSR-8+4-80

precursor 8
successor 4
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        model_inference         2.91%      46.688ms        99.99%        1.602s        1.602s       0.000us         0.00%     363.550ms     363.550ms          -4 b        -468 b      29.64 Gb      -1.13 Gb             1  
                                           aten::conv2d         0.17%       2.682ms        87.24%        1.398s       2.102ms       0.000us         0.00%     257.541ms     387.280us           0 b           0 b      11.48 Gb           0 b           665  
                                      aten::convolution         0.20%       3.143ms        87.08%        1.395s       2.098ms       0.000us         0.00%     257.541ms     387.280us           0 b           0 b      11.48 Gb           0 b           665  
                                     aten::_convolution         0.69%      11.027ms        86.88%        1.392s       2.093ms       0.000us         0.00%     257.541ms     387.280us           0 b           0 b      11.48 Gb           0 b           665  
                                aten::cudnn_convolution         6.11%      97.924ms        85.15%        1.364s       2.052ms     220.929ms        60.77%     220.929ms     332.224us           0 b           0 b      11.48 Gb       1.62 Gb           665  
sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nh...         0.00%       0.000us         0.00%       0.000us       0.000us      94.321ms        25.94%      94.321ms     354.590us           0 b           0 b           0 b           0 b           266  
ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile14...         0.00%       0.000us         0.00%       0.000us       0.000us      71.954ms        19.79%      71.954ms     250.711us           0 b           0 b           0 b           0 b           287  
                                              aten::cat         0.17%       2.711ms         6.36%     101.933ms     273.279us       0.000us         0.00%      47.716ms     127.925us           0 b           0 b      14.42 Gb           0 b           373  
                                             aten::_cat         0.37%       5.984ms         6.19%      99.222ms     266.011us      47.716ms        13.13%      47.716ms     127.925us           0 b           0 b      14.42 Gb           0 b           373  
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us      47.206ms        12.98%      47.206ms     127.240us           0 b           0 b           0 b           0 b           371  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.602s
Self CUDA time total: 363.550ms

Warm up ...

Testing ...

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [01:52<00:00,  2.66it/s]

Average Runtime: 375.496 ms

Model have 7.062M parameters in total
Input: torch.Size([1, 7, 3, 180, 320])
| module                       | #parameters or shape   | #flops     | #activations   |
|:-----------------------------|:-----------------------|:-----------|:---------------|
| model                        | 7.062M                 | 2.847T     | 3.013G         |
|  precursor                   |  4.521M                |  1.822T    |  1.942G        |
|   precursor.conv_c           |   2.24K                |   0.871G   |   32.256M      |
|    precursor.conv_c.weight   |    (80, 3, 3, 3)       |            |                |
|    precursor.conv_c.bias     |    (80,)               |            |                |
|   precursor.conv_sup         |   4.4K                 |   1.742G   |   32.256M      |
|    precursor.conv_sup.weight |    (80, 6, 3, 3)       |            |                |
|    precursor.conv_sup.bias   |    (80,)               |            |                |
|   precursor.blocks           |   4.305M               |   1.734T   |   1.806G       |
|    precursor.blocks.0        |    0.538M              |    0.217T  |    0.226G      |
|    precursor.blocks.1        |    0.538M              |    0.217T  |    0.226G      |
|    precursor.blocks.2        |    0.538M              |    0.217T  |    0.226G      |
|    precursor.blocks.3        |    0.538M              |    0.217T  |    0.226G      |
|    precursor.blocks.4        |    0.538M              |    0.217T  |    0.226G      |
|    precursor.blocks.5        |    0.538M              |    0.217T  |    0.226G      |
|    precursor.blocks.6        |    0.538M              |    0.217T  |    0.226G      |
|    precursor.blocks.7        |    0.538M              |    0.217T  |    0.226G      |
|   precursor.merge            |   0.173M               |   69.673G  |   32.256M      |
|    precursor.merge.weight    |    (80, 240, 3, 3)     |            |                |
|    precursor.merge.bias      |    (80,)               |            |                |
|   precursor.upscale.body     |   35.916K              |   16.025G  |   38.707M      |
|    precursor.upscale.body.0  |    34.608K             |    13.935G |    19.354M     |
|    precursor.upscale.body.3  |    1.308K              |    2.09G   |    19.354M     |
|  successor                   |  2.541M                |  1.025T    |  1.071G        |
|   successor.conv_c           |   0.18M                |   72.286G  |   96.768M      |
|    successor.conv_c.0        |    59.84K              |    24.095G |    32.256M     |
|    successor.conv_c.1        |    59.84K              |    24.095G |    32.256M     |
|    successor.conv_c.2        |    59.84K              |    24.095G |    32.256M     |
|   successor.blocks           |   2.153M               |   0.867T   |   0.903G       |
|    successor.blocks.0        |    0.538M              |    0.217T  |    0.226G      |
|    successor.blocks.1        |    0.538M              |    0.217T  |    0.226G      |
|    successor.blocks.2        |    0.538M              |    0.217T  |    0.226G      |
|    successor.blocks.3        |    0.538M              |    0.217T  |    0.226G      |
|   successor.merge            |   0.173M               |   69.673G  |   32.256M      |
|    successor.merge.weight    |    (80, 240, 3, 3)     |            |                |
|    successor.merge.bias      |    (80,)               |            |                |
|   successor.upscale.body     |   35.916K              |   16.025G  |   38.707M      |
|    successor.upscale.body.0  |    34.608K             |    13.935G |    19.354M     |
|    successor.upscale.body.3  |    1.308K              |    2.09G   |    19.354M     |
Output: torch.Size([1, 7, 3, 720, 1280])
"""