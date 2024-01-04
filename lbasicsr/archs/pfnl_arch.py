from typing import List
import torch
from torch import nn as nn
from torch.nn import functional as F

from lbasicsr.metrics.runtime import VSR_runtime_test
from lbasicsr.utils.registry import ARCH_REGISTRY


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sub_sample=1, nltype=0):
        super(NonLocalBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sub_sample = sub_sample
        self.nltype = nltype

        self.convolution_g = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.pooling_g = nn.AvgPool2d(self.sub_sample, self.sub_sample)

        self.convolution_phi = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.pooling_phi = nn.AvgPool2d(self.sub_sample, self.sub_sample)

        self.convolution_theta = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

        self.convolution_y = nn.Conv2d(self.out_channels, self.in_channels, 1, 1, 0)

        self.relu = nn.ReLU()

    def forward(self, input_x):
        batch_size, in_channels, height, width = input_x.shape
        
        assert self.nltype <= 2, ValueError("nltype must <= 2")
        
        # g
        g = self.convolution_g(input_x)
        if self.sub_sample > 1:
            g = self.pooling_g(g)
        
        # phi
        if self.nltype == 0 or self.nltype == 2:
            phi = self.convolution_phi(input_x)
        elif self.nltype == 1:
            phi = input_x
        else:
            raise ValueError('nltype can not be: {}'.format(self.nltype))
        if self.sub_sample > 1:
            phi = self.pooling_phi(phi)

        # theta
        if self.nltype == 0 or self.nltype == 2:
            theta = self.convolution_theta(input_x)
        elif self.nltype == 1:
            theta = input_x
        else:
            raise ValueError('nltype can not be: {}'.format(self.nltype))

        g_x = g.reshape([batch_size, -1, self.out_channels])
        theta_x = theta.reshape([batch_size, -1, self.out_channels])
        phi_x = phi.reshape([batch_size, -1, self.out_channels])
        phi_x = phi_x.permute(0, 2, 1)

        # f = np.matmul(theta_x, phi_x)
        f = torch.matmul(theta_x, phi_x)
        if self.nltype <= 1:
            f = torch.exp(f)
            f_softmax = f / f.sum(dim=-1, keepdim=True)
        elif self.nltype == 2:
            self.relu(f)
            f_mean = f.sum(dim=2, keepdim=True)
            f_softmax = f / f_mean
        else:
            raise ValueError('nltype can not be: {}'.format(self.nltype))

        y = torch.matmul(f_softmax, g_x)
        y = y.reshape([batch_size, self.out_channels, height, width])
        z = self.convolution_y(y)

        return z


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.bs, self.bs, c // (self.bs ** 2), h, w)      # (n, bs, bs, c//bs^2, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()                    # (n, c//bs^2, h, bs, w, bs)
        x = x.view(n, c // (self.bs ** 2), h * self.bs, w * self.bs)    # (n, c//bs^2, h * bs, w * bs)
        return x


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.bs, self.bs, w // self.bs, self.bs)      # (n, c, h//bs, bs, w//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()                        # (n, bs, bs, c, h//bs, w//bs)
        x = x.view(n, c * (self.bs ** 2), h // self.bs, w // self.bs)       # (n, c*bs^2, h//bs, w//bs)
        return x


@ARCH_REGISTRY.register()
class PFNL(nn.Module):
    def __init__(self,
                 num_frames: int = 7,
                #  in_size: int = 32,
                #  eval_in_size: List = [128, 240],
                 n_filters: int = 64,
                 n_block: int = 20,
                 scale: int = 4):
        """Network architecture for PFNL
        
        ``Progressive Fusion Video Super-Resolution Network via Exploiting Non-Local Spatio-Temporal Correlations''
        
        Reference: https://github.com/jiangxiaoyu610/PFNL-pytorch/tree/master
        
        """
        super(PFNL, self).__init__()
        
        self.n_frames = num_frames
        # self.train_size = (in_size, in_size)
        # self.eval_size = eval_in_size
        self.n_block = n_block
        self.scale = scale

        self.convolution_layer0 = nn.Sequential(
            nn.Conv2d(3, n_filters, 5, 1, 2), 
            nn.LeakyReLU())
        nn.init.xavier_uniform_(self.convolution_layer0[0].weight)

        self.convolution_layer1 = torch.nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_filters, n_filters, 3, 1, 1),
                nn.LeakyReLU())
            for _ in range(n_block)])

        self.convolution_layer10 = torch.nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.n_frames * n_filters, n_filters, 1, 1, 0),
                nn.LeakyReLU())
            for _ in range(n_block)])

        self.convolution_layer2 = torch.nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * n_filters, n_filters, 3, 1, 1),
                nn.LeakyReLU())
            for _ in range(n_block)])

        # xavier init parameter
        for i in range(n_block):
            nn.init.xavier_uniform_(self.convolution_layer1[i][0].weight)
            nn.init.xavier_uniform_(self.convolution_layer10[i][0].weight)
            nn.init.xavier_uniform_(self.convolution_layer2[i][0].weight)

        self.convolution_merge_layer1 = nn.Sequential(
            nn.Conv2d(self.n_frames * n_filters, 48, 3, 1, 1), 
            nn.LeakyReLU())
        nn.init.xavier_uniform_(self.convolution_merge_layer1[0].weight)

        self.convolution_merge_layer2 = nn.Sequential(
            nn.Conv2d(48 // (2 * 2), 12, 3, 1, 1), 
            nn.LeakyReLU())
        nn.init.xavier_uniform_(self.convolution_merge_layer2[0].weight)

        # 参考：https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
        self.space_to_depth, self.depth_to_space = SpaceToDepth(2), DepthToSpace(2)
        self.nonlocal_block = NonLocalBlock(2*2*21, 3*self.n_frames*4, 1, 1)

    def forward(self, input_image):
        # 注意！输入图片的 shape 应该变为 batch_size * n_frames * channel * width * height
        # input0 = [input_image[:, i, :, :, :] for i in range(self.n_frames)]
        # input0 = torch.cat(input0, 1)
        b, t, c, h, w = input_image.shape
        input0 = input_image.view(b, -1, h, w)

        input1 = self.space_to_depth(input0)
        input1 = self.nonlocal_block(input1)
        input1 = self.depth_to_space(input1)
        input0 += input1

        input0 = torch.split(input0, 3, dim=1)
        input0 = [self.convolution_layer0(frame) for frame in input0]

        basic = input_image[:, self.n_frames//2, :, :, :]       # .squeeze(0)
        # basic = self.perform_bicubic(basic, self.args.scale)
        basic = F.interpolate(basic, scale_factor=self.scale, mode='bicubic', align_corners=False)
        # basic = basic.unsqueeze(0)

        for i in range(self.n_block):
            input1 = [self.convolution_layer1[i](frame) for frame in input0]
            base = torch.cat(input1, 1)
            base = self.convolution_layer10[i](base)

            input2 = [torch.cat([base, frame], 1) for frame in input1]
            input2 = [self.convolution_layer2[i](frame) for frame in input2]
            input0 = [torch.add(input0[j], input2[j]) for j in range(self.n_frames)]

        merge = torch.cat(input0, 1)
        merge = self.convolution_merge_layer1(merge)

        large = self.depth_to_space(merge)
        output = self.convolution_merge_layer2(large)
        output = self.depth_to_space(output)
        
        output += basic

        return output


if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    from torch.profiler import profile, record_function, ProfilerActivity
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    
    # EDVR-M -----------------
    # net = EDVR().to(device)
    # EDVR-L -----------------
    scale = (4, 4)
    model = PFNL().to(device)
    model.eval()
    
    input = torch.rand(1, 7, 3, 180, 320).to(device)
    
    # ------ torch profile -------------------------
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,], 
        record_shapes=True,
        profile_memory=True,
        # use_cuda=True
    ) as prof:
        with record_function("model_inference"):
            for _ in range(input.shape[1]):
                out = model(input)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    
    # ------ Runtime ----------------------
    VSR_runtime_test(model, input, scale)

    # ------ Parameter --------------------
    print(
        "Model have {:.3f}M parameters in total".format(sum(x.numel() for x in model.parameters()) / 1000000.0))

    # ------ FLOPs ------------------------
    with torch.no_grad():
        print('Input:', input.shape)
        print(flop_count_table(FlopCountAnalysis(model, input), activations=ActivationCountAnalysis(model, input)))
        out = model(input)
        print('Output:', out.shape)


"""
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        model_inference        81.66%        9.718s        99.35%       11.823s       11.823s       0.000us         0.00%     912.394ms     912.394ms          -4 b     -22.05 Kb      22.19 Gb    -108.25 Gb             1  
                                           aten::conv2d         0.10%      11.408ms        11.76%        1.400s     643.009us       0.000us         0.00%     532.027ms     244.385us           0 b           0 b      29.72 Gb           0 b          2177  
                                      aten::convolution         0.09%      11.244ms        11.67%        1.388s     637.769us       0.000us         0.00%     532.027ms     244.385us           0 b           0 b      29.72 Gb           0 b          2177  
                                     aten::_convolution         0.27%      31.601ms        11.57%        1.377s     632.604us       0.000us         0.00%     532.027ms     244.385us           0 b           0 b      29.72 Gb           0 b          2177  
                                aten::cudnn_convolution         1.88%     223.127ms        10.63%        1.265s     581.239us     435.326ms        47.71%     435.326ms     199.966us           0 b           0 b      29.72 Gb      27.76 Gb          2177  
ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile14...         0.00%       0.000us         0.00%       0.000us       0.000us     389.845ms        42.73%     389.845ms     198.901us           0 b           0 b           0 b           0 b          1960  
                                             aten::_cat         0.23%      27.153ms         1.46%     173.405ms      73.383us     140.350ms        15.38%     140.350ms      59.395us           0 b           0 b      41.05 Gb           0 b          2363  
                                              aten::cat         0.09%      11.270ms         1.55%     184.665ms      78.149us       0.000us         0.00%     140.347ms      59.394us           0 b           0 b      41.05 Gb           0 b          2363  
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us     137.536ms        15.07%     137.536ms     121.929us           0 b           0 b           0 b           0 b          1128  
                                       aten::leaky_relu         0.36%      43.344ms         1.36%     161.252ms      74.550us      99.096ms        10.86%      99.096ms      45.814us           0 b           0 b      29.66 Gb      29.66 Gb          2163  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 11.900s
Self CUDA time total: 912.394ms

Warm up ...

Testing ...

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:38<00:00,  7.70it/s]

Average Runtime: 129.109 ms

Model have 3.017M parameters in total
Input: torch.Size([1, 7, 3, 180, 320])
| module                                     | #parameters or shape   | #flops    | #activations   |
|:-------------------------------------------|:-----------------------|:----------|:---------------|
| model                                      | 3.017M                 | 0.973T    | 1.348G         |
|  convolution_layer0.0                      |  4.864K                |  1.935G   |  25.805M       |
|   convolution_layer0.0.weight              |   (64, 3, 5, 5)        |           |                |
|   convolution_layer0.0.bias                |   (64,)                |           |                |
|  convolution_layer1                        |  0.739M                |  0.297T   |  0.516G        |
|   convolution_layer1.0.0                   |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.0.0.weight           |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.0.0.bias             |    (64,)               |           |                |
|   convolution_layer1.1.0                   |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.1.0.weight           |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.1.0.bias             |    (64,)               |           |                |
|   convolution_layer1.2.0                   |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.2.0.weight           |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.2.0.bias             |    (64,)               |           |                |
|   convolution_layer1.3.0                   |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.3.0.weight           |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.3.0.bias             |    (64,)               |           |                |
|   convolution_layer1.4.0                   |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.4.0.weight           |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.4.0.bias             |    (64,)               |           |                |
|   convolution_layer1.5.0                   |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.5.0.weight           |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.5.0.bias             |    (64,)               |           |                |
|   convolution_layer1.6.0                   |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.6.0.weight           |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.6.0.bias             |    (64,)               |           |                |
|   convolution_layer1.7.0                   |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.7.0.weight           |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.7.0.bias             |    (64,)               |           |                |
|   convolution_layer1.8.0                   |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.8.0.weight           |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.8.0.bias             |    (64,)               |           |                |
|   convolution_layer1.9.0                   |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.9.0.weight           |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.9.0.bias             |    (64,)               |           |                |
|   convolution_layer1.10.0                  |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.10.0.weight          |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.10.0.bias            |    (64,)               |           |                |
|   convolution_layer1.11.0                  |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.11.0.weight          |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.11.0.bias            |    (64,)               |           |                |
|   convolution_layer1.12.0                  |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.12.0.weight          |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.12.0.bias            |    (64,)               |           |                |
|   convolution_layer1.13.0                  |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.13.0.weight          |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.13.0.bias            |    (64,)               |           |                |
|   convolution_layer1.14.0                  |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.14.0.weight          |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.14.0.bias            |    (64,)               |           |                |
|   convolution_layer1.15.0                  |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.15.0.weight          |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.15.0.bias            |    (64,)               |           |                |
|   convolution_layer1.16.0                  |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.16.0.weight          |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.16.0.bias            |    (64,)               |           |                |
|   convolution_layer1.17.0                  |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.17.0.weight          |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.17.0.bias            |    (64,)               |           |                |
|   convolution_layer1.18.0                  |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.18.0.weight          |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.18.0.bias            |    (64,)               |           |                |
|   convolution_layer1.19.0                  |   36.928K              |   14.864G |   25.805M      |
|    convolution_layer1.19.0.weight          |    (64, 64, 3, 3)      |           |                |
|    convolution_layer1.19.0.bias            |    (64,)               |           |                |
|  convolution_layer10                       |  0.575M                |  33.03G   |  73.728M       |
|   convolution_layer10.0.0                  |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.0.0.weight          |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.0.0.bias            |    (64,)               |           |                |
|   convolution_layer10.1.0                  |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.1.0.weight          |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.1.0.bias            |    (64,)               |           |                |
|   convolution_layer10.2.0                  |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.2.0.weight          |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.2.0.bias            |    (64,)               |           |                |
|   convolution_layer10.3.0                  |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.3.0.weight          |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.3.0.bias            |    (64,)               |           |                |
|   convolution_layer10.4.0                  |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.4.0.weight          |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.4.0.bias            |    (64,)               |           |                |
|   convolution_layer10.5.0                  |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.5.0.weight          |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.5.0.bias            |    (64,)               |           |                |
|   convolution_layer10.6.0                  |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.6.0.weight          |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.6.0.bias            |    (64,)               |           |                |
|   convolution_layer10.7.0                  |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.7.0.weight          |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.7.0.bias            |    (64,)               |           |                |
|   convolution_layer10.8.0                  |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.8.0.weight          |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.8.0.bias            |    (64,)               |           |                |
|   convolution_layer10.9.0                  |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.9.0.weight          |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.9.0.bias            |    (64,)               |           |                |
|   convolution_layer10.10.0                 |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.10.0.weight         |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.10.0.bias           |    (64,)               |           |                |
|   convolution_layer10.11.0                 |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.11.0.weight         |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.11.0.bias           |    (64,)               |           |                |
|   convolution_layer10.12.0                 |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.12.0.weight         |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.12.0.bias           |    (64,)               |           |                |
|   convolution_layer10.13.0                 |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.13.0.weight         |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.13.0.bias           |    (64,)               |           |                |
|   convolution_layer10.14.0                 |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.14.0.weight         |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.14.0.bias           |    (64,)               |           |                |
|   convolution_layer10.15.0                 |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.15.0.weight         |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.15.0.bias           |    (64,)               |           |                |
|   convolution_layer10.16.0                 |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.16.0.weight         |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.16.0.bias           |    (64,)               |           |                |
|   convolution_layer10.17.0                 |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.17.0.weight         |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.17.0.bias           |    (64,)               |           |                |
|   convolution_layer10.18.0                 |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.18.0.weight         |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.18.0.bias           |    (64,)               |           |                |
|   convolution_layer10.19.0                 |   28.736K              |   1.652G  |   3.686M       |
|    convolution_layer10.19.0.weight         |    (64, 448, 1, 1)     |           |                |
|    convolution_layer10.19.0.bias           |    (64,)               |           |                |
|  convolution_layer2                        |  1.476M                |  0.595T   |  0.516G        |
|   convolution_layer2.0.0                   |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.0.0.weight           |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.0.0.bias             |    (64,)               |           |                |
|   convolution_layer2.1.0                   |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.1.0.weight           |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.1.0.bias             |    (64,)               |           |                |
|   convolution_layer2.2.0                   |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.2.0.weight           |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.2.0.bias             |    (64,)               |           |                |
|   convolution_layer2.3.0                   |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.3.0.weight           |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.3.0.bias             |    (64,)               |           |                |
|   convolution_layer2.4.0                   |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.4.0.weight           |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.4.0.bias             |    (64,)               |           |                |
|   convolution_layer2.5.0                   |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.5.0.weight           |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.5.0.bias             |    (64,)               |           |                |
|   convolution_layer2.6.0                   |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.6.0.weight           |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.6.0.bias             |    (64,)               |           |                |
|   convolution_layer2.7.0                   |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.7.0.weight           |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.7.0.bias             |    (64,)               |           |                |
|   convolution_layer2.8.0                   |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.8.0.weight           |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.8.0.bias             |    (64,)               |           |                |
|   convolution_layer2.9.0                   |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.9.0.weight           |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.9.0.bias             |    (64,)               |           |                |
|   convolution_layer2.10.0                  |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.10.0.weight          |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.10.0.bias            |    (64,)               |           |                |
|   convolution_layer2.11.0                  |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.11.0.weight          |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.11.0.bias            |    (64,)               |           |                |
|   convolution_layer2.12.0                  |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.12.0.weight          |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.12.0.bias            |    (64,)               |           |                |
|   convolution_layer2.13.0                  |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.13.0.weight          |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.13.0.bias            |    (64,)               |           |                |
|   convolution_layer2.14.0                  |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.14.0.weight          |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.14.0.bias            |    (64,)               |           |                |
|   convolution_layer2.15.0                  |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.15.0.weight          |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.15.0.bias            |    (64,)               |           |                |
|   convolution_layer2.16.0                  |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.16.0.weight          |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.16.0.bias            |    (64,)               |           |                |
|   convolution_layer2.17.0                  |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.17.0.weight          |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.17.0.bias            |    (64,)               |           |                |
|   convolution_layer2.18.0                  |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.18.0.weight          |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.18.0.bias            |    (64,)               |           |                |
|   convolution_layer2.19.0                  |   73.792K              |   29.727G |   25.805M      |
|    convolution_layer2.19.0.weight          |    (64, 128, 3, 3)     |           |                |
|    convolution_layer2.19.0.bias            |    (64,)               |           |                |
|  convolution_merge_layer1.0                |  0.194M                |  11.148G  |  2.765M        |
|   convolution_merge_layer1.0.weight        |   (48, 448, 3, 3)      |           |                |
|   convolution_merge_layer1.0.bias          |   (48,)                |           |                |
|  convolution_merge_layer2.0                |  1.308K                |  0.299G   |  2.765M        |
|   convolution_merge_layer2.0.weight        |   (12, 12, 3, 3)       |           |                |
|   convolution_merge_layer2.0.bias          |   (12,)                |           |                |
|  nonlocal_block                            |  28.56K                |  35.04G   |  0.211G        |
|   nonlocal_block.convolution_g             |   7.14K                |   0.102G  |   1.21M        |
|    nonlocal_block.convolution_g.weight     |    (84, 84, 1, 1)      |           |                |
|    nonlocal_block.convolution_g.bias       |    (84,)               |           |                |
|   nonlocal_block.convolution_phi           |   7.14K                |           |                |
|    nonlocal_block.convolution_phi.weight   |    (84, 84, 1, 1)      |           |                |
|    nonlocal_block.convolution_phi.bias     |    (84,)               |           |                |
|   nonlocal_block.convolution_theta         |   7.14K                |           |                |
|    nonlocal_block.convolution_theta.weight |    (84, 84, 1, 1)      |           |                |
|    nonlocal_block.convolution_theta.bias   |    (84,)               |           |                |
|   nonlocal_block.convolution_y             |   7.14K                |   0.102G  |   1.21M        |
|    nonlocal_block.convolution_y.weight     |    (84, 84, 1, 1)      |           |                |
|    nonlocal_block.convolution_y.bias       |    (84,)               |           |                |
Output: torch.Size([1, 3, 720, 1280])
"""
