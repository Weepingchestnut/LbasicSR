import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lbasicsr.utils.registry import ARCH_REGISTRY


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class Pos2Weight(nn.Module):
    def __init__(self, inC, kernel_size=3, outC=3):
        super(Pos2Weight, self).__init__()
        self.inC = inC
        self.kernel_size = kernel_size
        self.outC = outC
        self.meta_block = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.kernel_size * self.kernel_size * self.inC * self.outC)
        )

    def forward(self, x):
        output = self.meta_block(x)
        return output


@ARCH_REGISTRY.register()
class MetaRDN(nn.Module):
    def __init__(self, 
                #  args,
                 rgb_range: int = 255,
                 n_colors: int = 3,
                 G0: int = 64,
                 RDNkSize: int = 3,
                 RDNconfig: str = 'B'):
        super(MetaRDN, self).__init__()
        # r = args.scale[0]
        # G0 = args.G0
        kSize = RDNkSize
        self.scale = 1
        # self.args = args
        self.scale_list = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 
                           2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 
                           3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
        self.scale_idx = 0
        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[RDNconfig]

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )
        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # position to weight
        self.P2W = Pos2Weight(inC=G0)

    def repeat_x(self, x):
        scale_int = math.ceil(self.scale)
        N, C, H, W = x.size()
        x = x.view(N, C, H, 1, W, 1)

        x = torch.cat([x] * scale_int, 3)
        x = torch.cat([x] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return x.contiguous().view(-1, C, H, W)

    def forward(self, x):
        
        n, c, h, w = x.shape
        outH, outW = int(h * self.scale), int(w * self.scale)
        scale_coord_map, mask = self.input_matrix_wpn(h, w, self.scale)
        pos_mat = scale_coord_map.to(x.device)
        mask = mask.to(x.device)
        
        # d1 = time.time()
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))  # (outH*outW, outC*inC*kernel_size*kernel_size)
        # print(d2)
        up_x = self.repeat_x(x)  # the output is (N*r*r,inC,inH,inW)

        # cols = nn.functional.unfold(up_x, 3, padding=1)
        cols = F.unfold(up_x, 3, padding=1)
        scale_int = math.ceil(self.scale)

        cols = cols.contiguous().view(
            cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2), 1).permute(0, 1, 3, 4, 2).contiguous()

        local_weight = local_weight.contiguous().view(
            x.size(2), scale_int, x.size(3), scale_int, -1, 3).permute(1, 3, 0, 2, 4, 5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int ** 2, x.size(2) * x.size(3), -1, 3)

        out = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)   # torch.Size([8, 16, 3, 2500, 1])
        out = out.contiguous().view(x.size(0), scale_int, scale_int, 3, x.size(2), x.size(3)).permute(0, 3, 4, 1, 5, 2)     # torch.Size([8, 3, 50, 4, 50, 4])
        out = out.contiguous().view(x.size(0), 3, scale_int * x.size(2), scale_int * x.size(3))     # torch.Size([8, 3, 200, 200])
        out = self.add_mean(out)
        
        re_sr = torch.masked_select(out, mask)
        sr = re_sr.contiguous().view(n, c, outH, outW)

        return sr

    # def set_scale(self, scale_idx):
    #     self.scale_idx = scale_idx
    #     self.scale = self.scale_list[scale_idx]

    # def set_scale_inf(self, scale):
    #     self.scale = scale
    
    def set_scale(self, scale):
        self.scale = scale
    
    def input_matrix_wpn(self, inH, inW, scale, add_scale=True):
        """
        by given the scale and the size of input image,
        we caculate the input matrix for the weight prediction network input matrix for weight prediction network
        （在给定输入图像的尺度和大小的情况下，计算出权重预测网络的输入矩阵）
        
        inH, inW: the size of the feature maps
        scale: is the upsampling times
        """
        outH, outW = int(scale * inH), int(scale * inW)

        # mask records which pixel is invalid, 1 valid or o invalid,
        # h_offset and w_offset calculate the offset to generate the input matrix
        # （mask 记录哪些像素无效，1有效 or 0无效，h_offset 和 w_offset 计算偏移量以生成输入矩阵）
        scale_int = int(math.ceil(scale))   # scale=1.9 ==> scale_int=2
        h_offset = torch.ones(inH, scale_int, 1)    # torch.Size([50, 2, 1])
        mask_h = torch.zeros(inH, scale_int, 1)     # torch.Size([50, 2, 1])
        w_offset = torch.ones(1, inW, scale_int)    # torch.Size([1, 50, 2])
        mask_w = torch.zeros(1, inW, scale_int)     # torch.Size([1, 50, 2])
        if add_scale:
            scale_mat = torch.zeros(1, 1)   # torch.Size([1, 1]) tensor([[0.]])
            scale_mat[0, 0] = 1.0 / scale   # tensor([[0.5263]])
            # res_scale = scale_int - scale
            # scale_mat[0,scale_int-1]=1-res_scale
            # scale_mat[0,scale_int-2]= res_scale
            scale_mat = torch.cat([scale_mat] * (inH * inW * (scale_int ** 2)), 0)  # (inH*inW*scale_int**2, 4) torch.Size([10000, 1])

        # projection coordinate and calculate the offset
        # （投影坐标和计算偏移量）
        h_project_coord = torch.arange(0, outH, 1).float().mul(1.0 / scale)     # 1D: size 95
        int_h_project_coord = torch.floor(h_project_coord)

        offset_h_coord = h_project_coord - int_h_project_coord
        int_h_project_coord = int_h_project_coord.int()

        w_project_coord = torch.arange(0, outW, 1).float().mul(1.0 / scale)
        int_w_project_coord = torch.floor(w_project_coord)

        offset_w_coord = w_project_coord - int_w_project_coord
        int_w_project_coord = int_w_project_coord.int()

        # flag for number for current coordinate LR image
        # （标记当前LR图像坐标的编号）
        flag = 0
        number = 0
        for i in range(outH):
            if int_h_project_coord[i] == number:
                h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], flag, 0] = 1
                flag += 1
            else:
                h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], 0, 0] = 1
                number += 1
                flag = 1

        flag = 0
        number = 0
        for i in range(outW):
            if int_w_project_coord[i] == number:
                w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], flag] = 1
                flag += 1
            else:
                w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], 0] = 1
                number += 1
                flag = 1

        # the size is scale_int * inH * (scale_int * inW)
        h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
        #
        mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

        pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
        mask_mat = torch.sum(torch.cat((mask_h, mask_w), 2), 2).view(scale_int * inH, scale_int * inW)
        mask_mat = mask_mat.eq(2)
        pos_mat = pos_mat.contiguous().view(1, -1, 2)
        if add_scale:
            pos_mat = torch.cat((scale_mat.view(1, -1, 1), pos_mat), 2)

        return pos_mat, mask_mat  # outH*outW*2 outH=scale_int*inH , outW = scale_int *inW
        # pos_mat: torch.Size([1, 10000, 3]); mask_mat: torch.Size([100, 100])


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
    model = MetaRDN().to(device)
    model.set_scale_inf(scale[0])
    
    input = torch.rand(1, 7, 3, 100, 100).to(device)
    
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            for i in range(input.shape[1]):
                lr_img = input[:, i, ...]
                _ = model(lr_img)
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
                _ = model(lr_img)
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
        print(flop_count_table(FlopCountAnalysis(model, lr_img), activations=ActivationCountAnalysis(model, lr_img)))
        out = model(lr_img)
    print(out.shape)


"""
warm up ...

testing ...

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [02:09<00:00,  2.32it/s]

avg=430.4189848836263

Model have 22.419M parameters in total
| module                     | #parameters or shape   | #flops     | #activations   |
|:---------------------------|:-----------------------|:-----------|:---------------|
| model                      | 22.419M                | 0.291T     | 0.413G         |
|  sub_mean                  |  12                    |  90K       |  30K           |
|   sub_mean.weight          |   (3, 3, 1, 1)         |            |                |
|   sub_mean.bias            |   (3,)                 |            |                |
|  add_mean                  |  12                    |  1.44M     |  0.48M         |
|   add_mean.weight          |   (3, 3, 1, 1)         |            |                |
|   add_mean.bias            |   (3,)                 |            |                |
|  SFENet1                   |  1.792K                |  17.28M    |  0.64M         |
|   SFENet1.weight           |   (64, 3, 3, 3)        |            |                |
|   SFENet1.bias             |   (64,)                |            |                |
|  SFENet2                   |  36.928K               |  0.369G    |  0.64M         |
|   SFENet2.weight           |   (64, 64, 3, 3)       |            |                |
|   SFENet2.bias             |   (64,)                |            |                |
|  RDBs                      |  21.833M               |  0.218T    |  92.16M        |
|   RDBs.0                   |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.0.convs            |    1.328M              |    13.271G |    5.12M       |
|    RDBs.0.LFF              |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.1                   |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.1.convs            |    1.328M              |    13.271G |    5.12M       |
|    RDBs.1.LFF              |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.2                   |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.2.convs            |    1.328M              |    13.271G |    5.12M       |
|    RDBs.2.LFF              |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.3                   |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.3.convs            |    1.328M              |    13.271G |    5.12M       |
|    RDBs.3.LFF              |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.4                   |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.4.convs            |    1.328M              |    13.271G |    5.12M       |
|    RDBs.4.LFF              |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.5                   |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.5.convs            |    1.328M              |    13.271G |    5.12M       |
|    RDBs.5.LFF              |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.6                   |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.6.convs            |    1.328M              |    13.271G |    5.12M       |
|    RDBs.6.LFF              |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.7                   |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.7.convs            |    1.328M              |    13.271G |    5.12M       |
|    RDBs.7.LFF              |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.8                   |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.8.convs            |    1.328M              |    13.271G |    5.12M       |
|    RDBs.8.LFF              |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.9                   |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.9.convs            |    1.328M              |    13.271G |    5.12M       |
|    RDBs.9.LFF              |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.10                  |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.10.convs           |    1.328M              |    13.271G |    5.12M       |
|    RDBs.10.LFF             |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.11                  |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.11.convs           |    1.328M              |    13.271G |    5.12M       |
|    RDBs.11.LFF             |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.12                  |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.12.convs           |    1.328M              |    13.271G |    5.12M       |
|    RDBs.12.LFF             |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.13                  |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.13.convs           |    1.328M              |    13.271G |    5.12M       |
|    RDBs.13.LFF             |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.14                  |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.14.convs           |    1.328M              |    13.271G |    5.12M       |
|    RDBs.14.LFF             |    36.928K             |    0.369G  |    0.64M       |
|   RDBs.15                  |   1.365M               |   13.64G   |   5.76M        |
|    RDBs.15.convs           |    1.328M              |    13.271G |    5.12M       |
|    RDBs.15.LFF             |    36.928K             |    0.369G  |    0.64M       |
|  GFF                       |  0.103M                |  1.024G    |  1.28M         |
|   GFF.0                    |   65.6K                |   0.655G   |   0.64M        |
|    GFF.0.weight            |    (64, 1024, 1, 1)    |            |                |
|    GFF.0.bias              |    (64,)               |            |                |
|   GFF.1                    |   36.928K              |   0.369G   |   0.64M        |
|    GFF.1.weight            |    (64, 64, 3, 3)      |            |                |
|    GFF.1.bias              |    (64,)               |            |                |
|  P2W.meta_block            |  0.445M                |  70.902G   |  0.317G        |
|   P2W.meta_block.0         |   1.024K               |   0.123G   |   40.96M       |
|    P2W.meta_block.0.weight |    (256, 3)            |            |                |
|    P2W.meta_block.0.bias   |    (256,)              |            |                |
|   P2W.meta_block.2         |   0.444M               |   70.779G  |   0.276G       |
|    P2W.meta_block.2.weight |    (1728, 256)         |            |                |
|    P2W.meta_block.2.bias   |    (1728,)             |            |                |
torch.Size([1, 3, 350, 350])
"""