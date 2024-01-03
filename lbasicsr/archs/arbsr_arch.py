import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math

from lbasicsr.utils.registry import ARCH_REGISTRY


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def grid_sample(x, offset, scale, scale2):
    # generate grids
    b, _, h, w = x.size()
    grid = np.meshgrid(range(round(scale2*w)), range(round(scale*h)))
    # grid = np.meshgrid(range(int(scale2*w)), range(int(scale*h)))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale2 - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    grid = grid + torch.cat((offset_0, offset_1),1)
    grid = grid.permute(0, 2, 3, 1)

    # sampling
    output = F.grid_sample(x, grid, padding_mode='zeros')

    return output


class SA_upsample(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False):
        super(SA_upsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale, scale2):
        b, c, h, w = x.size()

        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, round(w * scale2), 1).unsqueeze(0).float().to(x.device)]
        # coor_hr = [torch.arange(0, int(h * scale), 1).unsqueeze(0).float().to(x.device),
        #            torch.arange(0, int(w * scale2), 1).unsqueeze(0).float().to(x.device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale2) - (torch.floor((coor_hr[1] + 0.5) / scale2 + 1e-3)) - 0.5

        input = torch.cat((
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale2,
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale2 * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
            # torch.ones_like(coor_h).expand([-1, int(scale2 * w)]).unsqueeze(0) / scale2,
            # torch.ones_like(coor_h).expand([-1, int(scale2 * w)]).unsqueeze(0) / scale,
            # coor_h.expand([-1, int(scale2 * w)]).unsqueeze(0),
            # coor_w.expand([int(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)


        # (2) predict filters and offsets
        embedding = self.body(input)
        ## offsets
        offset = self.offset(embedding)

        ## filters
        routing_weights = self.routing(embedding)
        routing_weights = routing_weights.view(self.num_experts, round(scale*h) * round(scale2*w)).transpose(0, 1)      # (h*w) * n
        # routing_weights = routing_weights.view(self.num_experts, int(scale*h) * int(scale2*w)).transpose(0, 1)      # (h*w) * n

        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(1, round(scale*h), round(scale2*w), self.channels//8, self.channels)
        # weight_compress = weight_compress.view(1, int(scale*h), int(scale2*w), self.channels//8, self.channels)

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(1, round(scale*h), round(scale2*w), self.channels, self.channels//8)
        # weight_expand = weight_expand.view(1, int(scale*h), int(scale2*w), self.channels, self.channels//8)

        # (3) grid sample & spatially varying filtering
        ## grid sample
        fea0 = grid_sample(x, offset, scale, scale2)               ## b * h * w * c * 1
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b * h * w * c * 1

        ## spatially varying filtering
        out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
        out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0


class SA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False, num_experts=4):
        super(SA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        self.bias = bias

        # FC layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )

        # initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(channels_out, channels_in, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(num_experts, channels_out))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x, scale, scale2):
        # generate routing weights
        scale = torch.ones(1, 1).to(x.device) / scale
        scale2 = torch.ones(1, 1).to(x.device) / scale2
        # scale = torch.ones(1, 1).to(x.device) / scale.to(x.device)
        # scale2 = torch.ones(1, 1).to(x.device) / scale2.to(x.device)
        routing_weights = self.routing(torch.cat((scale, scale2), 1)).view(self.num_experts, 1, 1)

        # fuse experts
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, self.channels_in, self.kernel_size, self.kernel_size)

        if self.bias:
            fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
        else:
            fused_bias = None

        # convolution
        out = F.conv2d(x, fused_weight, fused_bias, stride=self.stride, padding=self.padding)

        return out


class SA_adapt(nn.Module):
    def __init__(self, channels):
        super(SA_adapt, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            
            nn.Sigmoid()
        )
        self.adapt = SA_conv(channels, channels, 3, 1, 1)

    def forward(self, x, scale, scale2):
        # ------ padding ------------------------------------
        _, _, h, w = x.size()
        
        if h % 2 != 0 or w % 2 != 0:
            pad_h = (2 - h % 2) % 2
            pad_w = (2 - w % 2) % 2
            
            x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        # ---------------------------------------------------
        
        mask = self.mask(x)
        adapted = self.adapt(x, scale, scale2)
        
        res = adapted * mask

        # # ------ padding ----------------------------------------
        # # my padding, only pad mask
        # if mask.size(-1) != adapted.size(-1):
        #     if mask.size(-1) - adapted.size(-1) > 0:
        #         mask = mask[..., 0:adapted.size(-1)]
        #     else:
        #         mask = F.pad(mask, (0, 1), "constant", 1)
        # if mask.size(-2) != adapted.size(-2):
        #     if mask.size(-2) - adapted.size(-2) > 0:
        #         mask = mask[..., 0:adapted.size(-2), :]
        #     else:
        #         mask = F.pad(mask, (0, 0, 0, 1), "constant", 1)
        # # -------------------------------------------------------

        # ------ official -------------------------
        # return x + adapted * mask
        # ------ after padding --------------------
        return x[..., :h, :w] + res[..., :h, :w]


@ARCH_REGISTRY.register()
class ArbRCAN(nn.Module):
    """_summary_
    
    reference: https://github.com/The-Learning-And-Vision-Atelier-LAVA/ArbSR

    Args:
        nn (_type_): _description_
    """
    def __init__(self, 
                 rgb_range: int = 255,
                 n_colors: int = 3,
                 n_resgroups: int = 10,
                 n_resblocks: int = 20,
                 n_feats: int = 64,
                 kernel_size: int = 3,
                 reduction: int = 16,
                 res_scale: float = 1.,
                 conv=default_conv):
        super(ArbRCAN, self).__init__()
        
        act = nn.ReLU(True)
        self.n_resgroups = n_resgroups

        # sub_mean & add_mean layers
        # if args.data_train == 'DIV2K':
        #     print('Use DIV2K mean (0.4488, 0.4371, 0.4040)')
        #     rgb_mean = (0.4488, 0.4371, 0.4040)
        # elif args.data_train == 'DIVFlickr2K':
        #     print('Use DIVFlickr2K mean (0.4690, 0.4490, 0.4036)')
        #     rgb_mean = (0.4690, 0.4490, 0.4036)
        print('Use DIV2K mean (0.4488, 0.4371, 0.4040)')
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)
        # self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        # head module
        # modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        modules_head = [conv(n_colors, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # body module
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale,
                          n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail module
        modules_tail = [
            None,                                              # placeholder to match pre-trained RCAN model
            conv(n_feats, n_colors, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

        ##########   our plug-in module     ##########
        # scale-aware feature adaption block
        # For RCAN, feature adaption is performed after each backbone block, i.e., K=1
        self.K = 1
        sa_adapt = []
        for i in range(self.n_resgroups // self.K):
            sa_adapt.append(SA_adapt(64))
        self.sa_adapt = nn.Sequential(*sa_adapt)

        # scale-aware upsampling layer
        self.sa_upsample = SA_upsample(64)

    # ------ official --------------------
    # def set_scale(self, scale, scale2):
    #     self.scale = scale
    #     self.scale2 = scale2
    
    def set_scale(self, scale):
        self.scale = float(scale[0])
        self.scale2 = float(scale[1])

    def forward(self, x):
        # head
        x = self.sub_mean(x)
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_resgroups):
            res = self.body[i](res)
            # scale-aware feature adaption
            if (i+1) % self.K == 0:
                res = self.sa_adapt[i](res, self.scale, self.scale2)

        res = self.body[-1](res)
        res += x

        # scale-aware upsampling
        res = self.sa_upsample(res, self.scale, self.scale2)

        # tail
        x = self.tail[1](res)
        x = self.add_mean(x)

        return x


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
    model = ArbRCAN(args=None).to(device)
    model.set_scale(scale[0], scale[1])
    
    input = torch.rand(1, 7, 3, 100, 100).to(device)
    
    # print('warm up ...\n')
    # with torch.no_grad():
    #     for _ in range(100):
    #         for i in range(input.shape[1]):
    #             lr_img = input[:, i, ...]
    #             _ = model(lr_img)
    
    # # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    # torch.cuda.synchronize()
    
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # timings = np.zeros((repetitions, 1))
    
    # print('testing ...\n')
    # with torch.no_grad():
    #     for rep in tqdm.tqdm(range(repetitions)):
    #         starter.record()
    #         for i in range(input.shape[1]):
    #             lr_img = input[:, i, ...]
    #             _ = model(lr_img)
    #         # sr = model(input)
    #         ender.record()
    #         torch.cuda.synchronize() # 等待GPU任务完成
    #         curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
    #         timings[rep] = curr_time
    
    # avg = timings.sum()/repetitions
    # print('\navg={}\n'.format(avg))
    
    print(
        "Model have {:.3f}M parameters in total".format(sum(x.numel() for x in model.parameters()) / 1000000.0))

    with torch.no_grad():
        lr_img = input[:, 0, ...]
        print(flop_count_table(FlopCountAnalysis(model, lr_img), activations=ActivationCountAnalysis(model, lr_img)))
        out = model(lr_img)
    print(out.shape)


"""
Model have 16.926M parameters in total
| module                          | #parameters or shape   | #flops    | #activations   |
|:--------------------------------|:-----------------------|:----------|:---------------|
| model                           | 16.926M                | 0.158T    | 0.424G         |
|  sub_mean                       |  12                    |  90K      |  30K           |
|   sub_mean.weight               |   (3, 3, 1, 1)         |           |                |
|   sub_mean.bias                 |   (3,)                 |           |                |
|  add_mean                       |  12                    |  1.103M   |  0.367M        |
|   add_mean.weight               |   (3, 3, 1, 1)         |           |                |
|   add_mean.bias                 |   (3,)                 |           |                |
|  head.0                         |  1.792K                |  17.28M   |  0.64M         |
|   head.0.weight                 |   (64, 3, 3, 3)        |           |                |
|   head.0.bias                   |   (64,)                |           |                |
|  body                           |  15.293M               |  0.152T   |  0.263G        |
|   body.0.body                   |   1.526M               |   15.127G |   26.241M      |
|    body.0.body.0.body           |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.1.body           |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.2.body           |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.3.body           |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.4.body           |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.5.body           |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.6.body           |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.7.body           |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.8.body           |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.9.body           |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.10.body          |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.11.body          |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.12.body          |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.13.body          |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.14.body          |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.15.body          |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.16.body          |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.17.body          |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.18.body          |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.19.body          |    74.436K             |    0.738G |    1.28M       |
|    body.0.body.20               |    36.928K             |    0.369G |    0.64M       |
|   body.1.body                   |   1.526M               |   15.127G |   26.241M      |
|    body.1.body.0.body           |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.1.body           |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.2.body           |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.3.body           |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.4.body           |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.5.body           |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.6.body           |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.7.body           |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.8.body           |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.9.body           |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.10.body          |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.11.body          |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.12.body          |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.13.body          |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.14.body          |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.15.body          |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.16.body          |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.17.body          |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.18.body          |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.19.body          |    74.436K             |    0.738G |    1.28M       |
|    body.1.body.20               |    36.928K             |    0.369G |    0.64M       |
|   body.2.body                   |   1.526M               |   15.127G |   26.241M      |
|    body.2.body.0.body           |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.1.body           |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.2.body           |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.3.body           |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.4.body           |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.5.body           |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.6.body           |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.7.body           |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.8.body           |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.9.body           |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.10.body          |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.11.body          |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.12.body          |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.13.body          |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.14.body          |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.15.body          |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.16.body          |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.17.body          |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.18.body          |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.19.body          |    74.436K             |    0.738G |    1.28M       |
|    body.2.body.20               |    36.928K             |    0.369G |    0.64M       |
|   body.3.body                   |   1.526M               |   15.127G |   26.241M      |
|    body.3.body.0.body           |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.1.body           |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.2.body           |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.3.body           |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.4.body           |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.5.body           |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.6.body           |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.7.body           |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.8.body           |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.9.body           |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.10.body          |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.11.body          |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.12.body          |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.13.body          |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.14.body          |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.15.body          |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.16.body          |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.17.body          |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.18.body          |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.19.body          |    74.436K             |    0.738G |    1.28M       |
|    body.3.body.20               |    36.928K             |    0.369G |    0.64M       |
|   body.4.body                   |   1.526M               |   15.127G |   26.241M      |
|    body.4.body.0.body           |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.1.body           |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.2.body           |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.3.body           |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.4.body           |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.5.body           |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.6.body           |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.7.body           |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.8.body           |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.9.body           |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.10.body          |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.11.body          |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.12.body          |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.13.body          |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.14.body          |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.15.body          |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.16.body          |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.17.body          |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.18.body          |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.19.body          |    74.436K             |    0.738G |    1.28M       |
|    body.4.body.20               |    36.928K             |    0.369G |    0.64M       |
|   body.5.body                   |   1.526M               |   15.127G |   26.241M      |
|    body.5.body.0.body           |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.1.body           |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.2.body           |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.3.body           |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.4.body           |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.5.body           |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.6.body           |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.7.body           |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.8.body           |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.9.body           |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.10.body          |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.11.body          |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.12.body          |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.13.body          |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.14.body          |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.15.body          |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.16.body          |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.17.body          |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.18.body          |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.19.body          |    74.436K             |    0.738G |    1.28M       |
|    body.5.body.20               |    36.928K             |    0.369G |    0.64M       |
|   body.6.body                   |   1.526M               |   15.127G |   26.241M      |
|    body.6.body.0.body           |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.1.body           |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.2.body           |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.3.body           |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.4.body           |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.5.body           |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.6.body           |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.7.body           |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.8.body           |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.9.body           |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.10.body          |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.11.body          |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.12.body          |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.13.body          |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.14.body          |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.15.body          |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.16.body          |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.17.body          |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.18.body          |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.19.body          |    74.436K             |    0.738G |    1.28M       |
|    body.6.body.20               |    36.928K             |    0.369G |    0.64M       |
|   body.7.body                   |   1.526M               |   15.127G |   26.241M      |
|    body.7.body.0.body           |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.1.body           |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.2.body           |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.3.body           |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.4.body           |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.5.body           |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.6.body           |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.7.body           |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.8.body           |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.9.body           |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.10.body          |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.11.body          |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.12.body          |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.13.body          |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.14.body          |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.15.body          |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.16.body          |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.17.body          |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.18.body          |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.19.body          |    74.436K             |    0.738G |    1.28M       |
|    body.7.body.20               |    36.928K             |    0.369G |    0.64M       |
|   body.8.body                   |   1.526M               |   15.127G |   26.241M      |
|    body.8.body.0.body           |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.1.body           |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.2.body           |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.3.body           |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.4.body           |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.5.body           |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.6.body           |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.7.body           |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.8.body           |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.9.body           |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.10.body          |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.11.body          |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.12.body          |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.13.body          |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.14.body          |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.15.body          |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.16.body          |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.17.body          |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.18.body          |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.19.body          |    74.436K             |    0.738G |    1.28M       |
|    body.8.body.20               |    36.928K             |    0.369G |    0.64M       |
|   body.9.body                   |   1.526M               |   15.127G |   26.241M      |
|    body.9.body.0.body           |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.1.body           |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.2.body           |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.3.body           |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.4.body           |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.5.body           |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.6.body           |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.7.body           |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.8.body           |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.9.body           |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.10.body          |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.11.body          |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.12.body          |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.13.body          |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.14.body          |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.15.body          |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.16.body          |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.17.body          |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.18.body          |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.19.body          |    74.436K             |    0.738G |    1.28M       |
|    body.9.body.20               |    36.928K             |    0.369G |    0.64M       |
|   body.10                       |   36.928K              |   0.369G  |   0.64M        |
|    body.10.weight               |    (64, 64, 3, 3)      |           |                |
|    body.10.bias                 |    (64,)               |           |                |
|  tail.1                         |  1.731K                |  0.212G   |  0.367M        |
|   tail.1.weight                 |   (3, 64, 3, 3)        |           |                |
|   tail.1.bias                   |   (3,)                 |           |                |
|  sa_adapt                       |  1.62M                 |  4.757G   |  8.901M        |
|   sa_adapt.0                    |   0.162M               |   0.476G  |   0.89M        |
|    sa_adapt.0.mask              |    14.115K             |    0.107G |    0.25M       |
|    sa_adapt.0.adapt             |    0.148M              |    0.369G |    0.64M       |
|   sa_adapt.1                    |   0.162M               |   0.476G  |   0.89M        |
|    sa_adapt.1.mask              |    14.115K             |    0.107G |    0.25M       |
|    sa_adapt.1.adapt             |    0.148M              |    0.369G |    0.64M       |
|   sa_adapt.2                    |   0.162M               |   0.476G  |   0.89M        |
|    sa_adapt.2.mask              |    14.115K             |    0.107G |    0.25M       |
|    sa_adapt.2.adapt             |    0.148M              |    0.369G |    0.64M       |
|   sa_adapt.3                    |   0.162M               |   0.476G  |   0.89M        |
|    sa_adapt.3.mask              |    14.115K             |    0.107G |    0.25M       |
|    sa_adapt.3.adapt             |    0.148M              |    0.369G |    0.64M       |
|   sa_adapt.4                    |   0.162M               |   0.476G  |   0.89M        |
|    sa_adapt.4.mask              |    14.115K             |    0.107G |    0.25M       |
|    sa_adapt.4.adapt             |    0.148M              |    0.369G |    0.64M       |
|   sa_adapt.5                    |   0.162M               |   0.476G  |   0.89M        |
|    sa_adapt.5.mask              |    14.115K             |    0.107G |    0.25M       |
|    sa_adapt.5.adapt             |    0.148M              |    0.369G |    0.64M       |
|   sa_adapt.6                    |   0.162M               |   0.476G  |   0.89M        |
|    sa_adapt.6.mask              |    14.115K             |    0.107G |    0.25M       |
|    sa_adapt.6.adapt             |    0.148M              |    0.369G |    0.64M       |
|   sa_adapt.7                    |   0.162M               |   0.476G  |   0.89M        |
|    sa_adapt.7.mask              |    14.115K             |    0.107G |    0.25M       |
|    sa_adapt.7.adapt             |    0.148M              |    0.369G |    0.64M       |
|   sa_adapt.8                    |   0.162M               |   0.476G  |   0.89M        |
|    sa_adapt.8.mask              |    14.115K             |    0.107G |    0.25M       |
|    sa_adapt.8.adapt             |    0.148M              |    0.369G |    0.64M       |
|   sa_adapt.9                    |   0.162M               |   0.476G  |   0.89M        |
|    sa_adapt.9.mask              |    14.115K             |    0.107G |    0.25M       |
|    sa_adapt.9.adapt             |    0.148M              |    0.369G |    0.64M       |
|  sa_upsample                    |  8.966K                |  1.239G   |  0.151G        |
|   sa_upsample.weight_compress   |   (4, 8, 64, 1, 1)     |           |                |
|   sa_upsample.weight_expand     |   (4, 64, 8, 1, 1)     |           |                |
|   sa_upsample.body              |   4.48K                |   0.533G  |   15.68M       |
|    sa_upsample.body.0           |    0.32K               |    31.36M |    7.84M       |
|    sa_upsample.body.2           |    4.16K               |    0.502G |    7.84M       |
|   sa_upsample.routing.0         |   0.26K                |   31.36M  |   0.49M        |
|    sa_upsample.routing.0.weight |    (4, 64, 1, 1)       |           |                |
|    sa_upsample.routing.0.bias   |    (4,)                |           |                |
|   sa_upsample.offset            |   0.13K                |   15.68M  |   0.245M       |
|    sa_upsample.offset.weight    |    (2, 64, 1, 1)       |           |                |
|    sa_upsample.offset.bias      |    (2,)                |           |                |
torch.Size([1, 3, 350, 350])
"""
