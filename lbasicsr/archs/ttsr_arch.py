import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision import models

from lbasicsr.utils.registry import ARCH_REGISTRY


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        
    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


class SFE(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(3, n_feats)
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(
                in_channels=n_feats, out_channels=n_feats, res_scale=res_scale))
            
        self.conv_tail = conv3x3(n_feats, n_feats)
        
    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class CSFI2(nn.Module):
    def __init__(self, n_feats):
        super(CSFI2, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv21 = conv3x3(n_feats, n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats*2, n_feats)
        self.conv_merge2 = conv3x3(n_feats*2, n_feats)

    def forward(self, x1, x2):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x21 = F.relu(self.conv21(x2))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12), dim=1) ))

        return x1, x2


class CSFI3(nn.Module):
    def __init__(self, n_feats):
        super(CSFI3, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv13 = conv1x1(n_feats, n_feats)

        self.conv21 = conv3x3(n_feats, n_feats, 2)
        self.conv23 = conv1x1(n_feats, n_feats)

        self.conv31_1 = conv3x3(n_feats, n_feats, 2)
        self.conv31_2 = conv3x3(n_feats, n_feats, 2)
        self.conv32 = conv3x3(n_feats, n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats*3, n_feats)
        self.conv_merge2 = conv3x3(n_feats*3, n_feats)
        self.conv_merge3 = conv3x3(n_feats*3, n_feats)

    def forward(self, x1, x2, x3):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))

        x21 = F.relu(self.conv21(x2))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x31 = F.relu(self.conv31_1(x3))
        x31 = F.relu(self.conv31_2(x31))
        x32 = F.relu(self.conv32(x3))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21, x31), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12, x32), dim=1) ))
        x3 = F.relu(self.conv_merge3( torch.cat((x3, x13, x23), dim=1) ))
        
        return x1, x2, x3


class MergeTail(nn.Module):
    def __init__(self, n_feats):
        super(MergeTail, self).__init__()
        self.conv13 = conv1x1(n_feats, n_feats)
        self.conv23 = conv1x1(n_feats, n_feats)
        self.conv_merge = conv3x3(n_feats*3, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats//2)
        self.conv_tail2 = conv1x1(n_feats//2, 3)

    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge( torch.cat((x3, x13, x23), dim=1) ))
        x = self.conv_tail1(x)
        x = self.conv_tail2(x)
        x = torch.clamp(x, -1, 1)
        
        return x


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class LTE(torch.nn.Module):
    def __init__(self, requires_grad=True, rgb_range=1):
        super(LTE, self).__init__()
        
        # use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad
        
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
    
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.slice1(x)
        x_lv1 = x
        x = self.slice2(x)
        x_lv2 = x
        x = self.slice3(x)
        x_lv3 = x
        return x_lv1, x_lv2, x_lv3


class MainNet(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(MainNet, self).__init__()
        self.num_res_blocks = num_res_blocks    # a list containing number of resblocks of different stages
        self.n_feats = n_feats

        self.SFE = SFE(self.num_res_blocks[0], n_feats, res_scale)

        ### stage11
        self.conv11_head = conv3x3(256+n_feats, n_feats)
        self.RB11 = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
        self.conv11_tail = conv3x3(n_feats, n_feats)

        ### subpixel 1 -> 2
        self.conv12 = conv3x3(n_feats, n_feats*4)
        self.ps12 = nn.PixelShuffle(2)

        ### stage21, 22
        #self.conv21_head = conv3x3(n_feats, n_feats)
        self.conv22_head = conv3x3(128+n_feats, n_feats)

        self.ex12 = CSFI2(n_feats)

        self.RB21 = nn.ModuleList()
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB21.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            self.RB22.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))

        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)

        ### subpixel 2 -> 3
        self.conv23 = conv3x3(n_feats, n_feats*4)
        self.ps23 = nn.PixelShuffle(2)

        ### stage31, 32, 33
        #self.conv31_head = conv3x3(n_feats, n_feats)
        #self.conv32_head = conv3x3(n_feats, n_feats)
        self.conv33_head = conv3x3(64+n_feats, n_feats)

        self.ex123 = CSFI3(n_feats)

        self.RB31 = nn.ModuleList()
        self.RB32 = nn.ModuleList()
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB31.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            self.RB32.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))

        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)

        self.merge_tail = MergeTail(n_feats)

    def forward(self, x, S=None, T_lv3=None, T_lv2=None, T_lv1=None):   # S: torch.Size([1, 1, 48, 48])
        ### shallow feature extraction
        x = self.SFE(x)     # torch.Size([1, 64, 48, 48])

        ### stage11
        x11 = x

        ### soft-attention
        x11_res = x11
        x11_res = torch.cat((x11_res, T_lv3), dim=1)                                # torch.Size([1, 320, 48, 48])
        x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))      # torch.Size([1, 64, 48, 48])
        x11_res = x11_res * S
        x11 = x11 + x11_res

        x11_res = x11

        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res

        ### stage21, 22
        x21 = x11
        x21_res = x21
        x22 = self.conv12(x11)
        x22 = F.relu(self.ps12(x22))

        ### soft-attention
        x22_res = x22
        x22_res = torch.cat((x22_res, T_lv2), dim=1)
        x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
        x22_res = x22_res * F.interpolate(S, scale_factor=2, mode='bicubic')
        x22 = x22 + x22_res

        x22_res = x22

        x21_res, x22_res = self.ex12(x21_res, x22_res)

        for i in range(self.num_res_blocks[2]):
            x21_res = self.RB21[i](x21_res)
            x22_res = self.RB22[i](x22_res)

        x21_res = self.conv21_tail(x21_res)
        x22_res = self.conv22_tail(x22_res)
        x21 = x21 + x21_res
        x22 = x22 + x22_res

        ### stage31, 32, 33
        x31 = x21
        x31_res = x31
        x32 = x22
        x32_res = x32
        x33 = self.conv23(x22)
        x33 = F.relu(self.ps23(x33))

        ### soft-attention
        x33_res = x33
        x33_res = torch.cat((x33_res, T_lv1), dim=1)
        x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
        x33_res = x33_res * F.interpolate(S, scale_factor=4, mode='bicubic')
        x33 = x33 + x33_res
        
        x33_res = x33

        x31_res, x32_res, x33_res = self.ex123(x31_res, x32_res, x33_res)

        for i in range(self.num_res_blocks[3]):
            x31_res = self.RB31[i](x31_res)
            x32_res = self.RB32[i](x32_res)
            x33_res = self.RB33[i](x33_res)

        x31_res = self.conv31_tail(x31_res)
        x32_res = self.conv32_tail(x32_res)
        x33_res = self.conv33_tail(x33_res)
        x31 = x31 + x31_res
        x32 = x32 + x32_res
        x33 = x33 + x33_res

        x = self.merge_tail(x31, x32, x33)

        return x


class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()

    def bis(self, input, dim, index):   # torch.Size([1, 2304, 2304]), dim=2, index=torch.Size([1, 2304])
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]    # [tensor(1), 1, -1]
        expanse = list(input.size())    # [tensor(1), tensor(2304), tensor(2304)]
        expanse[0] = -1
        expanse[dim] = -1               # [-1, tensor(2304), -1]
        index = index.view(views).expand(expanse)   # 
        return torch.gather(input, dim, index)      # 从原tensor中获取指定dim和指定index的数据

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
        ### search
        lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)    # torch.Size([1, 2304, 2304])
        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)   # torch.Size([1, 2304, 2304])
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)

        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2)                 # [N, Hr*Wr, C*k*k]
        lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1)                  # [N, C*k*k, H*W]

        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold)                    # [N, Hr*Wr, H*W]
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)                    # [N, H*W] torch.Size([1, 2304]) 指定dim的最大值及其索引

        ### transfer
        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)                   # torch.Size([1, 2304, 2304])
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)         # torch.Size([1, 4608, 2304])
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)       # torch.Size([1, 9216, 2304])

        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)      # torch.Size([1, 2304, 2304])
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_star_arg)      # torch.Size([1, 4608, 2304])
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv3_star_arg)      # torch.Size([1, 9216, 2304])

        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)                                      # torch.Size([1, 256, 48, 48])
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2)*2, lrsr_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)        # torch.Size([1, 128, 96, 96])
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2)*4, lrsr_lv3.size(3)*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)      # torch.Size([1, 64, 192, 192])

        S = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))      # torch.Size([1, 1, 48, 48])

        return S, T_lv3, T_lv2, T_lv1


class TTSR(nn.Module):
    def __init__(self,
                 num_feat: int = 64,
                 num_res_blocks: str = '16+16+8+4',
                 res_scale: float = 1.):
        super(TTSR, self).__init__()
        
        self.num_res_blocks = list(map(int, num_res_blocks.split('+')))     # [16, 16, 8, 4]
        
        self.MainNet = MainNet(
            num_res_blocks=self.num_res_blocks, n_feats=num_feat, res_scale=res_scale)
        
        self.LTE = LTE(requires_grad=True)
        self.LTE_copy = LTE(requires_grad=False)    # used in transferal perceptual loss
        
        self.SearchTransfer = SearchTransfer()

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        _, _, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.)       # torch.Size([1, 256, 48, 48])
        _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)      # torch.Size([1, 256, 48, 48])

        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)      # torch.Size([1, 64, 192, 192]), torch.Size([1, 128, 96, 96]), torch.Size([1, 256, 48, 48])

        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)

        sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)

        return sr, S, T_lv3, T_lv2, T_lv1


if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    model = TTSR().to(device)
    model.eval()

    print(
        "Model have {:.3f}M parameters in total".format(sum(x.numel() for x in model.parameters()) / 1000000.0))

    lr = torch.rand(1, 3, 48, 48).to(device)
    lr_sr = F.interpolate(lr, scale_factor=4, mode='bicubic')
    
    ref = torch.rand(1, 3, lr.shape[-2]*4, lr.shape[-1]*4).to(device)
    ref_lr = F.interpolate(ref, scale_factor=1/4, mode='bicubic')
    ref_sr = F.interpolate(ref_lr, scale_factor=4, mode='bicubic')

    with torch.no_grad():
        print(flop_count_table(FlopCountAnalysis(model, (lr, lr_sr, ref, ref_sr)), 
                               activations=ActivationCountAnalysis(model, (lr, lr_sr, ref, ref_sr))))
        sr, _, _, _, _ = model(lr, lr_sr, ref, ref_sr)

    print(sr.shape)


"""
Model have 7.285M parameters in total
| module                           | #parameters or shape   | #flops     | #activations   |
|:---------------------------------|:-----------------------|:-----------|:---------------|
| model                            | 7.285M                 | 71.969G    | 0.105G         |
|  MainNet                         |  6.174M                |  47.316G   |  76.05M        |
|   MainNet.SFE                    |   1.22M                |   2.807G   |   5.014M       |
|    MainNet.SFE.conv_head         |    1.792K              |    3.981M  |    0.147M      |
|    MainNet.SFE.RBs               |    1.182M              |    2.718G  |    4.719M      |
|    MainNet.SFE.conv_tail         |    36.928K             |    84.935M |    0.147M      |
|   MainNet.conv11_head            |   0.184M               |   0.425G   |   0.147M       |
|    MainNet.conv11_head.weight    |    (64, 320, 3, 3)     |            |                |
|    MainNet.conv11_head.bias      |    (64,)               |            |                |
|   MainNet.RB11                   |   1.182M               |   2.718G   |   4.719M       |
|    MainNet.RB11.0                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.1                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.2                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.3                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.4                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.5                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.6                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.7                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.8                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.9                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.10               |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.11               |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.12               |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.13               |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.14               |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB11.15               |    73.856K             |    0.17G   |    0.295M      |
|   MainNet.conv11_tail            |   36.928K              |   84.935M  |   0.147M       |
|    MainNet.conv11_tail.weight    |    (64, 64, 3, 3)      |            |                |
|    MainNet.conv11_tail.bias      |    (64,)               |            |                |
|   MainNet.conv12                 |   0.148M               |   0.34G    |   0.59M        |
|    MainNet.conv12.weight         |    (256, 64, 3, 3)     |            |                |
|    MainNet.conv12.bias           |    (256,)              |            |                |
|   MainNet.conv22_head            |   0.111M               |   1.019G   |   0.59M        |
|    MainNet.conv22_head.weight    |    (64, 192, 3, 3)     |            |                |
|    MainNet.conv22_head.bias      |    (64,)               |            |                |
|   MainNet.ex12                   |   0.189M               |   0.972G   |   1.475M       |
|    MainNet.ex12.conv12           |    4.16K               |    37.749M |    0.59M       |
|    MainNet.ex12.conv21           |    36.928K             |    84.935M |    0.147M      |
|    MainNet.ex12.conv_merge1      |    73.792K             |    0.17G   |    0.147M      |
|    MainNet.ex12.conv_merge2      |    73.792K             |    0.679G  |    0.59M       |
|   MainNet.RB21                   |   0.591M               |   1.359G   |   2.359M       |
|    MainNet.RB21.0                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB21.1                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB21.2                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB21.3                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB21.4                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB21.5                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB21.6                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB21.7                |    73.856K             |    0.17G   |    0.295M      |
|   MainNet.RB22                   |   0.591M               |   5.436G   |   9.437M       |
|    MainNet.RB22.0                |    73.856K             |    0.679G  |    1.18M       |
|    MainNet.RB22.1                |    73.856K             |    0.679G  |    1.18M       |
|    MainNet.RB22.2                |    73.856K             |    0.679G  |    1.18M       |
|    MainNet.RB22.3                |    73.856K             |    0.679G  |    1.18M       |
|    MainNet.RB22.4                |    73.856K             |    0.679G  |    1.18M       |
|    MainNet.RB22.5                |    73.856K             |    0.679G  |    1.18M       |
|    MainNet.RB22.6                |    73.856K             |    0.679G  |    1.18M       |
|    MainNet.RB22.7                |    73.856K             |    0.679G  |    1.18M       |
|   MainNet.conv21_tail            |   36.928K              |   84.935M  |   0.147M       |
|    MainNet.conv21_tail.weight    |    (64, 64, 3, 3)      |            |                |
|    MainNet.conv21_tail.bias      |    (64,)               |            |                |
|   MainNet.conv22_tail            |   36.928K              |   0.34G    |   0.59M        |
|    MainNet.conv22_tail.weight    |    (64, 64, 3, 3)      |            |                |
|    MainNet.conv22_tail.bias      |    (64,)               |            |                |
|   MainNet.conv23                 |   0.148M               |   1.359G   |   2.359M       |
|    MainNet.conv23.weight         |    (256, 64, 3, 3)     |            |                |
|    MainNet.conv23.bias           |    (256,)              |            |                |
|   MainNet.conv33_head            |   73.792K              |   2.718G   |   2.359M       |
|    MainNet.conv33_head.weight    |    (64, 128, 3, 3)     |            |                |
|    MainNet.conv33_head.bias      |    (64,)               |            |                |
|   MainNet.ex123                  |   0.492M               |   6.54G    |   9.88M        |
|    MainNet.ex123.conv12          |    4.16K               |    37.749M |    0.59M       |
|    MainNet.ex123.conv13          |    4.16K               |    0.151G  |    2.359M      |
|    MainNet.ex123.conv21          |    36.928K             |    84.935M |    0.147M      |
|    MainNet.ex123.conv23          |    4.16K               |    0.151G  |    2.359M      |
|    MainNet.ex123.conv31_1        |    36.928K             |    0.34G   |    0.59M       |
|    MainNet.ex123.conv31_2        |    36.928K             |    84.935M |    0.147M      |
|    MainNet.ex123.conv32          |    36.928K             |    0.34G   |    0.59M       |
|    MainNet.ex123.conv_merge1     |    0.111M              |    0.255G  |    0.147M      |
|    MainNet.ex123.conv_merge2     |    0.111M              |    1.019G  |    0.59M       |
|    MainNet.ex123.conv_merge3     |    0.111M              |    4.077G  |    2.359M      |
|   MainNet.RB31                   |   0.295M               |   0.679G   |   1.18M        |
|    MainNet.RB31.0                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB31.1                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB31.2                |    73.856K             |    0.17G   |    0.295M      |
|    MainNet.RB31.3                |    73.856K             |    0.17G   |    0.295M      |
|   MainNet.RB32                   |   0.295M               |   2.718G   |   4.719M       |
|    MainNet.RB32.0                |    73.856K             |    0.679G  |    1.18M       |
|    MainNet.RB32.1                |    73.856K             |    0.679G  |    1.18M       |
|    MainNet.RB32.2                |    73.856K             |    0.679G  |    1.18M       |
|    MainNet.RB32.3                |    73.856K             |    0.679G  |    1.18M       |
|   MainNet.RB33                   |   0.295M               |   10.872G  |   18.874M      |
|    MainNet.RB33.0                |    73.856K             |    2.718G  |    4.719M      |
|    MainNet.RB33.1                |    73.856K             |    2.718G  |    4.719M      |
|    MainNet.RB33.2                |    73.856K             |    2.718G  |    4.719M      |
|    MainNet.RB33.3                |    73.856K             |    2.718G  |    4.719M      |
|   MainNet.conv31_tail            |   36.928K              |   84.935M  |   0.147M       |
|    MainNet.conv31_tail.weight    |    (64, 64, 3, 3)      |            |                |
|    MainNet.conv31_tail.bias      |    (64,)               |            |                |
|   MainNet.conv32_tail            |   36.928K              |   0.34G    |   0.59M        |
|    MainNet.conv32_tail.weight    |    (64, 64, 3, 3)      |            |                |
|    MainNet.conv32_tail.bias      |    (64,)               |            |                |
|   MainNet.conv33_tail            |   36.928K              |   1.359G   |   2.359M       |
|    MainNet.conv33_tail.weight    |    (64, 64, 3, 3)      |            |                |
|    MainNet.conv33_tail.bias      |    (64,)               |            |                |
|   MainNet.merge_tail             |   0.138M               |   5.062G   |   8.368M       |
|    MainNet.merge_tail.conv13     |    4.16K               |    0.151G  |    2.359M      |
|    MainNet.merge_tail.conv23     |    4.16K               |    0.151G  |    2.359M      |
|    MainNet.merge_tail.conv_merge |    0.111M              |    4.077G  |    2.359M      |
|    MainNet.merge_tail.conv_tail1 |    18.464K             |    0.679G  |    1.18M       |
|    MainNet.merge_tail.conv_tail2 |    99                  |    3.539M  |    0.111M      |
|  LTE                             |  0.555M                |  12.423G   |  23.335M       |
|   LTE.slice1.0                   |   1.792K               |   0.191G   |   7.078M       |
|    LTE.slice1.0.weight           |    (64, 3, 3, 3)       |            |                |
|    LTE.slice1.0.bias             |    (64,)               |            |                |
|   LTE.slice2                     |   0.111M               |   6.115G   |   10.617M      |
|    LTE.slice2.2                  |    36.928K             |    4.077G  |    7.078M      |
|    LTE.slice2.5                  |    73.856K             |    2.038G  |    3.539M      |
|   LTE.slice3                     |   0.443M               |   6.115G   |   5.308M       |
|    LTE.slice3.7                  |    0.148M              |    4.077G  |    3.539M      |
|    LTE.slice3.10                 |    0.295M              |    2.038G  |    1.769M      |
|   LTE.sub_mean                   |   12                   |   0.995M   |   0.332M       |
|    LTE.sub_mean.weight           |    (3, 3, 1, 1)        |            |                |
|    LTE.sub_mean.bias             |    (3,)                |            |                |
|  LTE_copy                        |  0.555M                |            |                |
|   LTE_copy.slice1.0              |   1.792K               |            |                |
|    LTE_copy.slice1.0.weight      |    (64, 3, 3, 3)       |            |                |
|    LTE_copy.slice1.0.bias        |    (64,)               |            |                |
|   LTE_copy.slice2                |   0.111M               |            |                |
|    LTE_copy.slice2.2             |    36.928K             |            |                |
|    LTE_copy.slice2.5             |    73.856K             |            |                |
|   LTE_copy.slice3                |   0.443M               |            |                |
|    LTE_copy.slice3.7             |    0.148M              |            |                |
|    LTE_copy.slice3.10            |    0.295M              |            |                |
|   LTE_copy.sub_mean              |   12                   |            |                |
|    LTE_copy.sub_mean.weight      |    (3, 3, 1, 1)        |            |                |
|    LTE_copy.sub_mean.bias        |    (3,)                |            |                |
|  SearchTransfer                  |                        |  12.231G   |  5.308M        |
"""