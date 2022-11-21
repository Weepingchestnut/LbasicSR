import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def input_matrix_wpn(inH, inW, scale, add_scale=True):
    """
    inH, inW: the size of the feature maps
    scale: is the upsampling times
    """
    outH, outW = int(scale * inH + 0.5), int(scale * inW + 0.5)

    # mask records which pixel is invalid, 1 valid or o invalid,
    # h_offset and w_offset calculate the offset to generate the input matrix
    # （mask 记录哪些像素无效，1有效 or 0无效，h_offset 和 w_offset 计算偏移量以生成输入矩阵）
    scale_int = int(math.ceil(scale))  # scale=1.9 ==> scale_int=2
    h_offset = torch.ones(inH, scale_int, 1)  # torch.Size([50, 2, 1])
    mask_h = torch.zeros(inH, scale_int, 1)  # torch.Size([50, 2, 1])
    w_offset = torch.ones(1, inW, scale_int)  # torch.Size([1, 50, 2])
    mask_w = torch.zeros(1, inW, scale_int)  # torch.Size([1, 50, 2])
    if add_scale:
        scale_mat = torch.zeros(1, 1)  # torch.Size([1, 1]) tensor([[0.]])
        scale_mat[0, 0] = 1.0 / scale  # tensor([[0.5263]])
        # res_scale = scale_int - scale
        # scale_mat[0,scale_int-1]=1-res_scale
        # scale_mat[0,scale_int-2]= res_scale
        scale_mat = torch.cat([scale_mat] * (inH * inW * (scale_int ** 2)),
                              0)  # (inH*inW*scale_int**2, 4) torch.Size([10000, 1])

    # projection coordinate and calculate the offset
    # （投影坐标和计算偏移量）
    h_project_coord = torch.arange(0, outH, 1).float().mul(1.0 / scale)  # 1D: size 95
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


class MetaUpscale(nn.Module):
    def __init__(self):
        super(MetaUpscale, self).__init__()

    @staticmethod
    def repeat_x(x, scale):
        scale_int = math.ceil(scale)
        N, C, H, W = x.size()
        x = x.view(N, C, H, 1, W, 1)

        x = torch.cat([x] * scale_int, 3)
        x = torch.cat([x] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return x.contiguous().view(-1, C, H, W)

    def forward(self, x, scale, lw):
        up_x = self.repeat_x(x, scale)  # the output is (N*r*r,inC,inH,inW)
        cols = F.unfold(up_x, 3, padding=1)
        scale_int = math.ceil(scale)

        cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2), 1).permute(0, 1, 3, 4, 2).contiguous()
        local_weight = lw.contiguous().view(x.size(2), scale_int, x.size(3), scale_int, -1, 3).permute(1, 3, 0, 2, 4, 5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int ** 2, x.size(2) * x.size(3), -1, 3)

        out = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        out = out.contiguous().view(x.size(0), scale_int, scale_int, 3, x.size(2), x.size(3)).permute(0, 3, 4, 1, 5, 2)
        out = out.contiguous().view(x.size(0), 3, scale_int * x.size(2), scale_int * x.size(3))

        return out



