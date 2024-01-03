# '''
# The code is modified from the implementation of Zooming Slow-Mo:
# https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020/blob/master/codes/models/modules/Sakuya_arch.py
# '''
import functools
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lbasicsr.archs.arch_util import DCNv2Pack, ResidualBlockNoBN, make_coord, make_layer, pad_spatial
from lbasicsr.metrics.runtime import VSR_runtime_test
from lbasicsr.utils.registry import ARCH_REGISTRY


class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
        with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(PCD_Align, self).__init__()

        # fea1
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_1 = DCNv2Pack(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_1 = DCNv2Pack(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_1 = DCNv2Pack(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L1_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # fea2
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_2 = DCNv2Pack(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_2 = DCNv2Pack(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_2 = DCNv2Pack(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L1_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, fea1, fea2):
        '''align other neighboring frames to the reference frame in the feature level 
            fea1, fea2: [L1, L2, L3], each with [B,C,H,W] features
            estimate offset bidirectionally
        '''
        y = []
        # param. of fea1
        # L3
        L3_offset = torch.cat([fea1[2], fea2[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_1(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_1(fea1[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea1[1], fea2[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_1(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_1(L2_offset))
        L2_fea = self.L2_dcnpack_1(fea1[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_1(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea1[0], fea2[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_1(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_1(L1_offset))
        L1_fea = self.L1_dcnpack_1(fea1[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_1(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        # param. of fea2
        # L3
        L3_offset = torch.cat([fea2[2], fea1[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_2(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_2(fea2[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea2[1], fea1[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_2(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_2(L2_offset))
        L2_fea = self.L2_dcnpack_2(fea2[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_2(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea2[0], fea1[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_2(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_2(L1_offset))
        L1_fea = self.L1_dcnpack_2(fea2[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_2(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        y = torch.cat(y, dim=1)
        return y


class Easy_PCD(nn.Module):
    def __init__(self, nf=64, groups=8):
        super(Easy_PCD, self).__init__()

        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, f1, f2):
        # input: extracted features
        # feature size: f1 = f2 = [B, N, C, H, W]
        # print(f1.size())
        L1_fea = torch.stack([f1, f2], dim=1)
        B, N, C, H, W = L1_fea.size()
        L1_fea = L1_fea.view(-1, C, H, W)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        fea1 = [L1_fea[:, 0, :, :, :].clone(), L2_fea[:, 0, :, :, :].clone(), L3_fea[:, 0, :, :, :].clone()]
        fea2 = [L1_fea[:, 1, :, :, :].clone(), L2_fea[:, 1, :, :, :].clone(), L3_fea[:, 1, :, :, :].clone()]
        aligned_fea = self.pcd_align(fea1, fea2)
        fusion_fea = self.fusion(aligned_fea)  # [B, N, C, H, W]
        
        return fusion_fea


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, tensor_size):
        height, width = tensor_size
        return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            tensor_size = (input_tensor.size(3),input_tensor.size(4))
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0),tensor_size=tensor_size)

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, tensor_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, tensor_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class DeformableConvLSTM(ConvLSTM):
    def __init__(self, 
                 input_size, 
                 input_dim, 
                 hidden_dim, 
                 kernel_size, 
                 num_layers, 
                 front_RBs, 
                 groups,
                 batch_first=False, 
                 bias=True, 
                 return_all_layers=False):
        ConvLSTM.__init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                          batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)
        #### extract features (for each frame)
        nf = input_dim

        self.pcd_h = Easy_PCD(nf=nf, groups=groups)
        self.pcd_c = Easy_PCD(nf=nf, groups=groups)

        cell_list = []
        for i in range(0, num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input_tensor, hidden_state=None):
        '''
        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
            None.

        Returns
        -------
        last_state_list, layer_output
        '''
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            tensor_size = (input_tensor.size(3), input_tensor.size(4))
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), tensor_size=tensor_size)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                in_tensor = cur_layer_input[:, t, :, :, :]
                h_temp = self.pcd_h(in_tensor, h)
                c_temp = self.pcd_c(in_tensor, c)
                h, c = self.cell_list[layer_idx](input_tensor=in_tensor,
                                                 cur_state=[h_temp, c_temp])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, tensor_size):
        return super()._init_hidden(batch_size, tensor_size)


class BiDeformableConvLSTM(nn.Module):
    def __init__(self, 
                 input_size, 
                 input_dim, 
                 hidden_dim, 
                 kernel_size, 
                 num_layers, 
                 front_RBs, 
                 groups,
                 batch_first=False, 
                 bias=True, 
                 return_all_layers=False):
        super(BiDeformableConvLSTM, self).__init__()
        self.forward_net = DeformableConvLSTM(input_size=input_size, 
                                              input_dim=input_dim, 
                                              hidden_dim=hidden_dim,
                                              kernel_size=kernel_size, 
                                              num_layers=num_layers, 
                                              front_RBs=front_RBs,
                                              groups=groups, 
                                              batch_first=batch_first, 
                                              bias=bias,
                                              return_all_layers=return_all_layers)
        self.conv_1x1 = nn.Conv2d(2 * input_dim, input_dim, 1, 1, bias=True)

    def forward(self, x):
        reversed_idx = list(reversed(range(x.shape[1])))
        x_rev = x[:, reversed_idx, ...]
        out_fwd, _ = self.forward_net(x)
        out_rev, _ = self.forward_net(x_rev)
        rev_rev = out_rev[0][:, reversed_idx, ...]
        B, N, C, H, W = out_fwd[0].size()
        result = torch.cat((out_fwd[0], rev_rev), dim=2)
        result = result.view(B * N, -1, H, W)
        result = self.conv_1x1(result)
        return result.view(B, -1, C, H, W)


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features, 
                 outermost_linear=False, 
                 first_omega_0=30, 
                 hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features[0], 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features[i], hidden_features[i + 1], 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features[-1], out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features[-1]) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features[-1]) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features[-1], out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output     
    
    
class Siren_mask(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features[0], 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features[i], hidden_features[i + 1], 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features[-1], out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features[-1]) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features[-1]) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features[-1], out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
        # self.output_layer = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output

backwarp_tenGrid = {}
def warpgrid(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device='cuda').view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device='cuda').view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to('cuda')

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenFlow.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenFlow.shape[2] - 1.0) / 2.0)], 1)
    
    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return g, F.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)


@ARCH_REGISTRY.register()
class VideoINR(nn.Module):
    def __init__(self,
                 num_feat=64,
                 num_frame=6,
                 groups=8,
                 front_RBs=5,
                 back_RBs=10):
        super(VideoINR, self).__init__()
        self.nf = num_feat
        self.in_frames = 1 + num_frame // 2
        self.ot_frames = num_frame
        p_size = 48     # a place holder, not so useful
        patch_size = (p_size, p_size)
        n_layers = 1
        
        hidden_dim = []
        for i in range(n_layers):
            hidden_dim.append(num_feat)

        ResidualBlock_noBN_f = functools.partial(ResidualBlockNoBN, num_feat=num_feat)
        self.conv_first = nn.Conv2d(3, num_feat, kernel_size=3, stride=1, padding=1, bias=True)
        self.feature_extraction = make_layer(basic_block=ResidualBlock_noBN_f, num_basic_block=front_RBs)
        
        self.fea_L2_conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=2, padding=1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=2, padding=1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.pcd_align = PCD_Align(nf=num_feat, groups=groups)
        self.fusion = nn.Conv2d(2 * num_feat, num_feat, 1, 1, bias=True)
        self.ConvBLSTM = BiDeformableConvLSTM(input_size=patch_size, input_dim=num_feat, hidden_dim=hidden_dim, \
                                              kernel_size=(3, 3), num_layers=1, batch_first=True, front_RBs=front_RBs,
                                              groups=groups)
        
        # reconstruction
        self.recon_trunk = make_layer(basic_block=ResidualBlock_noBN_f, num_basic_block=back_RBs)
        # upsampling
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        imnet_spec2 = {'name': 'mlp', 
                       'args': {
                           'out_dim': 3, 
                           'hidden_list': [64, 64, 256, 256]}}
        # self.encode_imnet = liif_models.make(imnet_spec2, args={'in_dim': 194})
        self.feat_imnet = Siren(in_features=201, out_features=64, hidden_features=[64, 64, 256],
                                hidden_layers=2, outermost_linear=True)
        self.flow_imnet = Siren(in_features=65 + 192 + 6, out_features=4, hidden_features=[64, 64, 256],
                                hidden_layers=2, outermost_linear=True)
        self.encode_imnet = Siren(in_features=141 + 192 * 2, out_features=3, hidden_features=[64, 64, 256, 256],
                                  hidden_layers=3, outermost_linear=True)

    def gen_feat(self, x):
        self.inp = x                # [bs, 2, 3, h, w]
        B, N, C, H, W = x.size()  # N input video frames
        #### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))
        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        #### align using pcd
        to_lstm_fea = []
        '''
        0: + fea1, fusion_fea, fea2
        1: + ...    ...        ...  fusion_fea, fea2
        2: + ...    ...        ...    ...       ...   fusion_fea, fea2
        '''
        for idx in range(N - 1):
            fea1 = [
                L1_fea[:, idx, :, :, :].clone(), L2_fea[:, idx, :, :, :].clone(), L3_fea[:, idx, :, :, :].clone()
            ]
            fea2 = [
                L1_fea[:, idx + 1, :, :, :].clone(), L2_fea[:, idx + 1, :, :, :].clone(),
                L3_fea[:, idx + 1, :, :, :].clone()
            ]
            aligned_fea = self.pcd_align(fea1, fea2)

            fusion_fea = self.fusion(aligned_fea)  # [B, N, C, H, W]
            if idx == 0:
                to_lstm_fea.append(fea1[0])
            to_lstm_fea.append(fusion_fea)
            to_lstm_fea.append(fea2[0])
        lstm_feats = torch.stack(to_lstm_fea, dim=1)
        #### align using bidirectional deformable conv-lstm
        feats = self.ConvBLSTM(lstm_feats)
        B, T, C, H, W = feats.size()

        feats = feats.view(B * T, C, H, W)
        out = self.recon_trunk(feats)

        ###############################################
        out = out.view(B, T, 64, H, W)      # [bs, 3, C, h, w]
        self.feat = out
        return

    def decoding(self, times=None, gt_size: Tuple = (128, 128)):
        feat = torch.cat([self.feat[:, 0], self.feat[:, 1], self.feat[:, 2]], dim=1)    # [bs, 3, C, h, w] --> [bs, 3*C, 32, 32]

        bs, C, H, W = feat.shape
        # if isinstance(scale, int):
        #     HH, WW = H * scale, W * scale
        # else:
        #     HH, WW = scale[0], scale[1]
        HH, WW = gt_size[0], gt_size[1]
        
        coord_highres = make_coord((HH, WW)).repeat(bs, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()     # [bs, H*W, 2]

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])        # [bs, 2, h, w]

        preds = []
        for c in range(len(times)):
            qs = coord_highres.shape[1]
            q_feat = F.grid_sample(
                feat, coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_inp = F.grid_sample(
                self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3]), coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_coord = F.grid_sample(
                feat_coord, coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            rel_coord = coord_highres - q_coord
            rel_coord[:, :, 0] *= feat.shape[-2]
            rel_coord[:, :, 1] *= feat.shape[-1]
            pe_coord = torch.ones_like(coord_highres[:, :, 0].unsqueeze(2)) * times[c].unsqueeze(2)

            inp = torch.cat([q_feat, q_inp, rel_coord, pe_coord], dim=-1)
            HRfeat = self.feat_imnet(inp.view(bs * qs, -1)).view(bs, qs, -1)
            HRfeat = HRfeat.permute(0, 2, 1).view(bs, 64, HH, WW)
            HRinp = self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3])
            # HRinp = F.upsample(HRinp, scale_factor=4, mode='bilinear')
            del q_coord, rel_coord, inp
            torch.cuda.empty_cache()
            q_feat = F.grid_sample(
                HRfeat, coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_inp = F.grid_sample(
                HRinp, coord_highres.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat0 = F.grid_sample(
                feat, coord_highres.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            flow_inp = torch.cat([q_feat, q_feat0, q_inp, pe_coord], dim=-1)
            flow_pred = self.flow_imnet(flow_inp.view(bs * qs, -1)).view(bs, qs, -1)
            del q_feat, q_inp, q_feat0, flow_inp
            torch.cuda.empty_cache()
            flow_pred = flow_pred.permute(0, 2, 1).view(bs, 4, HH, WW)

            grid1, _ = warpgrid(self.inp[:, 0], flow_pred[:, :2])
            grid2, _ = warpgrid(self.inp[:, 1], flow_pred[:, 2:])
            del flow_pred
            torch.cuda.empty_cache()
            grid = grid1.view(grid1.shape[0], -1, grid1.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            q_feat1 = F.grid_sample(
                HRfeat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_img1 = F.grid_sample(
                HRinp, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat3 = F.grid_sample(
                feat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            grid = grid2.view(grid2.shape[0], -1, grid2.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            q_feat2 = F.grid_sample(
                HRfeat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_img2 = F.grid_sample(
                HRinp, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat4 = F.grid_sample(
                feat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)

            inp = torch.cat([q_feat1, q_feat2, q_feat3, q_feat4, q_img1, q_img2, pe_coord], dim=-1)
            pred = self.encode_imnet(inp.view(bs * qs, -1)).view(bs, qs, -1)
            pred = pred.permute(0, 2, 1).view(bs, 3, HH, WW)
            preds.append(pred)
        return preds

    def decoding_test(self, times=None, sr_size=None):
        feat = torch.cat([self.feat[:, 0], self.feat[:, 1], self.feat[:, 2]], dim=1)        # [bs, 3, C, h, w] --> [bs, 3C, h, w]

        bs, C, H, W = feat.shape
        # if isinstance(scale, int):
        #     HH, WW = H * scale, W * scale
        # elif isinstance(scale, Tuple):
        #     HH, WW = scale[0] * H, scale[1] * W
        #     # HH, WW = scale[0], scale[1]
        HH, WW = sr_size[0], sr_size[1]

        coord_highres = make_coord((HH, WW)).repeat(bs, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()     # [bs, H*W, 2]

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])                                # [bs, 2, h, w]

        preds = []
        for c in range(len(times)):
            qs = coord_highres.shape[1]             # H*W
            qs1 = qs // 3
            qs2 = qs // 3
            qs3 = qs - qs1 - qs2
            q_feat = F.grid_sample(
                feat, coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_inp = F.grid_sample(
                self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3]), coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_coord = F.grid_sample(
                feat_coord, coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            rel_coord = coord_highres - q_coord
            rel_coord[:, :, 0] *= feat.shape[-2]
            rel_coord[:, :, 1] *= feat.shape[-1]
            pe_coord = torch.ones_like(coord_highres[:, :, 0].unsqueeze(2)) * times[c].unsqueeze(2)

            inp = torch.cat([q_feat, q_inp, rel_coord, pe_coord], dim=-1)
            inp_p1 = inp[:, :qs1]
            inp_p2 = inp[:, qs1:qs1 + qs2]
            inp_p3 = inp[:, qs1 + qs2:]
            pred_p1 = self.feat_imnet(inp_p1.view(bs * qs1, -1)).view(bs, qs1, -1)
            pred_p2 = self.feat_imnet(inp_p2.view(bs * qs2, -1)).view(bs, qs2, -1)
            pred_p3 = self.feat_imnet(inp_p3.view(bs * qs3, -1)).view(bs, qs3, -1)

            HRfeat = torch.cat([pred_p1, pred_p2, pred_p3], dim=1)
            del q_coord, rel_coord, inp, inp_p1, inp_p2, inp_p3, pred_p1, pred_p2, pred_p3
            torch.cuda.empty_cache()

            HRfeat = HRfeat.permute(0, 2, 1).view(bs, 64, HH, WW)
            HRinp = self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3])
            # HRinp = F.upsample(HRinp, scale_factor=4, mode='bilinear')
            HRinp = F.interpolate(HRinp, scale_factor=4, mode='bilinear')

            q_feat = F.grid_sample(
                HRfeat, coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_inp = F.grid_sample(
                HRinp, coord_highres.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat0 = F.grid_sample(
                feat, coord_highres.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            flow_inp = torch.cat([q_feat, q_feat0, q_inp, pe_coord], dim=-1)
            flow_inp1 = flow_inp[:, :qs1]
            flow_inp2 = flow_inp[:, qs1:qs1 + qs2]
            flow_inp3 = flow_inp[:, qs1 + qs2:]
            flow_pred1 = self.flow_imnet(flow_inp1.view(bs * qs1, -1)).view(bs, qs1, -1)
            flow_pred2 = self.flow_imnet(flow_inp2.view(bs * qs2, -1)).view(bs, qs2, -1)
            flow_pred3 = self.flow_imnet(flow_inp3.view(bs * qs3, -1)).view(bs, qs3, -1)
            flow_pred = torch.cat([flow_pred1, flow_pred2, flow_pred3], dim=1)
            del q_feat, q_inp, q_feat0, flow_inp, flow_inp1, flow_inp2, flow_inp3, flow_pred1, flow_pred2, flow_pred3
            torch.cuda.empty_cache()
            flow_pred = flow_pred.permute(0, 2, 1).view(bs, 4, HH, WW)

            grid1, _ = warpgrid(self.inp[:, 0], flow_pred[:, :2])
            grid2, _ = warpgrid(self.inp[:, 1], flow_pred[:, 2:])
            del flow_pred
            torch.cuda.empty_cache()

            grid = grid1.view(grid1.shape[0], -1, grid1.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            q_feat1 = F.grid_sample(
                HRfeat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_img1 = F.grid_sample(
                HRinp, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat3 = F.grid_sample(
                feat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)

            grid = grid2.view(grid2.shape[0], -1, grid2.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            q_feat2 = F.grid_sample(
                HRfeat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_img2 = F.grid_sample(
                HRinp, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat4 = F.grid_sample(
                feat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)

            inp_p1 = torch.cat([q_feat1[:, :qs1], q_feat2[:, :qs1],
                                q_feat3[:, :qs1], q_feat4[:, :qs1],
                                q_img1[:, :qs1], q_img2[:, :qs1], pe_coord[:, :qs1]], dim=-1)
            pred_p1 = self.encode_imnet(inp_p1.view(bs * qs1, -1)).view(bs, qs1, -1)

            inp_p2 = torch.cat([q_feat1[:, qs1:qs1 + qs2], q_feat2[:, qs1:qs1 + qs2],
                                q_feat3[:, qs1:qs1 + qs2], q_feat4[:, qs1:qs1 + qs2],
                                q_img1[:, qs1:qs1 + qs2], q_img2[:, qs1:qs1 + qs2], pe_coord[:, qs1:qs1 + qs2]], dim=-1)
            pred_p2 = self.encode_imnet(inp_p2.view(bs * qs2, -1)).view(bs, qs2, -1)

            inp_p3 = torch.cat([q_feat1[:, qs1 + qs2:], q_feat2[:, qs1 + qs2:],
                                q_feat3[:, qs1 + qs2:], q_feat4[:, qs1 + qs2:],
                                q_img1[:, qs1 + qs2:], q_img2[:, qs1 + qs2:], pe_coord[:, qs1 + qs2:]], dim=-1)
            pred_p3 = self.encode_imnet(inp_p3.view(bs * qs3, -1)).view(bs, qs3, -1)

            pred = torch.cat([pred_p1, pred_p2, pred_p3], dim=1)
            del inp_p1, inp_p2, inp_p3, pred_p1, pred_p2, pred_p3, q_feat1, q_feat2, q_feat3, q_feat4, q_img1, q_img2, pe_coord
            torch.cuda.empty_cache()

            pred = pred.permute(0, 2, 1).view(bs, 3, HH, WW)
            preds.append(pred)
        return preds                # list[tensor[bs, 3, H, w], ...]
    

    def forward(self, x, times=None, scale=None, gt_size=None, test=False):   # x: [bs, 2, 3, h, w]; times: [0.7500, 0.8750, 1.0000]; scale: [[84], [84]]
        
        # padding
        h_input, w_input = x.shape[3:]
        H, W = get_HW_round(h_input, w_input, scale)
        x = pad_spatial(x, multiple=4, padding_mode='reflect')
        
        self.gen_feat(x)
        self.inp = x
        
        if test == True:
            outputs = self.decoding_test(times, (H, W))
            return [i[..., :H, :W] for i in outputs]
        else:
            # scale = (scale[0][0], scale[1][0])      # (H, W)
            return self.decoding(times, gt_size)


def single_forward(model, imgs_in, space_scale, time_scale):
    with torch.no_grad():
        b, n, c, h, w = imgs_in.size()
        h_n = int(4 * np.ceil(h / 4))
        w_n = int(4 * np.ceil(w / 4))
        imgs_temp = imgs_in.new_zeros(b, n, c, h_n, w_n)
        imgs_temp[:, :, :, 0:h, 0:w] = imgs_in

        time_Tensors = [torch.tensor([i / time_scale])[None].to(device) for i in range(time_scale)]
        model_output = model(imgs_temp, time_Tensors, space_scale, test=True)
        return model_output


def get_HW_round(h, w, scale: tuple):
    return round(h * scale[0]), round(w * scale[1])

def get_HW_int(h, w, scale: tuple):
    return int(h * scale[0]), int(w * scale[1])

get_HW = get_HW_round


if __name__ == '__main__':
    from torch.profiler import profile, record_function, ProfilerActivity
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'    cuda.cpp not support cpu
    
    scale = (3.5, 3.5)
    time_scale = 8
    model = VideoINR(num_feat=64,
                     num_frame=6,
                     groups=8,
                     front_RBs=5,
                     back_RBs=40).to(device)
    model.eval()
    
    # input = torch.rand(1, 30, 3, 64, 64).to(device)
    input = torch.rand(1, 2, 3, 128, 128).to(device)
    
    # ------ torch profile -------------------------
    # with profile(
    #     activities=[
    #         ProfilerActivity.CPU,
    #         ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    # ) as prof:
    #     with record_function("model_inference"):
    #         out = model(input)
    
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # ------ Runtime ------------------------------
    # VSR_runtime_test(model, input, scale)

    # ------ Parameter ----------------------------
    print(
        "Model have {:.3f}M parameters in total".format(sum(x.numel() for x in model.parameters()) / 1000000.0))
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    
    # ------ FLOPs --------------------------------
    with torch.no_grad():
        print('Input:', input.shape)
        # print(flop_count_table(FlopCountAnalysis(model, input), activations=ActivationCountAnalysis(model, input)))
        # out = model(input)
        # -->
        time_Tensors = [torch.tensor([i / time_scale])[None].to(device) for i in range(time_scale)]
        out = model(input, times=time_Tensors, scale=4, test=True)
        print('Output:', out.shape)
    
    print('warm up ...\n')
