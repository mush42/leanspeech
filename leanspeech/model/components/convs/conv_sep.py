"""From: https://github.com/microsoft/NeuralSpeech/tree/master/LightSpeech"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'glu':
        return nn.GLU()
    elif act_func == 'gelu':
        return GeLU()
    elif act_func == 'gelu_accurate':
        return GeLUAcc()
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))


def gelu(x):
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return gelu(x)

class GeLUAcc(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return gelu_accurate(x)


class ConvSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dropout=0):
        super(ConvSeparable, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, self.kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)
        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * out_channels))
        nn.init.normal_(self.depthwise_conv.weight, mean=0, std=std)
        nn.init.normal_(self.pointwise_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pointwise_conv.bias, 0)

    def forward(self, x):
        # x : B * C * T
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class EncSepConvLayer(nn.Module):

    def __init__(self, c, kernel_size, dropout, activation):
        super().__init__()
        self.layer_norm = LayerNorm(c)
        self.dropout = dropout
        self.activation_fn = build_activation(activation)
        self.conv1 = ConvSeparable(c, c, kernel_size, padding=kernel_size // 2, dropout=dropout)
        self.conv2 = ConvSeparable(c, c, kernel_size, padding=kernel_size // 2, dropout=dropout)
    
    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm.training = layer_norm_training
        residual = x
        x = self.layer_norm(x)
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.t().unsqueeze(-1), 0)
        x = x.permute(1, 2, 0)
        x = self.activation_fn(self.conv1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.activation_fn(self.conv2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.permute(2, 0, 1)
        x = residual + x

        return x

