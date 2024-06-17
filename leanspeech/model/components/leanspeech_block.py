import torch
from torch import nn

from .convnext import ConvNeXtBlock


class LeanSpeechBlock(nn.Module):
    def __init__(self,
        dim: int,
        kernel_size: int,
        num_conv_layers: int=1,
        intermediate_dim: int=None,
        padding: int='same'
    ):
        super().__init__()
        intermediate_dim = intermediate_dim or dim
        layer_scale_init_value = 1 / num_conv_layers
        self.lstm = nn.LSTM(
            dim, dim // 2, num_layers=2, batch_first=True, bidirectional=True
        )
        self.convs = nn.ModuleList()
        for __ in range(num_conv_layers):
            self.convs.append(
                ConvNeXtBlock(
                    dim,
                    intermediate_dim=intermediate_dim,
                    kernel_size=kernel_size,
                    layer_scale_init_value=layer_scale_init_value,
                    padding=padding
                )
            )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        lstm_out, lstm_state = self.lstm(x)
        conv_out = x.permute(0, 2, 1)
        for conv in self.convs:
            conv_out = conv(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        x = lstm_out + conv_out
        x = self.dropout(x)
        return x

