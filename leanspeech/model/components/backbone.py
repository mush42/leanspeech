from typing import Dict, List, Tuple
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .convs import ConvGLU
from .regularization import DropPath


class LeanSpeechBlock(nn.Module):
    def __init__(self,
        dim: int,
        kernel_size: int,
        num_conv_layers: int,
        drop_path: float=0.0,
        dropout: float=0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            dim, dim, num_layers=1, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList([
            ConvGLU(dim, kernel_size, dropout=dropout, batchnorm=i < (num_conv_layers - 1))
            for i in range(num_conv_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, lengths, mask):
        residual = x

        lx, hx = self.lstm(x)
        lx = self.dropout(lx.tanh())

        cx = x.transpose(1, 2)
        for conv_block in self.convs:
            cx = conv_block(cx)
            cx = cx * mask
        cx = cx.transpose(1, 2)

        x = lx + cx
        x = self.final_layer_norm(x)

        x = residual + self.drop_path(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class LeanSpeechBackbone(nn.Module):
    def __init__(
        self,
        dim: int,
        layers: List[Dict[str, int]],
        drop_path: float=0.0,
        dropout: float=0.0,
    ):
        super().__init__()
        drop_ppath_rates=[x.item() for x in torch.linspace(0, drop_path, len(layers))] 
        self.layers = nn.ModuleList([
            LeanSpeechBlock(
                dim=dim,
                kernel_size=layer.kernel_size,
                num_conv_layers=layer.num_conv_layers,
                dropout=dropout,
                drop_path=dp_rate
            )
            for (layer, dp_rate) in zip(layers, drop_ppath_rates)
        ])

    def forward(self, x, lengths, mask):
        for layer in self.layers:
            x = layer(x, lengths, mask)
        return x
