from typing import Tuple
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .convglu import ConvGLU

class LeanSpeechBlock(nn.Module):
    def __init__(self,
        dim: int,
        kernel_sizes: Tuple[int],
        dropout: float=0.1
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            dim, dim, num_layers=1, batch_first=True
        )
        self.convs = nn.ModuleList([
            ConvGLU(dim, kernel_size, dropout=dropout, batchnorm=i < (len(kernel_sizes) - 1))
            for (i, kernel_size) in enumerate(kernel_sizes)
        ])
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x, lengths, mask):
        lx, hx = self.lstm(x)

        cx = x.transpose(1, 2)
        for conv_block in self.convs:
            cx = conv_block(cx)
        cx = cx.transpose(1, 2)

        x = lx + cx
        x = self.final_layer_norm(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

