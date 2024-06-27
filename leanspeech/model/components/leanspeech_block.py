import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .convnext import ConvNeXtBlock, Warehouse_Manager


class LeanSpeechBlock(nn.Module):
    def __init__(self,
        dim: int,
        stage_idx: int,
        warehouse_manager: Warehouse_Manager,
        *,
        num_conv_layers: int=1,
        intermediate_dim: int=None
    ):
        super().__init__()
        self.dim = dim
        self.warehouse_manager = warehouse_manager
        intermediate_dim = intermediate_dim or dim
        layer_scale_init_value = 1 / num_conv_layers
        self.lstm = nn.LSTM(
            dim, dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    layer_scale_init_value=layer_scale_init_value,
                    warehouse_manager=self.warehouse_manager,
                    stage_idx=stage_idx,
                    layer_idx=layer_idx,
                    intermediate_dim=intermediate_dim,
                )
                for layer_idx in range(num_conv_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(0.1)
        # self.apply(self._init_weights)

    def forward(self, x, lengths):
        lx, hx = self.lstm(x)
        lx = lx[:, :, :self.dim] + lx[:, :, self.dim:]
        lx = self.layer_norm(lx)

        cx = x.transpose(1, 2)
        for conv_block in self.convnext:
            cx = conv_block(cx)
        cx = cx.transpose(1, 2)

        x = lx + cx
        x = self.final_layer_norm(x)
        x = self.dropout(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

