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
        self.lstm_dropout = nn.Dropout(0.1)
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

    def forward(self, x):
        lstm_out, lstm_state = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)
        conv_out = x.permute(0, 2, 1)
        for conv in self.convs:
            conv_out = conv(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        x = lstm_out + conv_out
        return x

if __name__ == '__main__':
    from ...utils.generic import count_parameters, get_model_size_mb

    dim = 256
    block = LeanSpeechBlock(dim, 5, 4)
    block = block.half()
    print(f"#Params: {count_parameters(block)}")
    print(f"Size (bytes): {get_model_size_mb(block)}")
    x = torch.rand(16, 100, dim).half()
    y = block(x)
    print(y.shape)
