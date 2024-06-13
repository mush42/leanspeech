import torch
from torch import nn

from .leanspeech_block  import LeanSpeechBlock


class TextEncoder(nn.Module):
    def __init__(self, n_vocab, dim, kernel_sizes, intermediate_dim=None):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, dim, padding_idx=0)
        self.ls_blocks = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.ls_blocks.append(
                LeanSpeechBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    num_conv_layers=1,
                    intermediate_dim=intermediate_dim
                )
            )

    def forward(self, x, x_mask):
        x = self.emb(x)
        for ls_block in self.ls_blocks:
            x = ls_block(x)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, dim, kernel_sizes, intermediate_dim=None):
        super().__init__()
        self.ls_blocks = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.ls_blocks.append(
                LeanSpeechBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    num_conv_layers=1,
                    intermediate_dim=intermediate_dim
                )
            )
        self.proj = torch.nn.Linear(dim, 1)

    def forward(self, x, x_mask):
        for ls_block in self.ls_blocks:
            x = ls_block(x)
        x = self.proj(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, n_mel_channels, dim, kernel_sizes, intermediate_dim=None
    ):
        super().__init__()
        self.ls_blocks = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.ls_blocks.append(
                LeanSpeechBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    num_conv_layers=1,
                    intermediate_dim=intermediate_dim
                )
            )
        self.mel_linear = nn.Linear(dim, n_mel_channels)

    def forward(self, x):
        for ls_block in self.ls_blocks:
            x = ls_block(x)
        x = self.mel_linear(x)
        return x

