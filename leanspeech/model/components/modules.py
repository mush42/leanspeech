import torch
from torch import nn
from torch.nn import functional as F

from .backbone import LeanSpeechBackbone
from .convs import ConvSeparable, build_activation


class LayerNormWithDim(nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1, eps=1e-12):
        """Construct an LayerNorm object."""
        super(LayerNormWithDim, self).__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNormWithDim, self).forward(x)
        return super(LayerNormWithDim, self).forward(x.transpose(1, -1)).transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, n_vocab, dim, layers, dropout, drop_path):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, dim, padding_idx=0)
        self.backbone = LeanSpeechBackbone(
            dim=dim,
            layers=layers,
            dropout=dropout,
            drop_path=drop_path,
        )

    def forward(self, x, lengths, mask):
        x = self.emb(x)
        x = self.backbone(x, lengths, mask)
        return x


class DurationPredictor(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size,
        intermediate_dim,
        num_layers,
        clip_val=1e-8,
        activation='relu',
        dropout=0.0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.clip_val = clip_val
        self.conv_blocks = nn.ModuleList([
            torch.nn.Sequential(
                ConvSeparable(dim if idx == 0 else intermediate_dim, intermediate_dim, self.kernel_size),
                build_activation(activation),
                LayerNormWithDim(intermediate_dim, dim=1),
                nn.Dropout(dropout)
            )
            for idx in range(num_layers)
        ])
        self.proj = torch.nn.Linear(intermediate_dim, 1)

    def forward(self, x, lengths, mask):
        x = x.transpose(1, 2)
        for block in self.conv_blocks:
            x = F.pad(x, [self.kernel_size // 2, self.kernel_size // 2])
            x = block(x)
            x = x * mask
        x = x.transpose(1, 2)
        x = self.proj(x).transpose(1, 2).squeeze(1)
        return x

    @torch.inference_mode()
    def infer(self, x, lengths, mask, factor=1.0):
        """
        Inference duration in linear domain.
        Args:
            x (Tensor):  (B, Tmax, H).
            lengths (Tensor): Batch of input lengths (B,).
            mask (Tensor): Batch of masks indicating padded part (B, 1, TMax).
            factor (float, optional): durations scale to control speech rate.
        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        log_durations = self(x, lengths, mask)
        # linear domain
        durations = (torch.exp(log_durations) - self.clip_val)
        durations = torch.ceil(durations) * factor
        # avoid negative values
        durations = torch.clamp(durations.long(), min=0) 
        mask = ~mask.squeeze(1).bool()
        durations = durations.masked_fill(mask, 0)
        return durations


class Decoder(nn.Module):
    def __init__(self, n_mel_channels, dim, layers, dropout, drop_path):
        super().__init__()
        self.backbone = LeanSpeechBackbone(
            dim=dim,
            layers=layers,
            dropout=dropout,
            drop_path=drop_path,
        )
        self.mel_linear = nn.Linear(dim, n_mel_channels)

    def forward(self, x, lengths, mask):
        x = self.backbone(x, lengths, mask)
        x = self.mel_linear(x)
        x = x.transpose(1, 2)
        return x

