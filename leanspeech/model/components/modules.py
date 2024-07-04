import torch
from torch import nn
from torch.nn import functional as F
from leanspeech.utils import fix_len_compatibility, sequence_mask, generate_path

from .leanspeech_block  import LeanSpeechBlock
from .conv_sep import ConvSeparable, build_activation


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
    def __init__(self, n_vocab, dim, layer_config):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, dim, padding_idx=0)
        self.leanspeech_blocks = nn.ModuleList([
            LeanSpeechBlock(dim, kernel_sizes)
            for kernel_sizes in layer_config.arch
        ])

    def forward(self, x, lengths, mask):
        x = self.emb(x)
        for block in self.leanspeech_blocks:
            x = block(x, lengths, mask)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, dim, layer_config):
        super().__init__()
        self.kernel_size = layer_config.kernel_size
        num_layers = layer_config.num_layers
        intermediate_dim = layer_config.intermediate_dim
        dropout = layer_config.dropout
        activation = layer_config.activation
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
        x = x.transpose(1, 2)
        x = self.proj(x).transpose(1, 2).squeeze(1)
        return x


class Decoder(nn.Module):
    def __init__(self, n_mel_channels, dim, layer_config):
        super().__init__()
        self.leanspeech_blocks = nn.ModuleList([
            LeanSpeechBlock(dim, kernel_sizes)
            for kernel_sizes in layer_config.arch
        ])
        self.mel_linear = nn.Linear(dim, n_mel_channels)

    def forward(self, x, lengths, mask):
        for block in self.leanspeech_blocks:
            x = block(x, lengths, mask)
        x = self.mel_linear(x)
        x = x.transpose(1, 2)
        return x


class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_lengths, x_mask, y, y_lengths, y_mask, logw, durations):
        y_max_length = y.shape[-1]
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.float().transpose(1, 2), x)
        mu_y = mu_y[:, :y_max_length, :]
        attn = attn[:, :, :y_max_length]
        # Compute loss between predicted log-scaled durations and the ground truth durations
        logw_gt = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        return mu_y, attn

    @torch.inference_mode
    def infer(self, x, x_mask, logw, length_scale=1.0):
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)
        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        mu_y = torch.matmul(attn.float().squeeze(1).transpose(1, 2), x)
        y = mu_y[:, :y_max_length, :]
        y_mask = y_mask[:, :, :y_max_length]
        return w_ceil, attn, y, y_lengths, y_mask
