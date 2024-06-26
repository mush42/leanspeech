import torch
from torch import nn

from leanspeech.utils import fix_len_compatibility, duration_loss, sequence_mask, generate_path
from .leanspeech_block  import LeanSpeechBlock


class TextEncoder(nn.Module):
    def __init__(self, n_vocab, dim, convnext_layers):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, dim, padding_idx=0)
        self.ls_blocks = nn.ModuleList([
            LeanSpeechBlock(dim=dim, **layer_config)
            for layer_config in convnext_layers
        ])

    def forward(self, x, lengths, mask):
        x = self.emb(x)
        for ls_block in self.ls_blocks:
            x = ls_block(x, lengths)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, dim, convnext_layers):
        super().__init__()
        self.ls_blocks = nn.ModuleList([
            LeanSpeechBlock(dim=dim, **layer_config)
            for layer_config in convnext_layers
        ])
        self.proj = torch.nn.Linear(dim, 1)

    def forward(self, x, lengths, mask):
        org_x = x
        for ls_block in self.ls_blocks:
            x = x + ls_block(org_x, lengths)
        x = self.proj(x).transpose(1, 2)
        return x


class Decoder(nn.Module):
    def __init__(self, n_mel_channels, dim, convnext_layers):
        super().__init__()
        self.ls_blocks = nn.ModuleList([
            LeanSpeechBlock(dim=dim, **layer_config)
            for layer_config in convnext_layers
        ])
        self.mel_linear = nn.Linear(dim, n_mel_channels)

    def forward(self, x, lengths, mask):
        for ls_block in self.ls_blocks:
            x = ls_block(x, lengths)
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
        dur_loss = duration_loss(logw, logw_gt, x_lengths)
        return mu_y, dur_loss, attn

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
