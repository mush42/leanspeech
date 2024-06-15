import torch
from torch import nn

from ..utils import sequence_mask, denormalize_mel
from .base_lightning_module import BaseLightningModule
from .components.modules import TextEncoder, DurationPredictor, Decoder
from .components.length_regulator import LengthRegulator
from .components.losses import DurationPredictorLoss


class LeanSpeech(BaseLightningModule):
    def __init__(self,
        dim,
        n_vocab,
        n_feats,
        data_statistics,
        encoder,
        duration_predictor,
        decoder,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.encoder = TextEncoder(
            n_vocab=n_vocab,
            dim=dim,
            kernel_sizes=encoder.kernel_sizes,
            intermediate_dim=encoder.intermediate_dim
        )
        self.duration_predictor = DurationPredictor(
            dim=dim,
            kernel_sizes=duration_predictor.kernel_sizes,
        )
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder(
            n_mel_channels=n_feats,
            dim=dim,
            kernel_sizes=decoder.kernel_sizes,
            intermediate_dim=decoder.intermediate_dim
        )
        self.duration_loss = DurationPredictorLoss()
        self.update_data_statistics(data_statistics)

    def forward(self, x, x_lengths, y, y_lengths, durations,):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
        Returns:
            loss (torch.Tensor): total loss
            mel (torch.Tensor): predicted mel spectogram
                shape: (batch_size, n_timesteps, mel_feats)
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(1)), 1).to(x.dtype)
        y_max_length = y_lengths.max()
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)

        # Encoder
        x = self.encoder(x, x_lengths)

        # Duration predictor
        logw= self.duration_predictor(x, x_mask)
        dur_loss = self.duration_loss(logw.squeeze(2), durations, x_mask)

        # LR: length regulator
        w = torch.exp(logw)
        w_ceil = torch.ceil(w).long()
        x, mel_length = self.length_regulator(x, w_ceil, y_max_length)
        
        # Decoder
        x = self.decoder(x)
        pred_mel = x.permute(0, 2, 1)

        # Mel loss
        mel_mask = y_mask.bool()
        target = y.masked_select(mel_mask)
        pred = pred_mel.masked_select(mel_mask)
        mel_loss = nn.L1Loss()(pred, target)

        # Total loss
        loss = dur_loss + mel_loss

        return pred_mel, loss, dur_loss, mel_loss

    @torch.inference_mode()
    def synthesize(self, x, x_lengths, length_scale=1.0):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(1)), 1).to(x.dtype)
        # Encoder
        x = self.encoder(x, x_lengths)

        # Duration predictor
        logw= self.duration_predictor(x, x_mask)

        # LR: length regulator
        w = torch.exp(logw)
        w_ceil = torch.ceil(w).long()
        x, mel_length = self.length_regulator(x, w_ceil)

        # Decoder
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        mel = denormalize_mel(x, self.mel_mean, self.mel_std)

        return mel, w_ceil
