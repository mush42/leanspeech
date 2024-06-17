import torch
from torch import nn
from torch.nn import functional as F

from ..utils import sequence_mask, denormalize_mel
from .base_lightning_module import BaseLightningModule
from .components.modules import TextEncoder, DurationPredictor, Decoder, LengthRegulator


class LeanSpeech(BaseLightningModule):
    def __init__(self,
        dim,
        n_vocab,
        n_feats,
        data_statistics,
        encoder,
        duration_predictor,
        decoder,
        loss_weights,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.encoder = TextEncoder(
            n_vocab=n_vocab,
            dim=dim,
            convnext_layers=encoder.convnext_layers,
            intermediate_dim=encoder.intermediate_dim
        )
        self.duration_predictor = DurationPredictor(
            dim=dim,
            convnext_layers=duration_predictor.convnext_layers,
        )
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder(
            n_mel_channels=n_feats,
            dim=dim,
            convnext_layers=decoder.convnext_layers,
            intermediate_dim=decoder.intermediate_dim
        )
        self.w_mel_loss = loss_weights["mel"]
        self.w_dur_loss = loss_weights["duration"]
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
            durations (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size, max_text_length)

        Returns:
            mel (torch.Tensor): predicted mel spectogram
                shape: (batch_size, mel_feats, n_timesteps)
            loss: (torch.Tensor): scaler representing total loss
            dur_loss: (torch.Tensor): scaler representing durations loss
            mel_loss: (torch.Tensor): scaler representing mel spectogram loss
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(1)), 1).to(x.dtype)
        y_max_length = y_lengths.max()
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)

        # Encoder
        x = self.encoder(x, x_mask)

        # Duration predictor
        logw= self.duration_predictor(x, x_mask)

        # Length regulator
        x, dur_loss, attn= self.length_regulator(x, x_lengths, x_mask, y, y_lengths, y_mask, logw, durations)

        # Decoder
        mel = self.decoder(x, y_mask)

        # Mel loss
        mel_mask = y_mask.bool()
        pred_mel = mel.masked_select(mel_mask)
        target = y.masked_select(mel_mask)
        mel_loss = F.mse_loss(pred_mel, target)

        # Total loss
        loss = (dur_loss * self.w_dur_loss) + (mel_loss * self.w_mel_loss)

        return mel, loss, dur_loss, mel_loss

    @torch.inference_mode()
    def synthesize(self, x, x_lengths, length_scale=1.0):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            length_scale (torch.Tensor): scaler to control phoneme durations.

        Returns:
            mel (torch.Tensor): predicted mel spectogram
                shape: (batch_size, mel_feats, n_timesteps)
            mel_lengths (torch.Tensor): lengths of generated mel spectograms
                shape: (batch_size,)
            w_ceil: (torch.Tensor): predicted phoneme durations
                shape: (batch_size, max_text_length)
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(1)), 1).to(x.dtype)

        # Encoder
        x = self.encoder(x, x_mask)

        # Duration predictor
        logw= self.duration_predictor(x, x_mask)

        # length regulator
        w_ceil, y, y_lengths, y_mask = self.length_regulator.infer(x, x_mask, logw, length_scale)

        # Decoder
        x = self.decoder(y, y_mask)

        # Prepare outputs
        mel = denormalize_mel(x, self.mel_mean, self.mel_std)
        mel_lengths = y_lengths
        w_ceil = w_ceil.squeeze(1)

        return mel, mel_lengths, w_ceil
