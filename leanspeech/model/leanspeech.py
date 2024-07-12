import torch
from torch import nn
from torch.nn import functional as F

from ..utils import sequence_mask, denormalize_mel
from .base_lightning_module import BaseLightningModule
from .components.length_regulator import GaussianUpsampling
from .components.losses import DurationPredictorLoss, MSEMelSpecReconstructionLoss


class LeanSpeech(BaseLightningModule):
    def __init__(self,
        dim,
        n_feats,
        sample_rate,
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

        self.encoder = encoder(dim=dim)
        self.duration_predictor = duration_predictor(dim=dim)
        self.length_regulator = GaussianUpsampling()
        self.decoder = decoder(
            n_mel_channels=n_feats,
            dim=dim,
        )
        self.dur_loss_criteria = DurationPredictorLoss()
        self.mel_loss = MSEMelSpecReconstructionLoss()
        self.w_mel_loss = loss_weights.mel
        self.w_dur_loss = loss_weights.duration
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
        x_lengths = x_lengths.long().to("cpu")
        y_lengths = y_lengths.long().to("cpu")

        # Encoder
        x = self.encoder(x, x_lengths, x_mask)

        # Duration predictor
        logw= self.duration_predictor(x, x_lengths, x_mask)
        dur_loss = self.dur_loss_criteria(logw, durations, x_mask.squeeze(1))

        # Length regulator
        z = self.length_regulator(
            x,
            durations,
            y_mask.squeeze(1).bool(),
            x_mask.squeeze(1).bool()
        )

        # Decoder
        y_hat = self.decoder(z, y_lengths, y_mask)

        # Mel loss
        mel_loss = self.mel_loss(y_hat, y, y_mask.bool())

        # Total loss
        loss = (dur_loss * self.w_dur_loss) + (mel_loss * self.w_mel_loss)

        return {
            "loss": loss,
            "dur_loss": dur_loss,
            "mel_loss": mel_loss,
        }

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
        x_lengths = x_lengths.long().to("cpu")

        # Encoder
        x = self.encoder(x, x_lengths, x_mask)

        # Duration predictor
        durations = self.duration_predictor.infer(x, x_lengths, x_mask)
        y_lengths = durations.sum(dim=1)
        y_max_length = y_lengths.max()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(x.dtype)

        # length regulator
        y = self.length_regulator(
            x,
            durations,
            y_mask.squeeze(1).bool(),
            x_mask.squeeze(1).bool(),
        )

        # Decoder
        y = self.decoder(y, y_lengths, y_mask)

        # Prepare outputs
        mel = denormalize_mel(y, self.mel_mean, self.mel_std)
        mel_lengths = y_lengths

        return {
            "mel": mel,
            "mel_lengths": mel_lengths,
            "durations": durations,
        }
