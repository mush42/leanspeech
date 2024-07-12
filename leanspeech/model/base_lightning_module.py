"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""

import inspect
from abc import ABC
from typing import Any, Dict

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from ..utils import get_pylogger
from ..utils.generic import plot_tensor


log = get_pylogger(__name__)


class BaseLightningModule(LightningModule, ABC):
    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
            data_statistics = {
                "mel_mean": 0.0,
                "mel_std": 1.0,
            }

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    def configure_optimizers(self):
        params = [
            {"params": self.parameters()},
        ]
        opt = self.hparams.optimizer(params)
        # Max steps per optimizer
        max_steps = self.trainer.max_steps
        scheduler = self.hparams.scheduler(
            opt,
            num_training_steps=max_steps,
            last_epoch=getattr("self", "ckpt_loaded_epoch", -1)
        )
        return (
            [opt],
            [{"scheduler": scheduler, "interval": "step"}],
        )

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]

        outputs = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            durations=batch["durations"],
        )
        return outputs

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        total_loss = loss_dict["loss"]

        self.log(
            "step",
            float(self.global_step),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "sub_loss/train_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_mel_loss",
            loss_dict["mel_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "loss/train",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": total_loss, "log": loss_dict}

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        total_loss =  loss_dict["loss"]
        self.log(
            "sub_loss/val_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_mel_loss",
            loss_dict["mel_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "loss/val",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"original_mel/{i}",
                        plot_tensor(y.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )

            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                output = self.synthesize(x[:, :x_lengths], x_lengths)
                mel = output["mel"]
                self.logger.experiment.add_image(
                    f"generated_mel/{i}",
                    plot_tensor(mel.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )

    def on_before_optimizer_step(self, optimizer):
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})
