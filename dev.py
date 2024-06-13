"""Module used during model development."""

import time
import torch
from lightning.pytorch.utilities.model_summary import summarize

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

with initialize(version_base=None, config_path="./configs"):
    cfg = compose(config_name="model/leanspeech.yaml")
    cfg.model.data_statistics = dict(
        mel_mean=-6.38385,
        mel_std=2.541796
    )
    model = hydra.utils.instantiate(cfg.model)

print(summarize(model))

dummy_input_length = 50
x = torch.randint(low=0, high=50, size=(1, dummy_input_length), dtype=torch.long)
x_lengths = torch.LongTensor([x.size(1)])
y = torch.rand(1, 80, 125)
y_lengths = torch.LongTensor([y.size(2)])
durations = torch.rand(1, x.size(1))
train_out = model(x, x_lengths, y, y_lengths, durations)


t0 = time.perf_counter()
mel, w_ceil = model.synthesize(x, x_lengths)
t_infer = (time.perf_counter() - t0) * 1000
t_audio = (mel.shape[-1] * 256) / 22.05
rtf = t_infer / t_audio
print(f"RTF: {rtf}")
