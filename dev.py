"""Module used during model development."""

import sys
import time
import torch
from lightning.pytorch.utilities.model_summary import summarize

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from leanspeech.text import process_and_phonemize_text_matcha, process_and_phonemize_text_piper


# Text processing pipeline
SENTENCE = "The history of the Galaxy has got a little muddled, for a number of reasons."
mat_phids, __ = process_and_phonemize_text_matcha(SENTENCE, "en-us")
print(f"Length of phoneme ids (Matcha): {len(mat_phids)}")
if sys.platform != 'win32':
    pip_phids, __ = process_and_phonemize_text_piper(SENTENCE, "en-us")
    print(f"Length of phoneme ids (piper): {len(pip_phids)}")

# Config pipeline
with initialize(version_base=None, config_path="./configs"):
    dataset_cfg = compose(config_name="data/hfc_female-en_US.yaml")
    cfg = compose(config_name="model/leanspeech.yaml")
    cfg.model.data_statistics = dict(
        mel_mean=-6.38385,
        mel_std=2.541796
    )

# Dataset pipeline
dataset_cfg.data.batch_size = 1
dataset_cfg.data.num_workers = 0
dataset_cfg.data.seed = 42
dataset = hydra.utils.instantiate(dataset_cfg.data)
dataset.setup()
vd = dataset.val_dataloader()
batch = next(iter(vd))
print(f"Batch['x'] shape: {batch['x'].shape}")
print(f"Batch['mel'] shape: {batch['y'].shape}")
print(f"Batch['durations'] shape: {batch['durations'].shape}")

# Model
model = hydra.utils.instantiate(cfg.model)
print(summarize(model))

# Training
x = batch["x"][0].unsqueeze(0)
x_lengths = torch.LongTensor([x.shape[-1]])
y = torch.rand(1, 80, 125)
y_lengths = torch.LongTensor([y.shape[-1]])
durations = torch.rand(1, x.size(1))
mel, loss, dur_loss, mel_loss = model(x, x_lengths, y, y_lengths, durations)

# Training loop
step_out = model.training_step(batch, 0)


# Inference
t0 = time.perf_counter()
mel, mel_lengths, w_ceil = model.synthesize(x, x_lengths)
t_infer = (time.perf_counter() - t0) * 1000
t_audio = (mel.shape[-1] * 256) / 22.05
rtf = t_infer / t_audio
print(f"RTF: {rtf}")
