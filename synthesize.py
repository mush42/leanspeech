import sys
import time
from pathlib import Path

import numpy as np
import torch
import onnxruntime
import soundfile as sf

from leanspeech.model import LeanSpeech
from leanspeech.text import process_and_phonemize_text_matcha

ckpt = next(
    Path("./logs/train/")
    .glob("*/runs/*/checkpoints/last.ckpt")
)
global_step = torch.load(ckpt)["global_step"]
model = LeanSpeech.load_from_checkpoint(ckpt, map_location="cpu")

# Text processing pipeline
# SENTENCE = "By the name of Allah."
# SENTENCE = "A rainbow is a meteorological phenomenon that is caused by reflection, refraction and dispersion of light in water droplets resulting in a spectrum of light appearing in the sky."
# SENTENCE = "The history of the Galaxy has got a little muddled, for a number of reasons: partly because those who are trying to keep track of it have got a little muddled, but also because some very muddling things have been happening anyway."
SENTENCE = "Dialling in New Zealand takes quite a bit of concentration because every digit is where you least expect to find it. Try and do it quickly and you will inevitably misdial because your automatic habit jumps in and takes over before you have a chance to stop it. The habit of telephone dials is so deep that it has become an assumption, and you don't even know you're making it."
phoneme_ids, __ = process_and_phonemize_text_matcha(SENTENCE, "en-us")
print(f"Length of phoneme ids (Matcha): {len(phoneme_ids)}")

# Inference
t0 = time.perf_counter()
x = torch.LongTensor([phoneme_ids])
x_lengths = torch.LongTensor([len(phoneme_ids)])
mel, mel_length, w_ceil = model.synthesize(x, x_lengths)
t_infer = (time.perf_counter() - t0) * 1000
t_audio = (mel.shape[-1] * 256) / 22.05
rtf = t_infer / t_audio
print(f"RTF: {rtf}")

# Vocoder
voc_path = r"D:\lab\matcha_vocos\models\wavenext.onnx"
mel = mel.numpy()
voc = onnxruntime.InferenceSession(voc_path, providers=["CPUExecutionProvider"])
aud = voc.run(None, {"mels": mel})[0]
sf.write(f"data/leanspeech-step={global_step}.wav", aud.squeeze(), 22050)

