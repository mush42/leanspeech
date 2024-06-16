import argparse
from hashlib import md5
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from leanspeech.hifigan.config import v1
from leanspeech.hifigan.env import AttrDict
from leanspeech.hifigan.models import Generator as HiFiGAN
from leanspeech.model import LeanSpeech
from leanspeech.text import process_and_phonemize_text_matcha, process_and_phonemize_text_piper
from leanspeech.utils import pylogger, plot_spectrogram_to_numpy


log = pylogger.get_pylogger(__name__)


def load_hifigan(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


def main():
    parser = argparse.ArgumentParser(description=" Synthesizing text using LeanSpeech")

    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to LeanSpeech checkpoint",
    )
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to write generated mel  to.",
    )
    parser.add_argument("--length-scale", type=float, default=1.0, help="Length scale to control speech rate.")
    parser.add_argument("--hfg-checkpoint", type=str, default=None, help="HiFiGAN vocoder V1 checkpoint.")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    datamodule_params = torch.load(args.checkpoint)["datamodule_hyper_parameters"]
    language = datamodule_params["language"]
    sample_rate = datamodule_params["sample_rate"]
    hop_length = datamodule_params["hop_length"]
    tokenizer_name = datamodule_params["text_processor"]
    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    model = LeanSpeech.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.to(device)
    hfg_vocoder = None
    if args.hfg_checkpoint is not None:
        hfg_vocoder = load_hifigan(args.hfg_checkpoint, device)

    if tokenizer_name == "matcha":
            tokenizer = process_and_phonemize_text_matcha
    elif tokenizer_name == "piper":
        tokenizer = process_and_phonemize_text_piper
    else:
        log.error(f"Unknown tokenizer: `{tokenizer_name}`")
        exit(-1)

    phids, norm_text = tokenizer(args.text, language, split_sentences=True)
    log.info(f"Normalized text: {norm_text}")
    x = []
    x_lengths = []
    for phid in phids:
        x.append(phid)
        x_lengths.append(len(phid))

    x = pad_sequence([torch.LongTensor(t) for t in x], batch_first=True).long().to(device)
    x_lengths = torch.LongTensor(x_lengths).to(device)
    scales = torch.Tensor([args.length_scale]).to(device)

    t0 = perf_counter()
    mels, mel_lengths, w_ceil = model.synthesize(x, x_lengths)
    t_infer = perf_counter() - t0
    t_audio = (mel_lengths.sum().item() * hop_length) / sample_rate
    ls_rtf = t_infer / t_audio
    log.info(f"LeanSpeech RTF: {ls_rtf}")

    mels = mels.detach().cpu()
    mel_lengths = mel_lengths.detach().cpu()
    w_ceil = w_ceil.detach().cpu()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for (i, mel) in enumerate(unpad_sequence(mels, mel_lengths, batch_first=True)):
        file_hash = md5(mel.numpy().tobytes()).hexdigest()[:8]
        outfile = output_dir.joinpath(f"gen-{i + 1}-" + file_hash)
        out_mel = outfile.with_suffix(".mel.npy")
        out_mel_plot = outfile.with_suffix(".png")
        out_wav = outfile.with_suffix(".wav")
        mel = mel.detach().cpu().numpy()
        np.save(out_mel, mel, allow_pickle=False)
        plot_spectrogram_to_numpy(mel, out_mel_plot)
        log.info(f"Wrote mel to {out_mel}")
        if hfg_vocoder is not None:
            aud = hfg_vocoder(torch.from_numpy(mel).unsqueeze(0).to(device))
            wav = aud.squeeze().detach().cpu().numpy()
            sf.write(out_wav, wav, sample_rate)
            log.info(f"Wrote audio to {out_wav}")


if __name__ == "__main__":
    main()
