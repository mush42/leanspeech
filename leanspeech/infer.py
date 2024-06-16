import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from leanspeech.model import LeanSpeech
from leanspeech.text import process_and_phonemize_text_matcha, process_and_phonemize_text_piper
from leanspeech.utils import pylogger, plot_spectrogram_to_numpy


log = pylogger.get_pylogger(__name__)


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
    parser.add_argument("-l", "--lang", type=str, default='en-us', help="Language to use for tokenization.")
    parser.add_argument("--length-scale", type=float, default=1.0, help="Length scale to control speech rate.")
    parser.add_argument("-t", "--tokenizer", type=str, choices=["matcha", "piper"], default="matcha", help="Text tokenizer")
    parser.add_argument("--sr", type=int, default=22050, help="Mel spectogram sampleing rate")
    parser.add_argument("--hop", type=int, default=256, help="Mel spectogram hop-length")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    model = LeanSpeech.load_from_checkpoint(args.checkpoint, map_location="cpu")

    if args.tokenizer == "matcha":
            tokenizer = process_and_phonemize_text_matcha
    elif args.tokenizer == "piper":
        tokenizer = process_and_phonemize_text_piper
    else:
        log.error(f"Unknown tokenizer: `{args.tokenizer}`")
        exit(-1)

    phids, norm_text = tokenizer(args.text, args.lang, split_sentences=True)
    log.info(f"Normalized text: {norm_text}")
    x = []
    x_lengths = []
    for phid in phids:
        x.append(phid)
        x_lengths.append(len(phid))

    x = pad_sequence([torch.LongTensor(t) for t in x], batch_first=True).long()
    x_lengths = torch.LongTensor(x_lengths)
    scales = torch.Tensor([args.length_scale])

    t0 = perf_counter()
    mels, mel_lengths, w_ceil = model.synthesize(x, x_lengths)
    t_infer = perf_counter() - t0
    t_audio = (mel_lengths.sum().item() * args.hop) // (args.sr / 1000)
    ls_rtf = t_infer / t_audio
    log.info(f"LeanSpeech RTF: {ls_rtf}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for (i, mel) in enumerate(unpad_sequence(mels, mel_lengths, batch_first=True)):
        outfile = output_dir.joinpath(f"gen-{i + 1}")
        out_mel = outfile.with_suffix(".mel.npy")
        out_mel_plot = outfile.with_suffix(".png")
        mel = mel.detach().cpu().numpy()
        np.save(out_mel, mel, allow_pickle=False)
        plot_spectrogram_to_numpy(mel, out_mel_plot)
        log.info(f"Wrote mel to {out_mel}")


if __name__ == "__main__":
    main()
