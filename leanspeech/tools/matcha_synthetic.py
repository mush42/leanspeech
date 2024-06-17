import argparse
import itertools
import json
import os
import random
from pathlib import Path

import numpy as np
import onnxruntime
import rootutils
from more_itertools import chunked
from tqdm import tqdm

from leanspeech.text import process_and_phonemize_text_matcha
from leanspeech.utils import pylogger, normalize_mel, numpy_pad_sequences, numpy_unpad_sequences


log = pylogger.get_pylogger(__name__)


ONNX_CUDA_PROVIDERS = [
    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
    "CPUExecutionProvider"
]
ONNX_CPU_PROVIDERS = ["CPUExecutionProvider",]


def main():
    root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    parser = argparse.ArgumentParser()

    parser.add_argument("matcha_onnx", type=str, help="Exported matcha checkpoint.")
    parser.add_argument("language", type=str, help="Language to use for phonemization.")
    parser.add_argument("text_file", type=str, help="Text file containing lines to synthesize.")
    parser.add_argument("output_directory", type=str, help="Directory to write files to.")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for onnx inference.")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Number of sentences to synthesize in one run."
    )
    parser.add_argument(
        "--mel-mean", type=float, default=0.0, help="dataset-specific mel mean"
    )
    parser.add_argument(
        "--mel-std", type=float, default=1.0, help="dataset-specific mel std"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.667, help="Matcha temperature."
    )
    parser.add_argument(
        "--length-scale", type=float, default=1.0, help="Matcha temperature."
    )
    parser.add_argument(
        "--val-percent", type=float, default=0.05, help="Percentage of sentences for validation"
    )

    args = parser.parse_args()

    sents = Path(args.text_file).read_text(encoding="utf-8").splitlines()
    sents = [s.strip() for s in sents if s.strip()]

    log.info(f"Found {len(sents)} sentences")

    log.info(f"Loading matcha checkpoint from: {args.matcha_onnx}")
    matcha = onnxruntime.InferenceSession(
        args.matcha_onnx,
        providers=ONNX_CUDA_PROVIDERS if args.cuda else ONNX_CPU_PROVIDERS
    )
    scales = np.array([args.temperature, args.length_scale], dtype=np.float32)

    total_mel_len = 0
    batch_iterator = tqdm(
        enumerate(chunked(sents, args.batch_size)),
        total=len(sents) // args.batch_size,
        desc="synthesizing",
        unit="batch"
    )
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir = output_dir.joinpath("data")
    data_output_dir.mkdir()
    filelist = []
    item_counter = itertools.count()
    for (idx, batch) in batch_iterator:
        phoneme_ids = []
        lengths = []
        texts = []
        jsons = []
        for sent in batch:
            phids, text = process_and_phonemize_text_matcha(sent, args.language)
            phoneme_ids.append(phids)
            lengths.append(len(phids))
            texts .append(text)
            json_data = {"phoneme_ids": phids, "text": text}
            jsons.append(json_data)

        x = numpy_pad_sequences(phoneme_ids).astype(np.int64)
        x_lengths = np.array(lengths, dtype=np.int64)
        matcha_inputs = {
            "x": x,
            "x_lengths": x_lengths,
            "scales": scales
        }
        mel_specs, mel_lengths, w_ceil = matcha.run(None, matcha_inputs)
        mels = numpy_unpad_sequences(mel_specs, mel_lengths)
        durs = numpy_unpad_sequences(w_ceil, x_lengths)
        total_mel_len += mel_lengths.sum().item()

        for (j, m, d) in zip(jsons, mels, durs):
            item_number = next(item_counter)
            file_stem = "gen-" + str(item_number).rjust(5, "0")
            filelist.append(file_stem)
            outfile = data_output_dir.joinpath(file_stem)
            out_json = outfile.with_suffix(".json")
            out_mel = outfile.with_suffix(".mel.npy")
            out_dur = outfile.with_suffix(".dur.npy")
            with open(out_json, "w", encoding="utf-8") as file:
                json.dump(j, file, ensure_ascii=False)
            mel = normalize_mel(m, args.mel_mean, args.mel_std)
            np.save(out_mel, mel, allow_pickle=False)
            np.save(out_dur, d.squeeze(), allow_pickle=False)

    log.info(f"Total mel lengths: {total_mel_len} frames")

    random.shuffle(filelist)
    val_limit = int(len(filelist) * args.val_percent)
    train_split = filelist[val_limit:]
    val_split = filelist[:val_limit]
    with open(output_dir.joinpath("filelist.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(filelist))
        log.info(f"Wrote filelist to `filelist.txt`")

    with open(output_dir.joinpath("train.txt"), "w", encoding="utf-8") as file:
        train_filepaths = [
            os.fspath(
                data_output_dir.joinpath(fname).absolute()
            )
            for fname in train_split
        ]
        file.write("\n".join(train_filepaths))
        log.info(f"Wrote file paths to `train.txt`")

    with open(output_dir.joinpath("val.txt"), "w", encoding="utf-8") as file:
        val_filepaths = [
            os.fspath(
                data_output_dir.joinpath(fname).absolute()
            )
            for fname in val_split
        ]
        file.write("\n".join(val_filepaths))
        log.info(f"Wrote file paths to `val.txt`")

    log.info(f"Wrote dataset to {args.output_directory}")


if __name__ == '__main__':
    main()
