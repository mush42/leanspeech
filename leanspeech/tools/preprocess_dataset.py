import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import hydra
import rootutils
from hydra import compose, initialize
from leanspeech.dataset import TextMelDataset
from leanspeech.utils import get_script_logger
from tqdm import tqdm


log = get_script_logger(__name__)


def write_data(data_dir, file_stem, data):
    phoneme_ids, text, mel, durations = data

    output_file = data_dir.joinpath(file_stem)
    out_json = output_file.with_suffix(".json")
    out_mel = output_file.with_suffix(".mel.npy")
    out_dur = output_file.with_suffix(".dur.npy")

    with open(out_json, "w", encoding="utf-8") as file:
        ph_text_data = {
            "phoneme_ids": phoneme_ids,
            "text": text,
        }
        json.dump(ph_text_data, file, ensure_ascii=False)

    np.save(out_mel, mel, allow_pickle=False)
    np.save(out_dur, durations, allow_pickle=False)


def main():
    root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="dataset config relative to `configs/data/` (without the suffix)",
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="original data directory",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory to write datafiles + train.txt and val.txt",
    )
    parser.add_argument(
        "--format",
        choices=["ljspeech"],
        default="ljspeech",
        help="Dataset format.",
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../../configs/data"):
        cfg = compose(config_name=args.dataset)
        cfg["seed"] = 1234
    dataset = TextMelDataset(
        language=cfg.language,
        text_processor=cfg.text_processor,
        filelist_path=os.devnull,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_feats,
        sample_rate=cfg.sample_rate,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        data_parameters=cfg.data_statistics,
    )

    if args.format != 'ljspeech':
        log.error(f"Unsupported dataset format `{args.format}`")
        exit(1)

    train_root = Path(args.input_dir).joinpath("train")
    val_root = Path(args.input_dir).joinpath("val")
    outputs = (
        ("train.txt", train_root),
        ("val.txt", val_root),
    )
    output_dir = Path(args.output_dir)
    if output_dir.is_dir():
        log.error(f"Output directory {output_dir} already exist. Stopping")
        exit(1)
    output_dir.mkdir(parents=True)
    data_dir = output_dir.joinpath("data")
    data_dir.mkdir()
    for (out_filename, root) in outputs:
        if not root.is_dir():
            log.warning(f"Datasplit `{root.name}` not found. Skipping...")
            continue
        log.info(f"Extracting datasplit `{root.name}`")
        with open(root.joinpath("metadata.csv"), "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="|")
            inrows = list(reader)
        log.info(f"Found {len(inrows)} utterances in file.")
        wav_path = root.joinpath("wav")
        out_rows = []
        for (filestem, text) in tqdm(inrows, total=len(inrows), desc="processing", unit="utterance"):
            audio_path = wav_path.joinpath(filestem + ".wav")
            audio_path = audio_path.resolve()
            data = dataset.preprocess_utterance(audio_path, text)
            write_data(data_dir, audio_path.stem, data)
            out_rows.append((filestem, text.strip()))
        out_txt = output_dir.joinpath(out_filename)
        with open(out_txt, "w", encoding="utf-8", newline="\n") as file:
            writer = csv.writer(file, delimiter="|")
            writer.writerows(out_rows)
        log.info(f"Wrote file: {out_txt}")

    log.info("Process done!")


if __name__ == "__main__":
    main()
