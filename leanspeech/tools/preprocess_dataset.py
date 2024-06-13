import argparse
import csv
import os
from pathlib import Path

import rootutils
from leanspeech.utils import pylogger


log = pylogger.get_pylogger(__name__)


def main():
    root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="original data directory",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory to write train.txt and val.txt",
    )
    parser.add_argument(
        "--format",
        choices=["ljspeech"],
        default="ljspeech",
        help="Dataset format.",
    )
    args = parser.parse_args()

    if args.format != 'ljspeech':
        log.error(f"Unsupported dataset format `{args.format}`")
        exit(1)

    train_root = Path(args.input_dir).joinpath("train")
    val_root = Path(args.input_dir).joinpath("val")
    outputs = (
        ("train.txt", train_root),
        ("val.txt", val_root),
    )
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
        mel_path = root.joinpath("mel")
        out_rows = []
        for (itype, filestem, text) in inrows:
            if itype == "mel":
                filepath = mel_path.joinpath(filestem + ".mel.npy")
            elif itype == "wav":
                filepath = wav_path.joinpath(filestem + ".wav")
            else:
                log.warning(f"unknown entry type `{itype}` for file `{filestem}`. Skipping...")
                continue
            filepath = filepath.resolve()
            out_rows.append((filepath, text.strip()))
        out_txt = Path(args.output_dir).joinpath(out_filename)
        with open(out_txt, "w", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter="|")
            writer.writerows(out_rows)
        log.info(f"Wrote file: {out_txt}")

    log.info("Process done!")


if __name__ == "__main__":
    main()
