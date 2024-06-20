import argparse
import json
import os
import sys
from pathlib import Path

import rootutils
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from tqdm.auto import tqdm

from leanspeech.data.text_mel_datamodule import TextMelDataModule
from leanspeech.utils import get_script_logger

log = get_script_logger(__name__)


def compute_data_statistics(data_loader: torch.utils.data.DataLoader, out_channels: int):
    """Generate data mean and standard deviation helpful in data normalisation

    Args:
        data_loader (torch.utils.data.Dataloader): _description_
        out_channels (int): mel spectrogram channels
    """
    total_mel_sum = 0
    total_mel_sq_sum = 0
    total_mel_len = 0

    for batch in tqdm(data_loader, leave=False):
        mels = batch["y"]
        mel_lengths = batch["y_lengths"]

        total_mel_len += torch.sum(mel_lengths)
        total_mel_sum += torch.sum(mels)
        total_mel_sq_sum += torch.sum(torch.pow(mels, 2))

    data_mean = total_mel_sum / (total_mel_len * out_channels)
    data_std = torch.sqrt((total_mel_sq_sum / (total_mel_len * out_channels)) - torch.pow(data_mean, 2))

    return {"mel_mean": data_mean.item(), "mel_std": data_std.item()}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-config",
        type=str,
        help="The name of the yaml config file under configs/data",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default="256",
        help="Can have increased batch size for faster computation",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        required=False,
        help="force overwrite the file",
    )
    args = parser.parse_args()
    output_file = Path(args.input_config).with_suffix(".json")

    if os.path.exists(output_file) and not args.force:
        print("File already exists. Use -f to force overwrite")
        sys.exit(1)

    with initialize(version_base="1.3", config_path="../../configs/data"):
        cfg = compose(config_name=args.input_config, return_hydra_config=True, overrides=[])

    root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    with open_dict(cfg):
        del cfg["hydra"]
        del cfg["_target_"]
        cfg["data_statistics"] = None
        cfg["seed"] = 1234
        cfg["batch_size"] = args.batch_size
        cfg["train_filelist_path"] = str(os.path.join(root_path, cfg["train_filelist_path"]))
        cfg["valid_filelist_path"] = str(os.path.join(root_path, cfg["valid_filelist_path"]))

    text_mel_datamodule = TextMelDataModule(**cfg)
    text_mel_datamodule.setup()
    data_loader = text_mel_datamodule.train_dataloader()
    log.info("Dataloader loaded! Now computing stats...")
    params = compute_data_statistics(data_loader, cfg["n_feats"])
    print(params)
    json.dump(
        params,
        open(output_file, "w"),
    )


if __name__ == "__main__":
    main()
