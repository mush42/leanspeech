import csv
import json
import os
import random
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio as ta
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from leanspeech.text import process_and_phonemize_text_matcha, process_and_phonemize_text_piper
from leanspeech.utils import normalize_mel, fix_len_compatibility
from leanspeech.utils.audio import mel_spectrogram


def parse_filelist(filelist_path):
    filepaths = Path(filelist_path).read_text(encoding="utf-8").splitlines()
    filepaths = [f for f in filepaths if f.strip()]
    return filepaths


class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        language,
        text_processor,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.language = self.hparams.language

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        self.trainset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.language,
            self.hparams.text_processor,
            self.hparams.train_filelist_path,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
        )
        self.validset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.language,
            self.hparams.text_processor,
            self.hparams.valid_filelist_path,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        language,
        text_processor,
        filelist_path,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_parameters=None,
        cache_dir=None,
        seed=None,
    ):
        self.language = language

        if text_processor == "matcha":
            self.text_processor_func = process_and_phonemize_text_matcha
        elif text_processor == "piper":
            self.text_processor_func = process_and_phonemize_text_piper
        else:
            raise ValueError(f"Unknown text processor `{text_processor}``")

        self.file_paths = parse_filelist(filelist_path)
        self.data_dir = Path(filelist_path).parent.joinpath("data")
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max

        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = {"mel_mean": 0, "mel_std": 1}
        random.seed(seed)
        random.shuffle(self.file_paths)

    def get_datapoint(self, filepath):
        input_file = Path(filepath)
        json_filepath, mel_filepath, dur_filepath = (
            input_file.with_suffix(".json"),
            input_file.with_suffix(".mel.npy"),
            input_file.with_suffix(".dur.npy"),
        )
        with open(json_filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            phoneme_ids = data["phoneme_ids"]
            text = data["text"]
            x = torch.LongTensor(phoneme_ids)
        mel = torch.from_numpy(
            np.load(mel_filepath, allow_pickle=False)
        )
        durations = torch.from_numpy(
            np.load(dur_filepath, allow_pickle=False)
        )
        return {
            "x": x,
            "x_text": text,
            "y": mel,
            "durations": durations,
            "filepath": filepath,
        }

    def preprocess_utterance(self, audio_filepath, text):
        phoneme_ids, text = self.process_text(text)
        mel = self.get_mel(audio_filepath)
        durations = self.get_durations(audio_filepath, phoneme_ids)
        return phoneme_ids, text, mel, durations

    def get_durations(self, filepath, x):
        filepath = Path(filepath)
        data_dir, name = filepath.parent.parent, filepath.stem
        try:
            dur_loc = data_dir.joinpath("durations", name).with_suffix(".dur.npy")
            durs = torch.from_numpy(np.load(dur_loc).astype(int))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}, make sure you've generate the durations first "
            ) from e

        assert len(durs) == len(x), f"Length of durations {len(durs)} and phonemes {len(x)} do not match"

        return durs

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        ).squeeze()
        mel = normalize_mel(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel

    def process_text(self, text):
        phoneme_ids, clean_text = self.text_processor_func(text, self.language)
        return phoneme_ids, clean_text

    def __getitem__(self, index):
        filepath = self.file_paths[index]
        datapoint = self.get_datapoint(filepath)
        return datapoint

    def __len__(self):
        return len(self.file_paths)


class TextMelBatchCollate:

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])
        # y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        durations = torch.zeros((B, x_max_length), dtype=torch.long)

        y_lengths, x_lengths = [], []
        filepaths, x_texts = [], []
        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            durations[i, :item["durations"].shape[-1]] = item["durations"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)

        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "durations": durations,
        }
