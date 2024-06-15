import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import onnxruntime
import rootutils
from more_itertools import chunked
from tqdm import tqdm

from leanspeech.text import process_and_phonemize_text_matcha
from leanspeech.utils import pylogger, normalize_mel


log = pylogger.get_pylogger(__name__)


ONNX_CUDA_PROVIDERS = [
    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
    "CPUExecutionProvider"
]
ONNX_CPU_PROVIDERS = ["CPUExecutionProvider",]


def pad_sequences(sequences, maxlen=None, value=0):
  """Pads a list of sequences to the same length using broadcasting.

  Args:
    sequences: A list of Python lists with variable lengths.
    maxlen: The maximum length to pad the sequences to. If not specified,
      the maximum length of all sequences in the list will be used.
    value: The value to use for padding (default 0).

  Returns:
    A numpy array with shape [batch_size, maxlen] where the sequences are padded
    with the specified value.
  """

  # Get the maximum length if not specified
  if maxlen is None:
    maxlen = max(len(seq) for seq in sequences)

  # Create a numpy array with the specified value and broadcast
  padded_seqs = np.full((len(sequences), maxlen), value)
  for i, seq in enumerate(sequences):
    padded_seqs[i, :len(seq)] = seq

  return padded_seqs


def unpad_sequences(sequences, lengths):
  """Unpads a list of sequences based on a list of lengths.

  Args:
    sequences: A numpy array with shape [batch_size, feature_dim, max_len].
    lengths: A numpy array with shape [batch_size] representing the lengths
      of each sequence in the batch.

  Returns:
    A list of unpadded sequences. The i-th element of the list corresponds
    to the i-th sequence in the batch. Each sequence is a numpy array with
    variable length.
  """

  # Check if lengths argument is a list or 1D numpy array
  if not isinstance(lengths, np.ndarray) or len(lengths.shape) != 1:
    raise ValueError('lengths must be a 1D numpy array')

  # Check if sequence lengths are within bounds
  if np.any(lengths < 0) or np.any(lengths > sequences.shape[-1]):
    raise ValueError('lengths must be between 0 and max_len')

  # Get the batch size
  batch_size = sequences.shape[0]

  # Extract unpadded sequences
  unpadded_seqs = []
  for i in range(batch_size):
    unpadded_seqs.append(sequences[i, :, :lengths[i]])

  return unpadded_seqs




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

        x = pad_sequences(phoneme_ids).astype(np.int64)
        x_lengths = np.array(lengths, dtype=np.int64)
        matcha_inputs = {
            "x": x,
            "x_lengths": x_lengths,
            "scales": scales
        }
        mel_specs, mel_lengths, w_ceil = matcha.run(None, matcha_inputs)
        mels = unpad_sequences(mel_specs, mel_lengths)
        durs = unpad_sequences(w_ceil, x_lengths)
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
            np.save(out_dur, d, allow_pickle=False)

    log.info(f"Total mel lengths: {total_mel_len} frames")

    with open(output_dir.joinpath("filelist.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(filelist))
        log.info(f"Wrote filelist to `filelist.txt`")

    log.info(f"Wrote dataset to {args.output_directory}")


if __name__ == '__main__':
    main()
