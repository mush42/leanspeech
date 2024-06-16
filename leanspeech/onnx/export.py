import argparse
import random
from pathlib import Path

import numpy as np
import torch
from lightning import seed_everything

from leanspeech.model import LeanSpeech
from leanspeech.utils import pylogger, normalize_mel


log = pylogger.get_pylogger(__name__)
DEFAULT_OPSET = 16
DEFAULT_SEED = 1234


def main():
    parser = argparse.ArgumentParser(description="Export LeanSpeech checkpoints to ONNX")

    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the model checkpoint",
    )
    parser.add_argument("output", type=str, help="Path to output `.onnx` file")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version to use (default 15")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")

    args = parser.parse_args()
    seed_everything(args.seed)

    log.info(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint_path = Path(args.checkpoint_path)
    model = LeanSpeech.load_from_checkpoint(checkpoint_path, map_location="cpu")

    dummy_input_length = 50
    x = torch.randint(low=0, high=20, size=(1, dummy_input_length), dtype=torch.long)
    x_lengths = torch.LongTensor([dummy_input_length])

    # Scales
    length_scale = 1.0
    scales = torch.Tensor([length_scale])

    input_names = ["x", "x_lengths", "scales",]
    output_names = ["mel", "mel_lengths", "w_ceil"]

    # Set dynamic shape for inputs/outputs
    dynamic_axes = {
        "x": {0: "batch_size", 1: "time"},
        "x_lengths": {0: "batch_size"},
        "mel": {0: "batch_size", 2: "frames"},
        "mel_lengths": {0: "batch_size", 2: "frames"},
        "w_ceil": {0: "batch_size", 1: "time"},
    }

    # Create the output directory (if not exists)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    dummy_input = [x, x_lengths, scales]
    model.forward = lambda inputs: model.synthesize(inputs[0], inputs[1], inputs[2])
    model.to_onnx(
        args.output,
        dummy_input,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        # export_params=True,
        do_constant_folding=True,
    )
    log.info(f"ONNX model exported to  {args.output}")


if __name__ == "__main__":
    main()
