## LeanSpeech

Unofficial pytorch implementation of [LeanSpeech: The Microsoft Lightweight Speech Synthesis System for Limmits Challenge 2023](https://ieeexplore.ieee.org/document/10096039)

**LeanSpeech** is ment to be an ultra **efficient**, **lightweight** and **fast** acoustic model for **on-device** text-to-speech.

## Installation

```bash
$ git clone https://github.com/mush42/leanspeech
$ cd leanspeech
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip3 install --upgrade pip setuptools wheels
$ pip3 install -r requirements.txt
```

## Inference

### Command line API

```bash
$ python3 -m leanspeech.infer  --help
usage: infer.py [-h] [--length-scale LENGTH_SCALE] [--hfg-checkpoint HFG_CHECKPOINT] [--cuda]
                checkpoint text output_dir

Synthesizing text using LeanSpeech

positional arguments:
  checkpoint            Path to LeanSpeech checkpoint
  text                  Text to synthesize
  output_dir            Directory to write generated mel to.

options:
  -h, --help            show this help message and exit
  --length-scale LENGTH_SCALE
                        Length scale to control speech rate.
  --hfg-checkpoint HFG_CHECKPOINT
                        HiFiGAN vocoder V1 checkpoint.
  --cuda                Use GPU for inference
```

### Python API

```python
from leanspeech.model import LeanSpeech
from leanspeech.text import process_and_phonemize_text_matcha

# Load model
ckpt_path = "/path/to/checkpoint"
model = LeanSpeech.load_from_checkpoint(ckpt_path, map_location="cpu")

# Text preprocessing and phonemization
sentence = "A rainbow is a meteorological phenomenon that is caused by reflection, refraction and dispersion of light in water droplets resulting in a spectrum of light appearing in the sky."
phoneme_ids, cleaned_text = process_and_phonemize_text_matcha(sentence, "en-us")

# Inference
x = torch.LongTensor([phoneme_ids])
x_lengths = torch.LongTensor([len(phoneme_ids)])
mel, mel_length, w_ceil = model.synthesize(x, x_lengths)
```

## Training

Since this code uses [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template), you have all the powers that come with it.

### Approach

LeanSpeech is trained using a **knowledge distilation** approach. A data augmentation approach is used, whereby ground truth data and **synthetic** data obtained from a powerful acoustic model are jointly used to train the model.

### Data preparation

Given a dataset that is organized as follows:

```bash
├── train
│   ├── metadata.csv
│   └── wav
│       ├── aud-00001-0003.wav
│       └── ...
└── val
    ├── metadata.csv
    └── wav
        ├── aud-00764.wav
        └── ...
```

You can use the `preprocess_dataset` script to prepare the ground truth dataset for training:

```bash
$ python3 -m leanspeech.tools.preprocess_dataset --help
usage: preprocess_dataset.py [-h] [--format {ljspeech}] dataset input_dir output_dir

positional arguments:
  dataset              dataset config relative to `configs/data/` (without the suffix)
  input_dir            original data directory
  output_dir           Output directory to write datafiles + train.txt and val.txt

options:
  -h, --help           show this help message and exit
  --format {ljspeech}  Dataset format.
```

### Preparing synthetic dataset

See example [matcha_synthetic](./leanspeech/tools/matcha_synthetic.py) for how to generate a synthetic dataset  from  [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) given a list of sentences.

Note: If you generate a synthetic dataset, you should merge it with ground-truth dataset manually by merging  the content of `train.txt` and `val.txt`.

### Starting training

To start training run the following command. Note that this training run uses **config** from [hfc_female-en_US](./configs/experiment/hfc_female-en_US.yaml). You can copy and update it with your own config values, and pass the name of the custom config file (without extension) instead.

```bash
$ python3 -m leanspeech.train experiment=hfc_female-en_US
``` 

## ONNX support

### ONNX export

```bash
$ python3 -m leanspeech.onnx.export --help
usage: export.py [-h] [--opset OPSET] [--seed SEED] checkpoint_path output

Export LeanSpeech checkpoints to ONNX

positional arguments:
  checkpoint_path  Path to the model checkpoint
  output           Path to output `.onnx` file

options:
  -h, --help       show this help message and exit
  --opset OPSET    ONNX opset version to use (default 15
  --seed SEED      Random seed
```

### ONNX inference

```bash
$ python3 -m leanspeech.onnx.infer --help
usage: infer.py [-h] [-l LANG] [--length-scale LENGTH_SCALE] [-t {matcha,piper}] [--sr SR] [--hop HOP]
                [-voc VOCODER] [--cuda]
                onnx_path text output_dir

ONNX inference of LeanSpeech

positional arguments:
  onnx_path             Path to the exported LeanSpeech ONNX model
  text                  Text to synthesize
  output_dir            Directory to write generated mel and/or audio to.

options:
  -h, --help            show this help message and exit
  -l LANG, --lang LANG  Language to use for tokenization.
  --length-scale LENGTH_SCALE
                        Length scale to control speech rate.
  -t {matcha,piper}, --tokenizer {matcha,piper}
                        Text tokenizer
  --sr SR               Mel spectogram sampleing rate
  --hop HOP             Mel spectogram hop-length
  -voc VOCODER, --vocoder VOCODER
                        Path to vocoder ONNX model
  --cuda                Use GPU for inference
```

## Acknowledgements

Repositories I would like to acknowledge:

- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS): For the repo backbone and phoneme-mel alignment framework.
- [Piper-TTS](https://github.com/rhasspy/piper): For leading the charge in on-device TTS. Also for the great phonemizer.
- [Vocos](https://github.com/gemelo-ai/vocos/): For pioneering the use of ConvNext in TTS.

## Reference

```
@INPROCEEDINGS{10096039,
  author={Zhang, Chen and Bansal, Shubham and Lakhera, Aakash and Li, Jinzhu and Wang, Gang and Satpal, Sandeepkumar and Zhao, Sheng and He, Lei},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={LeanSpeech: The Microsoft Lightweight Speech Synthesis System for Limmits Challenge 2023}, 
  year={2023},
  pages={1-2},
  keywords={Training;Convolution;Vocoders;Acoustics;Speech synthesis;Text to Speech;Lightweight;Multi-speaker;Multi-lingual;WaveGlow},
  doi={10.1109/ICASSP49357.2023.10096039}}
```

## Licence

Copyright (c) Musharraf Omer. MIT Licence. See [LICENSE](./LICENSE) for more details.
