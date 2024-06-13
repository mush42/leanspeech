import re
import unicodedata
from typing import List, Tuple

from piper_phonemize import phonemize_espeak, phoneme_ids_espeak


WHITESPACE_RE = re.compile(r"\s+")


def phonemize(text: str, lang: str) -> Tuple[List[int], str]:
    # Normalize
    text = unicodedata.normalize("NFD", text)
    text = collapse_whitespace(text)
    # Phomemize
    all_phonemes = phonemize_espeak(text, language)
    phonemes = [
        phoneme
        for sentence_phonemes in all_phonemes
        for phoneme in sentence_phonemes
    ]
    phoneme_ids = phoneme_ids_espeak(phonemes)
    return phoneme_ids, text


def collapse_whitespace(text):
    text = re.sub(_whitespace_re, " ", text)
    return text

