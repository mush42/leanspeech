import re
import unicodedata
from typing import List, Tuple

from piper_phonemize import phonemize_espeak, phoneme_ids_espeak

from . import matcha_processor
from ..utils import intersperse

WHITESPACE_RE = re.compile(r"\s+")


def process_and_phonemize_text_matcha(text: str, lang: str) -> Tuple[List[int], str]:
    text = matcha_processor.english_cleaners2(text)
    phonemes = phonemize_text(text, lang)
    phonemes = list(collapse_whitespace("".join(phonemes)))
    phoneme_ids = matcha_processor.text_to_sequence(phonemes)
    phoneme_ids = intersperse(phoneme_ids, 0)
    return phoneme_ids, text



def process_and_phonemize_text_piper(text: str, lang: str) -> Tuple[List[int], str]:
    phonemes = phonemize_text(text, lang)
    phoneme_ids = phoneme_ids_espeak(phonemes)
    return phoneme_ids, text


def phonemize_text(text: str, lang: str) -> str:
    # Normalize
    text = preprocess_text(text)
    # Phonemize
    all_phonemes = phonemize_espeak(text, lang)
    phonemes = [
        phoneme
        for sentence_phonemes in all_phonemes
        for phoneme in sentence_phonemes
    ]
    return phonemes


def preprocess_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = collapse_whitespace(text)
    return text


def collapse_whitespace(text):
    text = re.sub(WHITESPACE_RE, " ", text)
    return text

