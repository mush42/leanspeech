from typing import List, Tuple, Union

from piper_phonemize import phonemize_espeak

from ..utils import intersperse
from . import matcha_processor
from .textnorm import collapse_whitespace, preprocess_text


def process_and_phonemize_text_matcha(text: str, lang: str, *, split_sentences: bool=False) -> Tuple[Union[List[int], List[List[int]]], str]:
    text = matcha_processor.english_cleaners2(text)
    phonemes = phonemize_text(text, lang)
    if not split_sentences:
        phonemes = [
            phoneme
            for sentence_phonemes in phonemes
            for phoneme in sentence_phonemes
        ]
        phonemes = list(collapse_whitespace("".join(phonemes)))
        phoneme_ids = matcha_processor.text_to_sequence(phonemes)
        phoneme_ids = intersperse(phoneme_ids, 0)
    else:
        phoneme_ids = []
        for ph in phonemes:
            phonemes = list(collapse_whitespace("".join(ph)))
            phids = matcha_processor.text_to_sequence(ph)
            phids = intersperse(phids, 0)
            phoneme_ids.append(phids)
    return phoneme_ids, text


def process_and_phonemize_text_piper(text: str, lang: str, *, split_sentences: bool=False) -> Tuple[Union[List[int], List[List[int]]], str]:
    from piper_phonemize import phoneme_ids_espeak

    phonemes = phonemize_text(text, lang)
    if not split_sentences:
        phonemes = [
            phoneme
            for sentence_phonemes in phonemes
            for phoneme in sentence_phonemes
        ]
        phoneme_ids = phoneme_ids_espeak(phonemes)
    else:
        phoneme_ids = [phoneme_ids_espeak(ph) for ph in phonemes]
    return phoneme_ids, text


def phonemize_text(text: str, lang: str) -> str:
    # Normalize
    text = preprocess_text(text)
    # Phonemize
    phonemes = phonemize_espeak(text, lang)
    return phonemes


