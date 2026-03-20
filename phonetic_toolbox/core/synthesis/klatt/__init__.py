from .klatt_config import DEFAULT_DURATION, DEFAULT_FS, FADE_MS, PARAM_DEFAULTS
from .input_parser import VOWEL_FORMANTS, ParsedSegment, parse_vowel_sequence
from .spectral_filter import SpectralFilter
from .tdklatt import KlattParam1980, klatt_make

__all__ = [
    "DEFAULT_DURATION",
    "DEFAULT_FS",
    "FADE_MS",
    "PARAM_DEFAULTS",
    "VOWEL_FORMANTS",
    "ParsedSegment",
    "parse_vowel_sequence",
    "SpectralFilter",
    "KlattParam1980",
    "klatt_make",
]
