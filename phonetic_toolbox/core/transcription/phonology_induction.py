from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from phonetic_toolbox.core.synthesis.klatt.input_parser import VOWEL_FORMANTS


ZERO_INITIAL = "Ø"


@dataclass(frozen=True)
class ParsedSyllable:
    initial: str
    final: str
    tone: str


class PhonologyInductionParser:
    def __init__(self):
        extra_vowels = {
            "ɿ",
            "ʅ",
            "ʮ",
            "ʯ",
            "ᴀ",
            "ᴇ",
            "ɚ",
            "ɝ",
            "ɐ",
            "ə",
            "ɨ",
            "ɪ",
            "ʊ",
            "ʉ",
            "ʏ",
            "ɯ",
            "ɶ",
            "ʌ",
            "æ",
            "ɑ",
            "ɒ",
            "ɜ",
            "ɞ",
            "œ",
            "ɵ",
            "ø",
            "ɘ",
            "ɤ",
            "a",
            "e",
            "i",
            "o",
            "u",
            "y",
        }
        self.vowel_symbols = set(VOWEL_FORMANTS.keys()) | extra_vowels
        self.consonant_priority_symbols = {
            "m",
            "ɱ",
            "n",
            "ɳ",
            "ɲ",
            "ŋ",
            "ɴ",
            "p",
            "b",
            "t",
            "d",
            "ʈ",
            "ɖ",
            "c",
            "ɟ",
            "k",
            "ɡ",
            "q",
            "ɢ",
            "ʔ",
            "s",
            "z",
            "ʃ",
            "ʒ",
            "ʂ",
            "ʐ",
            "ɕ",
            "ʑ",
            "f",
            "v",
            "x",
            "ɣ",
            "χ",
            "ʁ",
            "h",
            "ɦ",
            "ɹ",
            "ɻ",
            "j",
            "ɰ",
            "w",
            "ʋ",
            "r",
            "l",
            "ɭ",
            "ʎ",
            "ʟ",
            "ɾ",
            "ɽ",
            "ɺ",
            "ȶ",
            "ȡ",
            "ȵ",
        }
        self.modifier_letters = {
            "ʰ",
            "ʷ",
            "ʲ",
            "ˠ",
            "ˤ",
            "˞",
            "ʼ",
            "ⁿ",
            "ˡ",
            "ᶿ",
            "ˣ",
            "ᵊ",
            "̩",
            "̍",
            "̯",
            "̚",
            "̥",
            "̬",
            "̊",
            "̤",
            "̰",
            "̪",
            "̼",
            "̺",
            "̻",
            "̟",
            "̠",
            "˖",
            "˗",
            "̈",
            "̽",
            "̝",
            "̞",
            "˔",
            "˕",
            "̹",
            "̜",
            "͗",
            "͑",
            "̃",
            "̘",
            "̙",
            "̴",
        }
        self.affricate_like_clusters = sorted(
            {
                "ts",
                "dz",
                "t̠ʃ",
                "d̠ʒ",
                "ʈʂ",
                "tʂ",
                "ɖʐ",
                "dʐ",
                "tɕ",
                "dʑ",
                "pɸ",
                "bβ",
                "p̪f",
                "b̪v",
                "t̪θ",
                "d̪ð",
                "tɹ̝̊",
                "dɹ̝",
                "t̠ɹ̠̊˔",
                "d̠ɹ̠˔",
                "cç",
                "ɟʝ",
                "kx",
                "ɡɣ",
                "qχ",
                "ʡʢ",
                "ʔh",
                "tɬ",
                "dɮ",
                "ʈɭ̊˔",
                "cʎ̝̊",
                "kʟ̝̊",
                "ɡʟ̝",
                "tsʼ",
                "t̠ʃʼ",
                "ʈʂʼ",
                "kxʼ",
                "qχʼ",
                "tɬʼ",
                "cʎ̝̊ʼ",
                "kʟ̝̊ʼ",
            },
            key=len,
            reverse=True,
        )

    def parse(
        self, ipa_text: str, consonant_only_as_zero_initial: bool = True
    ) -> ParsedSyllable:
        clean = self._normalize_ipa(ipa_text)
        base, tone = self._split_tone(clean)
        if not base:
            return ParsedSyllable(initial=ZERO_INITIAL, final="", tone=tone or "0")

        if self._is_single_consonant_syllable(base):
            if consonant_only_as_zero_initial:
                return ParsedSyllable(initial=ZERO_INITIAL, final=base, tone=tone or "0")
            return ParsedSyllable(initial=base, final="", tone=tone or "0")

        first_vowel_index = self._find_first_vowel_index(base)
        if first_vowel_index < 0:
            initial, remainder = self._split_first_consonant_unit(base)
            if not initial:
                return ParsedSyllable(initial=ZERO_INITIAL, final=base, tone=tone or "0")
            if not remainder:
                if consonant_only_as_zero_initial:
                    return ParsedSyllable(initial=ZERO_INITIAL, final=base, tone=tone or "0")
                return ParsedSyllable(initial=initial, final="", tone=tone or "0")
            return ParsedSyllable(initial=initial, final=remainder, tone=tone or "0")
        if first_vowel_index == 0:
            return ParsedSyllable(initial=ZERO_INITIAL, final=base, tone=tone or "0")

        initial = base[:first_vowel_index]
        final = base[first_vowel_index:]
        return ParsedSyllable(
            initial=initial if initial else ZERO_INITIAL,
            final=final,
            tone=tone or "0",
        )

    def is_single_consonant_syllable(self, ipa_text: str) -> bool:
        clean = self._normalize_ipa(ipa_text)
        base, _ = self._split_tone(clean)
        return self._is_single_consonant_syllable(base)

    def _split_tone(self, text: str) -> tuple[str, str]:
        match = re.search(r"([0-9]+)$", text)
        if match is None:
            return text, ""
        tone = match.group(1)
        return text[: -len(tone)], tone

    def _find_first_vowel_index(self, text: str) -> int:
        for idx, ch in enumerate(text):
            if ch in self.vowel_symbols:
                return idx
        return -1

    def _is_single_consonant_syllable(self, text: str) -> bool:
        cluster = self._match_affricate_cluster(text)
        if cluster:
            remainder = text[len(cluster) :]
            return self._has_only_modifiers(remainder)

        letters = [
            ch
            for ch in text
            if unicodedata.category(ch).startswith("L")
            and ch not in {"ː", "ˑ"}
            and ch not in self.modifier_letters
        ]
        if len(letters) != 1:
            return False
        symbol = letters[0]
        if symbol in self.consonant_priority_symbols:
            return True
        return symbol not in self.vowel_symbols

    def _split_first_consonant_unit(self, text: str) -> tuple[str, str]:
        cluster = self._match_affricate_cluster(text)
        if cluster:
            idx = len(cluster)
            while idx < len(text) and self._is_modifier_char(text[idx]):
                idx += 1
            return text[:idx], text[idx:]
        if not text:
            return "", ""
        idx = 1
        while idx < len(text) and self._is_modifier_char(text[idx]):
            idx += 1
        return text[:idx], text[idx:]

    def _match_affricate_cluster(self, text: str) -> str:
        for cluster in self.affricate_like_clusters:
            if text.startswith(cluster):
                return cluster
        return ""

    def _has_only_modifiers(self, text: str) -> bool:
        for ch in text:
            if not self._is_modifier_char(ch):
                return False
        return True

    def _is_modifier_char(self, ch: str) -> bool:
        if ch in self.modifier_letters:
            return True
        category = unicodedata.category(ch)
        return category.startswith("M")

    def _normalize_ipa(self, ipa_text: str) -> str:
        if not ipa_text:
            return ""
        translation = str.maketrans("０１２３４５６７８９", "0123456789")
        clean = ipa_text.translate(translation)
        clean = clean.strip()
        clean = clean.strip("[]/")
        clean = re.sub(r"\s+", "", clean)
        return clean
