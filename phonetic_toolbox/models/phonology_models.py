from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhonologyInputRow:
    character: str
    ipa: str
    note: str = ""


@dataclass(frozen=True)
class ParsedPhonologyRow:
    character: str
    ipa: str
    note: str
    initial: str
    final: str
    tone_value: str


@dataclass(frozen=True)
class PhonologyAnalysisResult:
    rows: list[ParsedPhonologyRow]
    unique_initials: list[str]
    unique_finals: list[str]
    unique_tones: list[str]
    unique_ipa: list[str]


@dataclass(frozen=True)
class PhonologyOutputResult:
    forward_docx_path: str
    reverse_docx_path: str
    matrix_xlsx_path: str
