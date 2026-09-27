# Direct V2 migration; source_id=PENDING-PHONOLOGY. See NOTICE.md and M14-source-map.md.
from __future__ import annotations

NAMES=('同音字表_韵母到声母.docx','同音字表_声母到韵母.docx','同音字表_二维表.xlsx')

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
