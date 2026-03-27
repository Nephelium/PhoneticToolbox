from pathlib import Path

from phonetic_toolbox.core.transcription.phonology_induction import (
    PhonologyInductionParser,
)
from phonetic_toolbox.models.phonology_models import PhonologyInputRow
from phonetic_toolbox.services.phonology_service import PhonologyInductionService


def test_parser_splits_initial_final_tone():
    parser = PhonologyInductionParser()
    parsed = parser.parse("kʷʰaŋ35")
    assert parsed.initial == "kʷʰ"
    assert parsed.final == "aŋ"
    assert parsed.tone == "35"


def test_parser_handles_no_initial_and_no_vowel_cases():
    parser = PhonologyInductionParser()
    parsed_vowel_initial = parser.parse("i24")
    assert parsed_vowel_initial.initial == "Ø"
    assert parsed_vowel_initial.final == "i"
    assert parsed_vowel_initial.tone == "24"

    parsed_consonant_only = parser.parse("m35")
    assert parsed_consonant_only.initial == "Ø"
    assert parsed_consonant_only.final == "m"
    assert parsed_consonant_only.tone == "35"

    parsed_as_empty_final = parser.parse("m35", consonant_only_as_zero_initial=False)
    assert parsed_as_empty_final.initial == "m"
    assert parsed_as_empty_final.final == ""
    assert parsed_as_empty_final.tone == "35"

    parsed_retroflex = parser.parse("ɻ3", consonant_only_as_zero_initial=False)
    assert parsed_retroflex.initial == "ɻ"
    assert parsed_retroflex.final == ""

    parsed_double_l = parser.parse("ll2", consonant_only_as_zero_initial=False)
    assert parsed_double_l.initial == "l"
    assert parsed_double_l.final == "l"

    parsed_ts = parser.parse("ts3", consonant_only_as_zero_initial=False)
    assert parsed_ts.initial == "ts"
    assert parsed_ts.final == ""


def test_service_loads_text_and_analyzes(tmp_path: Path):
    text_path = tmp_path / "sample.txt"
    text_path.write_text(
        "汉字,音标,备注\n帽,mɔ6,\n葡,dɔ2,葡（bəʔ8）～\n岛\ttɔ1\t\n",
        encoding="utf-8",
    )
    service = PhonologyInductionService()
    rows = service.load_rows(str(text_path))
    assert len(rows) == 3
    analysis = service.analyze(rows)
    assert set(analysis.unique_tones) == {"1", "2", "6"}
    assert "mɔ6" in analysis.unique_ipa


def test_service_extracts_note_from_multi_character_headword(tmp_path: Path):
    text_path = tmp_path / "note_sample.txt"
    text_path.write_text(
        "妈妈,ma1,\n爸（爹）,ba4,爸爸\n",
        encoding="utf-8",
    )
    service = PhonologyInductionService()
    rows = service.load_rows(str(text_path))
    assert rows[0].character == "妈"
    assert rows[0].note == "妈妈"
    assert rows[1].character == "爸"
    assert rows[1].note == "爹，爸爸"


def test_empty_final_is_kept_in_unique_finals():
    service = PhonologyInductionService()
    rows = [
        PhonologyInputRow(character="日", ipa="ɻ3", note=""),
        PhonologyInputRow(character="妈", ipa="ma1", note=""),
    ]
    analysis = service.analyze(rows, consonant_only_as_zero_initial=False)
    assert "" in analysis.unique_finals
