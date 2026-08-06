from pathlib import Path

import pandas as pd
import pytest

from phonetic_toolbox.models.config import AcousticConfig
from phonetic_toolbox.services.acoustic_service import AcousticAnalysisService, BatchAnalysisResult


def test_analyze_batch_returns_failed_files(tmp_path: Path):
    service = AcousticAnalysisService()

    result = service.analyze_batch(
        files=["missing.wav"],
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        lip_data_map={},
        config=AcousticConfig(),
    )

    assert isinstance(result, BatchAnalysisResult)
    assert result.processed == []
    assert len(result.failed) == 1
    assert result.failed[0].file_name == "missing.wav"
    assert "文件未找到" in result.failed[0].message


def test_save_results_propagates_excel_write_error(
    tmp_path: Path,
    monkeypatch,
):
    service = AcousticAnalysisService()
    result = pd.DataFrame({"Time_s": [0.0], "pF0": [120.0]})

    def _raise_locked(*_args, **_kwargs):
        raise OSError("file is locked")

    monkeypatch.setattr(pd.DataFrame, "to_excel", _raise_locked)

    with pytest.raises(OSError, match="locked"):
        service.save_results(result, str(tmp_path / "result.xlsx"))


def test_analyze_batch_marks_excel_write_error_as_failed(
    tmp_path: Path,
    monkeypatch,
):
    service = AcousticAnalysisService()
    monkeypatch.setattr(
        service,
        "analyze_file",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"Time_s": [0.0], "pF0": [120.0]},
        ),
    )

    def _raise_locked(*_args, **_kwargs):
        raise OSError("file is locked")

    monkeypatch.setattr(pd.DataFrame, "to_excel", _raise_locked)

    result = service.analyze_batch(
        files=["sample.wav"],
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        lip_data_map={},
        config=AcousticConfig(),
    )

    assert result.processed == []
    assert len(result.failed) == 1
    assert result.failed[0].file_name == "sample.wav"
    assert "locked" in result.failed[0].message
