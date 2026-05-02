from pathlib import Path

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
