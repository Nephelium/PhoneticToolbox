from pathlib import Path

from phonetic_toolbox.services.mfa_alignment_service import (
    MFAAutoAlignmentService,
)


def test_resolve_project_dir_with_explicit_path(tmp_path: Path):
    project_dir = tmp_path / "auto_alignment"
    project_dir.mkdir()
    batch_file = project_dir / "auto_alignment.bat"
    batch_file.write_text("@echo off", encoding="utf-8")

    service = MFAAutoAlignmentService(project_dir=str(project_dir))

    resolved = service._resolve_project_dir()

    assert resolved == project_dir


def test_validate_runtime_reports_missing_files(tmp_path: Path):
    project_dir = tmp_path / "auto_alignment"
    project_dir.mkdir()
    batch_file = project_dir / "auto_alignment.bat"
    batch_file.write_text("@echo off", encoding="utf-8")

    service = MFAAutoAlignmentService(project_dir=str(project_dir))

    ok, message = service._validate_runtime(project_dir)

    assert ok is False
    assert "activate.bat" in message


def test_launch_returns_failure_when_project_missing(monkeypatch):
    service = MFAAutoAlignmentService(
        project_dir=r"D:\not_exists\auto_alignment"
    )
    monkeypatch.setattr(service, "_resolve_project_dir", lambda: None)

    result = service.launch()

    assert result.success is False
    assert "auto_alignment.zip" in result.message


def test_launch_returns_success_and_process_id(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "auto_alignment"
    scripts_dir = project_dir / "env" / "Scripts"
    scripts_dir.mkdir(parents=True)
    batch_file = project_dir / "auto_alignment.bat"
    batch_file.write_text("@echo off", encoding="utf-8")
    (scripts_dir / "activate.bat").write_text("@echo off", encoding="utf-8")

    service = MFAAutoAlignmentService(project_dir=str(project_dir))

    class _FakeProcess:
        def __init__(self, pid: int):
            self.pid = pid

    def _fake_popen(command, cwd, shell):
        assert command == [str(batch_file)]
        assert cwd == str(project_dir)
        assert shell is True
        return _FakeProcess(pid=1357)

    monkeypatch.setattr(
        "phonetic_toolbox.services.mfa_alignment_service.subprocess.Popen",
        _fake_popen,
    )

    result = service.launch()

    assert result.success is True
    assert result.process_id == 1357
    assert result.batch_file == str(batch_file)


def test_run_alignment_validates_paths(tmp_path: Path):
    service = MFAAutoAlignmentService()

    result = service.run_alignment(
        audio_path=str(tmp_path / "missing"),
        dict_path=str(tmp_path / "dict.txt"),
        acoustic_path=str(tmp_path / "acoustic.zip"),
        output_path=str(tmp_path / "output"),
    )

    assert result.success is False
    assert "音频路径不是有效文件夹" in result.message


def test_run_alignment_calls_pipeline(tmp_path: Path, monkeypatch):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    dict_file = tmp_path / "dict.txt"
    dict_file.write_text("a b", encoding="utf-8")
    acoustic_file = tmp_path / "acoustic.zip"
    acoustic_file.write_text("zip", encoding="utf-8")
    output_dir = tmp_path / "output"

    class _FakePipeline:
        def run(self, audio_path, dict_path, acoustic_path, output_path):
            assert audio_path == str(audio_dir)
            assert dict_path == str(dict_file)
            assert acoustic_path == str(acoustic_file)
            assert output_path == str(output_dir)
            return True, "ok"

    monkeypatch.setattr(
        "phonetic_toolbox.services.mfa_alignment_service.MFAAlignmentPipeline",
        lambda: _FakePipeline(),
    )

    service = MFAAutoAlignmentService()
    result = service.run_alignment(
        audio_path=str(audio_dir),
        dict_path=str(dict_file),
        acoustic_path=str(acoustic_file),
        output_path=str(output_dir),
    )

    assert result.success is True
    assert result.message == "ok"
