from pathlib import Path

from phonetic_toolbox.services.lip_service import LipExtractionService


def test_resolve_target_prefers_builtin_script(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    builtin_script = (
        repo_root
        / "phonetic_toolbox"
        / "gui"
        / "dialogs"
        / "lip_feature_analysis_standalone.py"
    )
    builtin_script.parent.mkdir(parents=True)
    builtin_script.write_text("print('ok')", encoding="utf-8")

    project_dir = tmp_path / "lip_feature_analysis"
    project_dir.mkdir()
    script_path = project_dir / "lip_feature_analysis.py"
    script_path.write_text("print('ok')", encoding="utf-8")

    service = LipExtractionService(project_dir=str(project_dir))
    monkeypatch.setattr(service, "_repo_root", lambda: repo_root)

    target = service._resolve_target()

    assert target == builtin_script


def test_resolve_target_does_not_fallback_to_exe(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    project_dir = tmp_path / "lip_feature_analysis"
    project_dir.mkdir()
    exe_path = project_dir / "lip_feature_analysis.exe"
    exe_path.write_bytes(b"")

    service = LipExtractionService(project_dir=str(project_dir))
    monkeypatch.setattr(service, "_repo_root", lambda: repo_root)

    target = service._resolve_target()

    assert target is None


def test_build_command_for_python_file_adds_load_flag():
    service = LipExtractionService()
    target = Path(r"D:\project\lip_feature_analysis\lip_feature_analysis.py")

    command = service._build_command(target, load_existing=True)

    assert command[0].endswith("python.exe")
    assert command[1] == str(target)
    assert command[2] == "--load"


def test_build_command_uses_configured_python_env(monkeypatch):
    service = LipExtractionService()
    target = Path(r"D:\project\lip_feature_analysis\lip_feature_analysis.py")
    monkeypatch.setenv(
        "PHONETIC_TOOLBOX_LIP_PYTHON", r"D:\envs\phonetic_311\python.exe"
    )

    command = service._build_command(target, load_existing=False)

    assert command[0] == r"D:\envs\phonetic_311\python.exe"
    assert command[1] == str(target)


def test_resolve_python_executable_prefers_conda_env(monkeypatch):
    expected_path = Path(
        r"C:\Users\tester\Miniconda3\envs\phonetic_311\python.exe"
    )

    monkeypatch.delenv("PHONETIC_TOOLBOX_LIP_PYTHON", raising=False)
    monkeypatch.setattr(
        "phonetic_toolbox.services.lip_service.Path.home",
        lambda: Path(r"C:\Users\tester"),
    )

    original_exists = Path.exists

    def _patched_exists(path_obj: Path):
        if str(path_obj) == str(expected_path):
            return True
        return original_exists(path_obj)

    monkeypatch.setattr(Path, "exists", _patched_exists)

    resolved = LipExtractionService._resolve_python_executable()

    assert resolved == str(expected_path)


def test_launch_returns_failure_when_project_missing(monkeypatch):
    service = LipExtractionService(
        project_dir=str(Path(r"D:\not_exists\lip_feature_analysis"))
    )
    monkeypatch.setattr(service, "_candidate_script_paths", lambda: [])

    result = service.launch()

    assert result.success is False
    assert "未找到" in result.message


def test_launch_returns_success_and_process_id(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "lip_feature_analysis"
    project_dir.mkdir()
    script_path = project_dir / "lip_feature_analysis.py"
    script_path.write_text("print('ok')", encoding="utf-8")

    service = LipExtractionService(project_dir=str(project_dir))
    monkeypatch.setattr(service, "_candidate_script_paths", lambda: [script_path])

    class _FakeProcess:
        def __init__(self, pid: int):
            self.pid = pid

    def _fake_popen(command, cwd):
        assert command[1] == str(script_path)
        assert cwd == str(project_dir)
        return _FakeProcess(pid=2468)

    monkeypatch.setattr(
        "phonetic_toolbox.services.lip_service.subprocess.Popen", _fake_popen
    )

    result = service.launch(load_existing=False)

    assert result.success is True
    assert result.process_id == 2468
    assert result.target_path == str(script_path)


def test_launch_uses_self_exe_in_frozen_mode(monkeypatch):
    service = LipExtractionService()
    monkeypatch.setattr(service, "_is_frozen", lambda: True)
    monkeypatch.setattr(
        "phonetic_toolbox.services.lip_service.sys.executable",
        r"D:\dist\PhoneticToolbox.exe",
    )

    class _FakeProcess:
        def __init__(self, pid: int):
            self.pid = pid

    def _fake_popen(command, cwd):
        assert command[0] == r"D:\dist\PhoneticToolbox.exe"
        assert "--lip-standalone" in command
        assert command[-1] == "--load"
        assert cwd == r"D:\dist"
        return _FakeProcess(pid=1357)

    monkeypatch.setattr(
        "phonetic_toolbox.services.lip_service.subprocess.Popen", _fake_popen
    )

    result = service.launch(load_existing=True)

    assert result.success is True
    assert result.process_id == 1357
    assert result.target_path == r"D:\dist\PhoneticToolbox.exe"
