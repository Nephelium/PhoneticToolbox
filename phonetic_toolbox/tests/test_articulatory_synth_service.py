from pathlib import Path

from phonetic_toolbox.services.articulatory_synth_service import (
    ArticulatorySynthService,
)


def test_resolve_project_dir_uses_runtime_base_dirs(
    tmp_path: Path, monkeypatch
):
    resource_dir = (
        tmp_path
        / "phonetic_toolbox"
        / "gui"
        / "resources"
        / "articulatory_synth"
    )
    resource_dir.mkdir(parents=True)

    service = ArticulatorySynthService(project_dir=None)
    monkeypatch.delenv(
        "PHONETIC_TOOLBOX_ARTICULATORY_SYNTH_DIR",
        raising=False,
    )
    monkeypatch.setattr(service, "_runtime_base_dirs", lambda: [tmp_path])
    monkeypatch.setattr(service, "_repo_root", lambda: tmp_path / "repo_root")

    resolved = service._resolve_project_dir()

    assert resolved == resource_dir


def test_resolve_project_dir_prefers_explicit_project_dir(
    tmp_path: Path, monkeypatch
):
    explicit_dir = tmp_path / "explicit"
    explicit_dir.mkdir()
    runtime_dir = (
        tmp_path
        / "runtime"
        / "phonetic_toolbox"
        / "gui"
        / "resources"
        / "articulatory_synth"
    )
    runtime_dir.mkdir(parents=True)

    service = ArticulatorySynthService(project_dir=str(explicit_dir))
    monkeypatch.delenv(
        "PHONETIC_TOOLBOX_ARTICULATORY_SYNTH_DIR",
        raising=False,
    )
    monkeypatch.setattr(
        service,
        "_runtime_base_dirs",
        lambda: [tmp_path / "runtime"],
    )
    monkeypatch.setattr(service, "_repo_root", lambda: tmp_path / "repo_root")

    resolved = service._resolve_project_dir()

    assert resolved == explicit_dir


def test_resolve_project_dir_uses_environment_override(
    tmp_path: Path, monkeypatch
):
    env_dir = tmp_path / "env_articulatory_synth"
    env_dir.mkdir()

    service = ArticulatorySynthService(project_dir=None)
    monkeypatch.setenv(
        "PHONETIC_TOOLBOX_ARTICULATORY_SYNTH_DIR",
        str(env_dir),
    )
    monkeypatch.setattr(service, "_runtime_base_dirs", lambda: [])
    monkeypatch.setattr(service, "_repo_root", lambda: tmp_path / "repo_root")

    resolved = service._resolve_project_dir()

    assert resolved == env_dir


def test_launch_returns_success_and_open_html(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "articulatory_synth"
    project_dir.mkdir()
    html_path = project_dir / "articulatory_synth.html"
    html_path.write_text("<html></html>", encoding="utf-8")

    service = ArticulatorySynthService(project_dir=str(project_dir))
    monkeypatch.setattr(service, "_open_html", lambda path: path == html_path)

    result = service.launch()

    assert result.success is True
    assert result.html_path == str(html_path)
    assert "已打开" in result.message


def test_launch_returns_failure_when_html_missing(tmp_path: Path):
    project_dir = tmp_path / "articulatory_synth"
    project_dir.mkdir()

    service = ArticulatorySynthService(project_dir=str(project_dir))
    result = service.launch()

    assert result.success is False
    assert result.working_directory == str(project_dir)
    assert "HTML" in result.message


def test_launch_returns_failure_when_open_html_failed(
    tmp_path: Path, monkeypatch
):
    project_dir = tmp_path / "articulatory_synth"
    project_dir.mkdir()
    html_path = project_dir / "articulatory_synth.html"
    html_path.write_text("<html></html>", encoding="utf-8")

    service = ArticulatorySynthService(project_dir=str(project_dir))
    monkeypatch.setattr(service, "_open_html", lambda path: False)

    result = service.launch()

    assert result.success is False
    assert result.html_path == str(html_path)
    assert "未能打开" in result.message
