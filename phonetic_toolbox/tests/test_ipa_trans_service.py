from pathlib import Path

from phonetic_toolbox.services.ipa_trans_service import IPATransService


def test_resolve_resource_dir_uses_runtime_base_dirs(
    tmp_path: Path, monkeypatch
):
    resource_dir = (
        tmp_path
        / "phonetic_toolbox"
        / "gui"
        / "resources"
        / "ipa_trans"
    )
    resource_dir.mkdir(parents=True)

    service = IPATransService(project_dir=None)
    monkeypatch.delenv("PHONETIC_TOOLBOX_IPA_PROJECT_DIR", raising=False)
    monkeypatch.setattr(
        service,
        "_runtime_base_dirs",
        lambda: [tmp_path],
    )
    monkeypatch.setattr(
        service,
        "_repo_root",
        lambda: tmp_path / "repo_root",
    )

    resolved = service._resolve_resource_dir()

    assert resolved == resource_dir


def test_launch_success_when_html_exists_without_script(
    tmp_path: Path, monkeypatch
):
    resource_dir = tmp_path / "ipa_trans"
    resource_dir.mkdir()
    html_path = resource_dir / "ipa_converter.html"
    html_path.write_text("<html></html>", encoding="utf-8")

    service = IPATransService(project_dir=str(resource_dir))
    monkeypatch.setattr(service, "_open_html", lambda path: path == html_path)

    result = service.launch()

    assert result.success is True
    assert result.html_path == str(html_path)
    assert "直接打开" in result.message


def test_launch_failure_when_open_html_failed(tmp_path: Path, monkeypatch):
    resource_dir = tmp_path / "ipa_trans"
    resource_dir.mkdir()
    html_path = resource_dir / "ipa_converter.html"
    html_path.write_text("<html></html>", encoding="utf-8")

    service = IPATransService(project_dir=str(resource_dir))
    monkeypatch.setattr(service, "_open_html", lambda path: False)

    result = service.launch()

    assert result.success is False
    assert result.html_path == str(html_path)
    assert "未能打开" in result.message


def test_launch_runs_generator_when_needed(tmp_path: Path, monkeypatch):
    resource_dir = tmp_path / "ipa_trans"
    resource_dir.mkdir()
    script_path = resource_dir / "generate_ipa_website.py"
    html_path = resource_dir / "ipa_converter.html"
    script_path.write_text("print('ok')", encoding="utf-8")

    class ProcessResult:
        def __init__(self):
            self.returncode = 0
            self.stdout = b"ok"
            self.stderr = b""

    def fake_run(command, cwd, capture_output):
        html_path.write_text("<html></html>", encoding="utf-8")
        return ProcessResult()

    service = IPATransService(project_dir=str(resource_dir))
    monkeypatch.setattr(
        "phonetic_toolbox.services.ipa_trans_service.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        service,
        "_resolve_python_executable",
        lambda: "python",
    )
    monkeypatch.setattr(service, "_open_html", lambda path: path == html_path)

    result = service.launch()

    assert result.success is True
    assert result.command == ["python", str(script_path)]
    assert "重新生成" in result.message


def test_launch_failure_when_no_html_and_no_script(tmp_path: Path):
    resource_dir = tmp_path / "ipa_trans"
    resource_dir.mkdir()

    service = IPATransService(project_dir=str(resource_dir))
    result = service.launch()

    assert result.success is False
    assert "未找到 IPA 页面文件" in result.message
