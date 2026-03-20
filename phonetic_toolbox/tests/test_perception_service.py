from pathlib import Path

from phonetic_toolbox.services.perception_service import (
    PerceptionExperimentService,
)


def test_resolve_html_path_prefers_default_file(tmp_path: Path):
    project_dir = tmp_path / "perception_experiment"
    project_dir.mkdir()
    default_html = project_dir / "perception_experiment.html"
    another_html = project_dir / "another.html"
    default_html.write_text("<html></html>", encoding="utf-8")
    another_html.write_text("<html></html>", encoding="utf-8")

    resolved = PerceptionExperimentService._resolve_html_path(project_dir)

    assert resolved == default_html


def test_resolve_html_path_fallback_to_first_html(tmp_path: Path):
    project_dir = tmp_path / "perception_experiment"
    project_dir.mkdir()
    first_html = project_dir / "a_page.html"
    second_html = project_dir / "b_page.html"
    first_html.write_text("<html></html>", encoding="utf-8")
    second_html.write_text("<html></html>", encoding="utf-8")

    resolved = PerceptionExperimentService._resolve_html_path(project_dir)

    assert resolved == first_html


def test_launch_returns_failure_when_project_missing(monkeypatch):
    service = PerceptionExperimentService(
        project_dir=str(Path(r"D:\not_exists\perception_experiment"))
    )
    monkeypatch.setattr(service, "_resolve_project_dir", lambda: None)

    result = service.launch()

    assert result.success is False
    assert "未找到" in result.message


def test_launch_returns_success_and_open_html(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "perception_experiment"
    project_dir.mkdir()
    html_path = project_dir / "perception_experiment.html"
    html_path.write_text("<html></html>", encoding="utf-8")

    service = PerceptionExperimentService(project_dir=str(project_dir))
    monkeypatch.setattr(service, "_open_html", lambda path: path == html_path)

    result = service.launch()

    assert result.success is True
    assert result.html_path == str(html_path)


def test_launch_returns_failure_when_open_html_failed(
    tmp_path: Path, monkeypatch
):
    project_dir = tmp_path / "perception_experiment"
    project_dir.mkdir()
    html_path = project_dir / "perception_experiment.html"
    html_path.write_text("<html></html>", encoding="utf-8")

    service = PerceptionExperimentService(project_dir=str(project_dir))
    monkeypatch.setattr(service, "_open_html", lambda path: False)

    result = service.launch()

    assert result.success is False
    assert result.html_path == str(html_path)
    assert "未能打开" in result.message


def test_resolve_project_dir_uses_runtime_base_dirs(
    tmp_path: Path, monkeypatch
):
    project_dir = tmp_path / "perception_experiment"
    project_dir.mkdir()

    service = PerceptionExperimentService(project_dir=None)
    monkeypatch.delenv(
        "PHONETIC_TOOLBOX_PERCEPTION_PROJECT_DIR",
        raising=False,
    )
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

    resolved = service._resolve_project_dir()

    assert resolved == project_dir


def test_resolve_project_dir_prefers_resources_path(
    tmp_path: Path, monkeypatch
):
    repo_root = tmp_path / "repo_root"
    resources_dir = (
        repo_root
        / "phonetic_toolbox"
        / "gui"
        / "resources"
        / "perception_experiment"
    )
    resources_dir.mkdir(parents=True)

    fallback_dir = tmp_path / "perception_experiment"
    fallback_dir.mkdir()

    service = PerceptionExperimentService(project_dir=None)
    monkeypatch.delenv(
        "PHONETIC_TOOLBOX_PERCEPTION_PROJECT_DIR",
        raising=False,
    )
    monkeypatch.setattr(service, "_runtime_base_dirs", lambda: [tmp_path])
    monkeypatch.setattr(service, "_repo_root", lambda: repo_root)

    resolved = service._resolve_project_dir()

    assert resolved == resources_dir


def test_resolve_project_dir_uses_discover_fallback(
    tmp_path: Path, monkeypatch
):
    nested_root = tmp_path / "project"
    nested_target = nested_root / "abc" / "def" / "perception_experiment"
    nested_target.mkdir(parents=True)

    service = PerceptionExperimentService(project_dir=None)
    monkeypatch.delenv(
        "PHONETIC_TOOLBOX_PERCEPTION_PROJECT_DIR",
        raising=False,
    )
    monkeypatch.setattr(service, "_runtime_base_dirs", lambda: [])
    monkeypatch.setattr(service, "_repo_root", lambda: tmp_path / "repo_root")
    monkeypatch.setattr(service, "_common_search_roots", lambda: [nested_root])

    resolved = service._resolve_project_dir()

    assert resolved == nested_target


def test_find_named_dir_returns_none_when_depth_exceeded(tmp_path: Path):
    level_1 = tmp_path / "a"
    level_2 = level_1 / "b"
    level_3 = level_2 / "c"
    level_4 = level_3 / "d"
    level_5 = level_4 / "e"
    deep_target = level_5 / "perception_experiment"
    deep_target.mkdir(parents=True)

    found = PerceptionExperimentService._find_named_dir(
        root=tmp_path,
        target_name="perception_experiment",
        max_depth=4,
    )

    assert found is None
