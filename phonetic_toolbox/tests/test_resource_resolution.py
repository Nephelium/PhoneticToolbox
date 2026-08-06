from __future__ import annotations

from pathlib import Path

from phonetic_toolbox.core.acoustic import f0_reaper


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_find_reaper_binary_in_meipass_from_unrelated_cwd(
    tmp_path: Path,
    monkeypatch,
):
    meipass = tmp_path / "bundle"
    bundled_reaper = (
        meipass
        / "phonetic_toolbox"
        / "core"
        / "acoustic"
        / "reaper.exe"
    )
    bundled_reaper.parent.mkdir(parents=True)
    bundled_reaper.write_bytes(b"reaper")

    unrelated_cwd = tmp_path / "cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    monkeypatch.setattr(f0_reaper.sys, "_MEIPASS", str(meipass), raising=False)
    monkeypatch.setattr(
        f0_reaper,
        "__file__",
        str(tmp_path / "missing" / "f0_reaper.py"),
    )
    monkeypatch.setattr(f0_reaper.shutil, "which", lambda _name: None)

    resolved = f0_reaper._find_reaper_bin(
        "phonetic_toolbox/core/acoustic/reaper.exe",
    )

    assert Path(resolved) == bundled_reaper.resolve()


def test_gui_resources_contain_no_project_machine_paths():
    gui_root = PROJECT_ROOT / "phonetic_toolbox" / "gui"
    markers = (
        "d:\\phonetictoolbox",
        "d:/phonetictoolbox",
        "file:///d:/phonetictoolbox",
    )

    for path in gui_root.rglob("*"):
        if path.suffix.lower() not in {".py", ".html"}:
            continue
        source = path.read_text(encoding="utf-8").lower()
        for marker in markers:
            assert marker not in source, f"{path} contains {marker}"


def test_python_help_entries_use_resource_resolver():
    files = [
        PROJECT_ROOT
        / "phonetic_toolbox"
        / "gui"
        / "widgets"
        / "parameter_estimation_widget.py",
        PROJECT_ROOT
        / "phonetic_toolbox"
        / "gui"
        / "dialogs"
        / "parameter_tools_dialog.py",
        PROJECT_ROOT / "phonetic_toolbox" / "gui" / "widgets" / "egg_widget.py",
        PROJECT_ROOT
        / "phonetic_toolbox"
        / "gui"
        / "widgets"
        / "parameter_display_widget.py",
        PROJECT_ROOT
        / "phonetic_toolbox"
        / "gui"
        / "widgets"
        / "pitch_manipulation_widget.py",
        PROJECT_ROOT
        / "phonetic_toolbox"
        / "gui"
        / "widgets"
        / "spec2wav_widget.py",
        PROJECT_ROOT
        / "phonetic_toolbox"
        / "gui"
        / "widgets"
        / "speech_synthesis_widget.py",
    ]

    for path in files:
        source = path.read_text(encoding="utf-8")
        assert "get_resource_path" in source


def test_pyinstaller_spec_uses_spec_directory_not_checkout_path():
    source = (PROJECT_ROOT / "run.spec").read_text(encoding="utf-8")

    assert "D:" not in source
    assert "PROJECT_ROOT = Path(SPECPATH)" in source
