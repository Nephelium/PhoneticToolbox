from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_runtime_and_project_versions_match():
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package_init = (
        PROJECT_ROOT / "phonetic_toolbox" / "__init__.py"
    ).read_text(encoding="utf-8")

    project_version = _extract(r'^version\s*=\s*"([^"]+)"', pyproject)
    runtime_version = _extract(r'^__version__\s*=\s*"([^"]+)"', package_init)

    assert project_version == runtime_version


def test_gui_fallback_and_executable_name_use_shared_version():
    main_window = (
        PROJECT_ROOT / "phonetic_toolbox" / "gui" / "main_window.py"
    ).read_text(encoding="utf-8")
    spec = (PROJECT_ROOT / "run.spec").read_text(encoding="utf-8")

    assert "from phonetic_toolbox import __version__" in main_window
    assert 'return f"v{__version__}"' in main_window
    assert "APP_VERSION" in spec
    assert 'name=f"PhoneticToolbox_v{APP_VERSION}"' in spec
