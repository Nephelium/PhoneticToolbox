import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_paths import resolve_asset


def test_known_resource_can_be_read(tmp_path):
    (tmp_path / 'assets').mkdir()
    file = tmp_path / 'assets/app.js'
    file.write_text('export {}', encoding='utf-8')
    assert resolve_asset(tmp_path, '/assets/app.js') == file


@pytest.mark.parametrize('path', ['/../secret', '/%2e%2e/secret', '/C:/Windows/win.ini',
                                 '/assets/../../secret', '/assets\\..\\secret', '//server/share', '/x%00'])
def test_path_escape_is_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        resolve_asset(tmp_path, path)


def test_directories_and_missing_files_are_not_resources(tmp_path):
    with pytest.raises(ValueError):
        resolve_asset(tmp_path, '/')
    with pytest.raises(ValueError):
        resolve_asset(tmp_path, '/missing.js')
