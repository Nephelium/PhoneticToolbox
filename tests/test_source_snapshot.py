"""P19 build inputs stay fixed while the working tree continues to change."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('source_snapshot_test', ROOT / 'scripts/source_snapshot.py')
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


def sources(tmp_path):
    root = tmp_path / 'checkout'
    paths = {'scripts/v3_local_preview_entry.py': '# entry',
             'packages/phonetic_core/src/phonetic_core/__init__.py': 'value = 1',
             'backend/src/ptb_worker/child.py': '# child',
             'desktop/src/ptb_desktop/__init__.py': '# desktop'}
    for name, content in paths.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, 'utf8')
    return root


def test_analysis_reads_snapshot_after_checkout_changes(tmp_path):
    root = sources(tmp_path)
    target = tmp_path / 'frozen'
    manifest = snapshot.freeze(root, target)
    (root / 'packages/phonetic_core/src/phonetic_core/__init__.py').write_text('value = 2', 'utf8')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(snapshot.import_paths(target)), PYTHONNOUSERSITE='1')
    result = subprocess.run([sys.executable, '-B', '-c',
                             'import phonetic_core;print(phonetic_core.value);print(phonetic_core.__file__)'],
                            cwd=target, env=env, capture_output=True, text=True, check=True)
    value, origin = result.stdout.splitlines()
    assert value == '1'
    assert Path(origin).is_relative_to(target)
    assert snapshot.validate(target) == manifest


def test_mutation_during_copy_is_rejected(tmp_path, monkeypatch):
    root = sources(tmp_path)
    copy = snapshot.shutil.copyfile
    def changed(source, target):
        result = copy(source, target)
        source.write_text('changed during build', 'utf8')
        return result
    monkeypatch.setattr(snapshot.shutil, 'copyfile', changed)
    with pytest.raises(RuntimeError, match='changed while snapshotting'):
        snapshot.freeze(root, tmp_path / 'frozen')


def test_changed_snapshot_cannot_be_reported_as_original(tmp_path):
    root = sources(tmp_path)
    target = tmp_path / 'frozen'
    snapshot.freeze(root, target)
    (target / 'scripts/v3_local_preview_entry.py').write_text('changed', 'utf8')
    with pytest.raises(RuntimeError, match='identity changed'):
        snapshot.validate(target)
