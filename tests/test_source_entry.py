"""P19: a stale installed package must never win over the source checkout."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('workbench_source_test', ROOT / 'scripts/workbench_source.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_ambient_old_core_is_overridden_and_children_get_current_source(tmp_path):
    old = tmp_path / 'phonetic_core'
    old.mkdir()
    (old / '__init__.py').write_text("raise RuntimeError('stale package imported')", encoding='utf8')
    code = '''import sys,json,subprocess,importlib.util
sys.path.insert(0, sys.argv[1])
from workbench_source import bind_sources
first=bind_sources()
child=subprocess.run([sys.executable,'-B','-c',"import phonetic_core;print(phonetic_core.__file__)"],capture_output=True,text=True,check=True)
print(json.dumps({'main':first['phonetic_core'],'child':child.stdout.strip()}))
'''
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTHONNOUSERSITE='1')
    result = subprocess.run([sys.executable, '-B', '-c', code, str(ROOT / 'scripts')],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    expected = ROOT / 'packages/phonetic_core/src/phonetic_core/__init__.py'
    assert Path(value['main']) == expected
    assert Path(value['child']) == expected


def test_already_loaded_old_submodule_is_rejected(monkeypatch, tmp_path):
    old = ModuleType('phonetic_core.stale')
    old.__file__ = str(tmp_path / 'stale.py')
    monkeypatch.setitem(sys.modules, 'phonetic_core.stale', old)
    with pytest.raises(RuntimeError, match='already imported outside'):
        entry.bind_sources()


def test_missing_source_has_no_installed_fallback(tmp_path):
    with pytest.raises(RuntimeError, match='Missing current source'):
        entry.bind_sources(tmp_path)


def test_checked_in_wrappers_read_the_same_sources_from_another_directory(tmp_path):
    if sys.platform != 'win32':
        pytest.skip('PowerShell development entry is Windows only')
    wrappers = sorted((ROOT / 'scripts').glob('Start-*-Workbench.ps1'))
    assert len(wrappers) >= 14
    for wrapper in wrappers:
        result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                                 '-File', str(wrapper), '-CheckOnly'], cwd=tmp_path,
                                capture_output=True, text=True, encoding='utf8', timeout=30)
        assert result.returncode == 0, (wrapper.name, result.stderr)
        report = json.loads(result.stdout)
        for package, relative in entry.PACKAGES.items():
            assert Path(report['sources'][package]) == ROOT / relative / '__init__.py'
        assert Path(report['python']) == ROOT / '.venv/m14/Scripts/python.exe'
        for key, relative in entry.RUNTIMES.items():
            assert Path(report['runtimes'][key]) == ROOT / relative


def test_launcher_ignores_ambient_python_home_without_changing_parent_environment(tmp_path):
    if sys.platform != 'win32':
        pytest.skip('PowerShell development entry is Windows only')
    script = tmp_path / 'environment-check.ps1'
    script.write_text("""param([string]$Entry)
$ErrorActionPreference = 'Stop'
$env:PYTHONHOME = 'deliberately-invalid-python-home'
$env:PYTHONPATH = 'unrelated-source-directory'
$env:PTB_EGG_PYTHON = 'unrelated-egg-runtime'
$value = & $Entry -CheckOnly -Module M14 | ConvertFrom-Json
if ($value.module -ne 'M14') { throw 'Wrong module' }
if ($env:PYTHONHOME -ne 'deliberately-invalid-python-home' -or $env:PYTHONPATH -ne 'unrelated-source-directory' -or $env:PTB_EGG_PYTHON -ne 'unrelated-egg-runtime') { throw 'Parent environment changed' }
""", encoding='utf8')
    result = subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(script),
                             '-Entry',str(ROOT/'scripts/Start-Research-Workbench.ps1')],
                            capture_output=True,encoding='utf8',timeout=30)
    assert result.returncode == 0, result.stderr
