"""Build an authorized local trial without changing old artifacts or runtimes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='PhoneticToolbox-v3-LocalPreview-20260927')
    options = parser.parse_args()
    name = options.name
    if not name.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Use a plain artifact name')
    destination = ROOT / 'dist' / name
    work = ROOT / 'output' / ('build-' + name)
    if destination.exists() or work.exists():
        raise ValueError('Choose a new name; existing artifacts are retained')
    if sys.platform != 'win32' or not (ROOT / 'frontend/dist/index.html').is_file():
        raise ValueError('Windows and a built frontend are required')
    work.mkdir(parents=True)
    snapshot = work / 'snapshot'
    snapshot.mkdir()
    # All module Python sources are frozen to this build, never imported from V2.
    for relative in ('backend/src', 'desktop/src', 'packages/phonetic_core/src'):
        shutil.copytree(ROOT / relative, snapshot / relative,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    runtimes = {
        'PTB_EGG_PYTHON': str(ROOT / '.venv/m03-compatible/python.exe'),
        'PTB_M05_PYTHON': str(ROOT / '.venv/m05/Scripts/python.exe'),
    }
    mfa_root = ROOT / 'output/m11c-028b881d'
    if (mfa_root / 'registry.json').is_file():
        runtimes['PTB_M11_COMPONENT_ROOT'] = str(mfa_root)
    config = dict(kind='local-only-preview', source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), runtimes=runtimes,
        portable=False, modules='M01-M15; per-module validation remains separate')
    (snapshot / 'local-preview.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    args = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--onefile', '--windowed',
            '--name', name, '--distpath', str(destination), '--workpath', str(work / 'pyinstaller'),
            '--specpath', str(work)]
    for relative in ('scripts', 'backend/src', 'desktop/src', 'packages/phonetic_core/src'):
        args += ['--paths', str(ROOT / relative)]
    data = [(ROOT / r, r) for r in ('frontend/dist', 'resources/vocal_tract', 'resources/m05',
            'contracts', 'backend/migrations', 'docs/manual', 'third_party/licenses')]
    data += [(snapshot / r, r) for r in ('backend/src', 'desktop/src', 'packages/phonetic_core/src')]
    data += [(snapshot / 'local-preview.json', '.'),
             (ROOT / 'third_party/source-registry.json', 'third_party'),
             (ROOT / 'phonetic_toolbox/core/acoustic/reaper.exe', 'resources/research'),
             (ROOT / 'tests/fixtures/m14/public.xlsx', 'preview-fixtures'),
             (ROOT / 'tests/fixtures/m03/EGG-SYN-PCM16.npz', 'preview-fixtures')]
    for source, target in data:
        args += ['--add-data', str(source) + ';' + target]
    for package in ('ptb_worker', 'ptb_api', 'ptb_desktop', 'phonetic_core', 'uvicorn'):
        args += ['--collect-submodules', package]
    # Installed wheels can lag behind the checkout. Fixed child names and
    # namespace packages must be analyzed from this exact source snapshot.
    for relative in ('backend/src', 'desktop/src', 'packages/phonetic_core/src'):
        source_root = snapshot / relative
        for source in sorted(source_root.rglob('*.py')):
            parts = list(source.relative_to(source_root).with_suffix('').parts)
            if any(part.endswith('.egg-info') for part in parts):
                continue
            if parts[-1] == '__init__':
                parts.pop()
            if parts:
                args += ['--hidden-import', '.'.join(parts)]
    args += ['--collect-submodules', 'docx']
    for package in ('phonetic-core', 'ptb-api', 'ptb-desktop', 'numpy', 'scipy',
                    'praat-parselmouth', 'pandas', 'soundfile', 'python-docx', 'openpyxl', 'xlrd'):
        args += ['--copy-metadata', package]
    from m11_bundle import arguments
    args += arguments(ROOT)
    args += ['--collect-all', '_sounddevice_data', '--collect-data', 'docx',
             '--hidden-import', '_cffi_backend', '--hidden-import', 'xlrd', '--exclude-module', 'matplotlib',
             '--exclude-module', 'IPython', str(ROOT / 'scripts/v3_local_preview_entry.py')]
    env = os.environ.copy()
    windows = Path(env.get('SystemRoot', 'C:/Windows'))
    env['PATH'] = os.pathsep.join(map(str, (Path(sys.executable).parent, Path(sys.base_prefix),
                                          windows / 'System32', windows)))
    with (work / 'build.log').open('wb') as log:
        subprocess.run(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    exe = destination / (name + '.exe')
    metadata = config | dict(exe=exe.name, bytes=exe.stat().st_size,
                              sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (destination / 'build-info.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
