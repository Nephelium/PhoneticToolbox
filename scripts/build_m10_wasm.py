"""Build the pinned VTL sources as Pyodide-compatible Wasm side modules.

Uses an existing task-local SDK. Never replaces the desktop's native resources.
Only symbol visibility/ABI adapters are applied to the original API sources;
geometry uses the same preparation script and patches as native builds.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = 'aa696bf4cf5c44cbd433304943aebb241e5ea3064b949daf5ea09f96567cff9b'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output, sdk):
    archive = ROOT / 'resources/vocal_tract/sources/VTL2.4-API-source.zip'
    if digest(archive) != ARCHIVE_SHA:
        raise ValueError('VTL source identity mismatch')
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/build_m10_geometry.py'),
                    '--output-dir', str(output), '--no-install', '--prepare-only'], check=True)
    original = output / 'original'
    original.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            path = Path(name)
            if path.parent.as_posix() == 'Developer/Sources' and path.suffix in ('.cpp', '.h'):
                (original / path.name).write_bytes(source.read(name))
    runpy.run_path(str(ROOT / 'resources/vocal_tract/sources/m10_source_perturbation.py'))['apply'](original)
    header = original / 'VocalTractLabApi.h'
    text = header.read_text('utf-8')
    text, exports = re.subn(r'(?m)^C_EXPORT ', 'C_EXPORT __attribute__((visibility("default"))) ', text)
    if exports < 12:
        raise ValueError('Unexpected VTL export header')
    header.write_text(text, encoding='utf-8')
    native = output / 'native'
    binary = native / 'emscripten-wasm32'
    binary.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'EM_CONFIG': str(sdk / '.emscripten'), 'EMSDK': str(sdk)}
    compiler = [sys.executable, str(sdk / 'upstream/emscripten/em++.py')]
    commands = []

    def compile_library(name, sources, include, public, extra=()):
        constants = output / (name + '-constants.cpp')
        constants.write_text('#include "Tube.h"\nconstexpr int Tube::SINUS_COUPLING_SECTION[Tube::NUM_SINUS_SECTIONS];\n', encoding='utf-8')
        command = [*compiler, '-O2', '-std=c++14', '-fwasm-exceptions',
                   '-ffp-contract=off', '-fvisibility=hidden', '-fvisibility-inlines-hidden',
                   '-sSIDE_MODULE=2', '-sEXPORTED_FUNCTIONS=' + json.dumps(['_' + symbol for symbol in public]), '-I' + str(include), *extra,
                   *map(str, [*sources, constants]), '-o', str(binary / name)]
        commands.append(command)
        print('building ' + name, flush=True)
        with (output / (name + '.log')).open('w', encoding='utf-8') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    api_exports = sorted(set(re.findall(r'\b(vtl[A-Za-z0-9_]+)\s*\(', text)))
    compile_library('libVocalTractLabApi.so',
                    [p for p in sorted(original.glob('*.cpp')) if p.name != 'SoundLib.cpp'], original, api_exports)
    shutil.copyfile(binary / 'libVocalTractLabApi.so', binary / 'libVocalTractLabAnalysis.so')
    names = 'Geometry Surface VocalTract Tube Splines XmlNode XmlHelper Dsp Signal IirFilter Constants TlModel Matrix2x2'.split()
    bridge = ROOT / 'resources/vocal_tract/sources/geometry_bridge.cpp'
    geometry_exports = sorted(set(re.findall(r'\b(p[0-9]+_[A-Za-z0-9_]+)\s*\(', bridge.read_text('utf-8'))))
    compile_library('libgeometry_p2.so',
                    [ROOT / 'resources/vocal_tract/sources/geometry_bridge.cpp',
                     *[output / 'source' / (name + '.cpp') for name in names]], output / 'source', geometry_exports,
                    ['-D__declspec(x)=__attribute__((visibility("default")))'])
    for name in ('JD2.speaker', 'M01.speaker', 'W02.speaker'):
        shutil.copyfile(ROOT / 'resources/vocal_tract/native' / name, native / name)
    shutil.copyfile(ROOT / 'resources/vocal_tract/VTL-LICENSE.txt', native / 'VTL-LICENSE.txt')
    prepare_runtime(output)
    sources = [ROOT / 'scripts/build_m10_geometry.py', ROOT / 'resources/vocal_tract/sources/m10_source_perturbation.py', ROOT / 'resources/vocal_tract/sources/geometry_bridge.cpp',
               *(ROOT / 'resources/vocal_tract/sources').glob('m10_r*_patch.py')]
    report = {'source_archive_sha256': ARCHIVE_SHA, 'geometry_method': 'm10/r21',
              'abi': 'pyodide_2025_0', 'compiler': subprocess.check_output([*compiler, '--version'], env=env, text=True).splitlines()[0],
              'commands': commands, 'source_files': {str(p.relative_to(ROOT)): digest(p) for p in sources},
              'files': {p.relative_to(native).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size}
                        for p in native.rglob('*') if p.is_file()},
              'scope': 'Wasm build only; runtime isolation, scientific parity and performance require separate verification'}
    (output / 'build-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: item['bytes'] for name, item in report['files'].items()}))


def prepare_runtime(output):
    """A small web-only bundle; Pyodide/NumPy/SciPy reuse the existing runtime."""
    target = output / 'vocal-wasm'
    target.mkdir(exist_ok=True)
    records = {}
    with zipfile.ZipFile(target / 'vocal-runtime.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        core = ROOT / 'packages/phonetic_core/src'
        sources = [core / 'phonetic_core/__init__.py', *(core / 'phonetic_core/vocal_tract').glob('*.py')]
        worker = ROOT / 'backend/src'
        sources += [worker / 'ptb_worker' / name for name in ('__init__.py', 'vocal_web.py', 'browser_vocal.py')]
        for source in sources:
            anchor = core if source.is_relative_to(core) else worker
            name = source.relative_to(anchor).as_posix()
            bundle.writestr(name, source.read_bytes())
            records[name] = digest(source)
        version = tomllib.loads((ROOT / 'packages/phonetic_core/pyproject.toml').read_text('utf-8'))['project']['version']
        bundle.writestr(f'phonetic_core-{version}.dist-info/METADATA', f'Metadata-Version: 2.1\nName: phonetic-core\nVersion: {version}\n')
        for source in (output / 'native').rglob('*'):
            if source.is_file():
                name = 'vocal-native/' + source.relative_to(output / 'native').as_posix()
                bundle.writestr(name, source.read_bytes())
                records[name] = digest(source)
    # Serve the exact corresponding sources beside the Wasm, independently of
    # GitHub publication. It is downloaded only when requested, not at boot.
    source_files=[ROOT/'scripts/build_m10_wasm.py',ROOT/'scripts/build_m10_geometry.py',
        ROOT/'resources/vocal_tract/VTL-LICENSE.txt',ROOT/'frontend/package.json']
    source_files += [p for p in (ROOT/'resources/vocal_tract/sources').iterdir() if p.is_file()]
    source_files += [core/'phonetic_core/__init__.py',*(core/'phonetic_core/vocal_tract').glob('*.py')]
    source_files += [worker/'ptb_worker'/name for name in ('__init__.py','browser_vocal.py','vocal_web.py')]
    source_files += [ROOT/'frontend/src/platform'/name for name in ('browser-vocal.ts','vocal.worker.ts','vocal.ts')]
    with zipfile.ZipFile(target/'corresponding-source.zip','w',zipfile.ZIP_DEFLATED) as source_bundle:
        for source in source_files:source_bundle.writestr(source.relative_to(ROOT).as_posix(),source.read_bytes())
        source_bundle.writestr('BUILD.txt','Use Python 3.11+ and the existing Emscripten 4.0.9 SDK.\nRun python scripts/build_m10_wasm.py --output-dir build --sdk /path/to/emsdk-4.0.9.\nOriginal VTL source archive and reviewed patches are included. Runtime uses Pyodide 0.29.3.\nSource extension is opt-in; see m10_source_perturbation.py. GPL-3.0-or-later.\n')
    manifest = {'schema': 'ptb-vocal-wasm/1', 'pyodide': '0.29.3', 'emscripten': '4.0.9',
                'geometry_method': 'm10/r21', 'source_archive_sha256': ARCHIVE_SHA,
                'archive': {'name': 'vocal-runtime.zip', 'sha256': digest(target / 'vocal-runtime.zip'),
                            'bytes': (target / 'vocal-runtime.zip').stat().st_size}, 'sources': records,
                'corresponding_source':{'name':'corresponding-source.zip','sha256':digest(target/'corresponding-source.zip'),'bytes':(target/'corresponding-source.zip').stat().st_size}}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (target / 'THIRD_PARTY_NOTICES.txt').write_text(
        'VocalTractLab API 2.4, Peter Birkholz and contributors: GPL-3.0-or-later.\n'
        'Original C++ archive SHA256: ' + ARCHIVE_SHA + '\n'
        'PhoneticToolbox m10/r21 geometry bridge and patches: GPL-3.0-or-later.\n'
        'Original speakers and geometry are retained. Optional seeded jitter/shimmer use the reviewed m10_source_perturbation.py extension; unconfigured source follows upstream behavior.\n'
        'Corresponding source: resources/vocal_tract/sources and scripts/build_m10_wasm.py in the PhoneticToolbox source tree.\n'
        'VTL license is included as vocal-native/VTL-LICENSE.txt in vocal-runtime.zip.\n'
        'Pyodide/CPython/NumPy/SciPy use the separately pinned browser-science runtime and its original notices.\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--refresh-runtime-only', action='store_true')
    parser.add_argument('--write-lock', type=Path)
    args = parser.parse_args()
    if args.refresh_runtime_only:
        prepare_runtime(args.output_dir.resolve())
    else:
        build(args.output_dir.resolve(), args.sdk.resolve())
    if args.write_lock:
        args.write_lock.write_bytes((args.output_dir / 'vocal-wasm/manifest.json').read_bytes())
