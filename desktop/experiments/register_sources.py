"""Register the exact P01 dependency metadata without changing legacy source claims."""
import importlib.metadata as metadata
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def identifier(value):
    return re.sub(r'[^A-Z0-9]+', '-', value.upper()).strip('-')


if __name__ == '__main__':
    registry_path = ROOT / 'third_party/source-registry.json'
    registry = json.loads(registry_path.read_text('utf-8'))
    new = []
    inventory = []
    license_dir = ROOT / 'third_party/licenses/p01'
    license_dir.mkdir(parents=True, exist_ok=True)
    for distribution in sorted(metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
        name, version = distribution.metadata['Name'], distribution.version
        text = distribution.metadata.get('License-Expression') or distribution.metadata.get('License') or 'unknown'
        author = distribution.metadata.get('Author') or distribution.metadata.get('Author-email') or 'unknown'
        files = [str(f) for f in distribution.files or [] if 'license' in str(f).lower() or 'copying' in str(f).lower()]
        record = {'name': name, 'version': version, 'license_metadata': text, 'author': author,
                  'license_files_in_environment': files, 'requires_dist': distribution.requires or []}
        inventory.append(record)
        summary = text if len(text) < 220 else 'See exact installed license metadata in third_party/p01-dependency-inventory.json'
        runtime_names = {'pyqt6', 'pyqt6-qt6', 'pyqt6_sip', 'pyqt6-webengine', 'pyqt6-webengine-qt6'}
        build_names = {'pyinstaller', 'altgraph', 'pefile', 'pyinstaller-hooks-contrib', 'pywin32-ctypes', 'setuptools', 'packaging'}
        role = 'runtime-dependency' if name.lower() in runtime_names else 'build-dependency' if name.lower() in build_names else 'test-dependency'
        new.append({'id': 'P01-PY-' + identifier(name), 'title': name, 'kind': role,
            'modules': ['P01-probe'], 'authors': author, 'actual_included_version': version,
            'observed_upstream_commit': None, 'license': summary,
            'urls': {'release_metadata': f'https://pypi.org/pypi/{name}/{version}/json'},
            'verification_date': '2026-09-09', 'distribution_status': 'prototype-only-full-native-audit-pending',
            'evidence': 'Exact installed metadata; inventory records package-level license and requirements.',
            'notes': 'Installed in P01 standalone environment; build/test packages are not all bundled in the EXE.'})
        if name.lower() in ['pyqt6', 'pyqt6-webengine', 'pyinstaller']:
            for file in files:
                path = Path(distribution.locate_file(file))
                if path.is_file() and path.stat().st_size < 200000:
                    (license_dir / f'{identifier(name)}-{Path(file).name}.txt').write_bytes(path.read_bytes())
    frontend = ROOT / 'frontend/experiments/audio-viewport'
    lock = json.loads((frontend / 'package-lock.json').read_text('utf-8'))
    npm_records = []
    for location, entry in lock['packages'].items():
        if not location:
            continue
        path = frontend / location / 'package.json'
        package = json.loads(path.read_text('utf-8')) if path.is_file() else {}
        name = package.get('name') or location.split('node_modules/')[-1]
        author = package.get('author', 'unknown')
        if isinstance(author, dict):
            author = author.get('name', 'unknown')
        license_name = entry.get('license', package.get('license', 'unknown'))
        npm_records.append({'name': name, 'version': entry['version'], 'license': license_name,
                            'resolved': entry.get('resolved'), 'integrity': entry.get('integrity'),
                            'installed_on_this_platform': path.is_file()})
        new.append({'id': 'P01-NPM-' + identifier(name), 'title': name, 'kind': 'probe-ui-build-dependency',
            'modules': ['P01-probe'], 'authors': author, 'actual_included_version': entry['version'],
            'observed_upstream_commit': None, 'license': license_name,
            'urls': {'registry': f'https://www.npmjs.com/package/{name}/v/{entry["version"]}'},
            'verification_date': '2026-09-09', 'distribution_status': 'prototype-only',
            'evidence': 'package-lock.json version/integrity; optional platform packages may not be installed.',
            'notes': 'P01 front-end build or runtime dependency; no upstream code modifications.'})
    new += [
        {'id': 'P01-CPYTHON', 'title': 'CPython via python-build-standalone', 'kind': 'probe-runtime-dependency',
         'modules': ['P01-probe'], 'authors': 'Python Software Foundation and python-build-standalone contributors',
         'actual_included_version': 'CPython 3.11.14, uv-managed Windows x86_64 standalone distribution',
         'observed_upstream_commit': None, 'license': 'PSF-2.0 and bundled component terms; runtime distribution audit pending',
         'urls': {'runtime_builds': 'https://github.com/astral-sh/python-build-standalone', 'python': 'https://www.python.org/'},
         'verification_date': '2026-09-09', 'distribution_status': 'prototype-only',
         'evidence': 'Installed under project .venv/runtimes with --no-bin --no-registry.',
         'notes': 'Does not reuse or change the conda phonetic_311 interpreter for the accepted probe.'},
        {'id': 'P01-PYSIDE6', 'title': 'PySide6 candidate host', 'kind': 'evaluated-alternative-dependency',
         'modules': ['P01-probe'], 'authors': 'The Qt Company Ltd. and Qt contributors',
         'actual_included_version': 'PySide6 / Essentials / Addons / Shiboken6 6.11.2 in separate trial environment',
         'observed_upstream_commit': None, 'license': 'LGPL/GPL/commercial and component-specific terms; see Qt license inventory',
         'urls': {'licenses': 'https://doc.qt.io/qtforpython-6/licenses.html', 'package': 'https://pypi.org/project/PySide6/6.11.2/'},
         'verification_date': '2026-09-09', 'distribution_status': 'not-in-pyqt6-prototype',
         'evidence': 'Separate environment; startup/font/device enumeration observed. See P01 report for interaction test result.',
         'notes': 'Candidate only; not selected or represented as a validated packaged alternative.'},
    ]
    existing = {s['id']: s for s in registry['sources']}
    existing.update({s['id']: s for s in new})
    registry['sources'] = list(existing.values())
    registry['total_records'] = len(registry['sources'])
    registry['checked_on'] = '2026-09-09'
    font = next(s for s in registry['sources'] if s['id'] == 'ASSET-DOULOS')
    font.update(actual_included_version='Version 7.000; read from the included font name table',
                license='SIL Open Font License 1.1; embedded license extracted without modification',
                evidence='Font name table + SHA-256 in frontend/experiments/audio-viewport/public/asset-manifest.json',
                notes='P01 uses the unchanged inherited font; exact copyright and OFL are in public/Doulos-OFL.txt.')
    font['modules'] = list(dict.fromkeys(font['modules'] + ['P01-probe']))
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'third_party/p01-dependency-inventory.json').write_text(json.dumps(
        {'python': inventory, 'npm': npm_records, 'note': 'Package metadata is not a complete audit of Chromium/Qt native third-party code.'},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'registry_records': len(registry['sources']), 'python_packages': len(inventory), 'npm_lock_records': len(npm_records)}))
