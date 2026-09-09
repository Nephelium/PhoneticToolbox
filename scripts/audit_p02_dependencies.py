"""Read installed metadata and v2 declarations, preserving their separate facts."""
import importlib.metadata as metadata
import json
import re
import subprocess
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
FIRST_PARTY = {'phonetic-core', 'ptb-api', 'ptb-desktop'}


def identifier(name):
    return re.sub('[^A-Z0-9]+', '-', name.upper()).strip('-')


def main():
    registry_path = ROOT / 'third_party/source-registry.json'
    registry = json.loads(registry_path.read_text('utf-8'))
    sources = {s['id']: s for s in registry['sources']}
    installed, npm = [], []
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
        name, version = dist.metadata['Name'], dist.version
        normalized = canonicalize_name(name)
        if normalized in FIRST_PARTY:
            continue
        license_text = dist.metadata.get('License-Expression') or dist.metadata.get('License') or 'unknown'
        authors = dist.metadata.get('Author') or dist.metadata.get('Author-email') or 'unknown'
        role = 'runtime-dependency' if normalized in {
            'fastapi', 'pydantic', 'pydantic-core', 'uvicorn', 'starlette', 'anyio', 'idna',
            'annotated-doc', 'annotated-types', 'click', 'h11', 'colorama', 'typing-extensions',
            'typing-inspection', 'pyqt6', 'pyqt6-qt6', 'pyqt6-sip', 'pyqt6-webengine', 'pyqt6-webengine-qt6'
        } else 'development-dependency'
        installed.append({'name': name, 'version': version, 'kind': role, 'authors': authors,
            'license_metadata': license_text, 'requires_dist': dist.requires or [],
            'license_files': [str(p) for p in dist.files or [] if 'license' in str(p).lower() or 'copying' in str(p).lower()]})
        sid = 'P02-PY-' + identifier(name)
        sources[sid] = {'id': sid, 'title': name, 'kind': role, 'modules': ['P02'], 'authors': authors,
            'actual_included_version': version, 'observed_upstream_commit': None,
            'license': license_text if len(license_text) < 200 else 'See p02-dependency-inventory.json for full metadata',
            'urls': {'release': f'https://pypi.org/project/{name}/{version}/'},
            'verification_date': '2026-09-09', 'distribution_status': 'development-only-native-release-audit-pending',
            'evidence': 'Installed metadata and requirements-v3-dev.lock; wheel hash verification at clean installation.',
            'notes': 'Unmodified upstream dependency; no v2 algorithm migration. Build/test dependencies are not necessarily runtime contents.'}
    frontend = ROOT / 'frontend'
    lock = json.loads((frontend / 'package-lock.json').read_text('utf-8'))
    for location, record in lock['packages'].items():
        if not location:
            continue
        pkg_path = frontend / location / 'package.json'
        pkg = json.loads(pkg_path.read_text('utf-8')) if pkg_path.is_file() else {}
        name = pkg.get('name') or location.split('node_modules/')[-1]
        authors = pkg.get('author', 'unknown')
        if isinstance(authors, dict):
            authors = authors.get('name', 'unknown')
        item = {'name': name, 'version': record['version'], 'location': location,
                'license': record.get('license', pkg.get('license', 'unknown')),
                'integrity': record.get('integrity'), 'resolved': record.get('resolved'),
                'installed': pkg_path.is_file(), 'development': record.get('dev', False), 'authors': authors}
        npm.append(item)
        # The same package can have several locked versions; preserve each one.
        sid = 'P02-NPM-' + identifier(name + '-' + record['version'])
        sources[sid] = {'id': sid, 'title': name,
            'kind': 'frontend-build-dependency' if item['development'] else 'frontend-runtime-dependency',
            'modules': ['P02'], 'authors': authors, 'actual_included_version': record['version'],
            'observed_upstream_commit': None, 'license': item['license'],
            'urls': {'release': f'https://www.npmjs.com/package/{name}/v/{record["version"]}'},
            'verification_date': '2026-09-09', 'distribution_status': 'development-locked',
            'evidence': 'frontend/package-lock.json with integrity; inventory distinguishes installed optional packages.',
            'notes': 'No upstream source edits. js-yaml 4.3.2 override fixes GHSA-2883-xcg3-v3hh.' if name == 'js-yaml' else 'Unmodified dependency.'}
    before = json.loads((ROOT / 'output/validation/p02/context-before.json').read_text('utf-8'))
    old_python = Path(before['v2_environment']) / 'python.exe'
    old = json.loads(subprocess.check_output([str(old_python), '-X', 'utf8', '-c',
        'import importlib.metadata as m,json; print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()}))']))
    old = {canonicalize_name(k): v for k, v in old.items()}
    declarations = tomllib.loads((ROOT / 'pyproject.toml').read_text('utf-8'))['project']['dependencies']
    comparison = []
    for declared in declarations:
        req = Requirement(declared)
        actual = old.get(canonicalize_name(req.name))
        comparison.append({'declared': declared, 'installed': actual,
            'matches': actual is not None and req.specifier.contains(actual),
            'p02_action': 'Preserved in v2; scientific parity and migration remain P03/P08' if not req.name.startswith('PyQt') else 'Separate P02 host version; see ADR-013'})
    # Reuse the same independently installed interpreter, without implying the P01 env is the new env.
    sid = 'P02-CPYTHON'
    sources[sid] = dict(sources['P01-CPYTHON'], id=sid, modules=['P02'],
        kind='development-runtime', notes='Same project-local standalone interpreter build; new .venv/v3-dev environment.')
    registry['sources'] = list(sources.values())
    registry['total_records'] = len(sources)
    inventory = {'schema_version': 1, 'checked_on': '2026-09-09',
        'platform': 'Windows x86_64 CPython 3.11.14', 'python': installed, 'npm': npm,
        'v2_declaration_comparison': comparison,
        'limitations': ['Metadata is not full native license audit', 'P02 core has no scientific third-party dependencies',
                       'macOS/Linux lock and native runtime acceptance pending corresponding platform tasks']}
    for path, data in [(registry_path, registry), (ROOT / 'third_party/p02-dependency-inventory.json', inventory)]:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'python_packages': len(installed), 'npm_lock_entries': len(npm),
                      'v2_mismatches': [r for r in comparison if not r['matches']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
