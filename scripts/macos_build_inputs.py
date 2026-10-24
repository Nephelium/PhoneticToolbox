"""Freeze explicit Mac build inputs, applying the shared release data policy."""
import hashlib
import json
from pathlib import Path
import shutil
from release_content_policy import exclusion, external_frontend_data

SOURCE_TREES = ('backend/src', 'desktop/src', 'packages/phonetic_core/src')
DATA_TREES = ('frontend/dist', 'backend/migrations', 'contracts', 'third_party/licenses',
              'resources/m05', 'resources/vocal_tract/sources',
              'resources/vocal_tract/native/darwin-arm64')
DATA_FILES = ('release/version.json', 'third_party/source-registry.json',
              'resources/vocal_tract/VTL-LICENSE.txt', 'resources/vocal_tract/THIRD_PARTY_NOTICES.md',
              'resources/vocal_tract/sources.lock.json', 'resources/vocal_tract/bundle-manifest.json',
              'resources/research/reaper', 'resources/research/REAPER-LICENSE.txt',
              'resources/research/REAPER-1d6e9b95e6b08b500fccbc9a043989dbda747276.tar.gz',
              'resources/macos-validation.json', 'resources/macos-reaper-provenance.json',
              'resources/macos-media-provenance.json',
              'third_party/patches/ffmpeg-8.0-macos-avfoundation-namespace.patch')
SCRIPT_DATA = ('macos_science_entry.py', 'macos_validation_identity.py', 'verify_macos_runtime.py')


def skip_reason(relative):
    path = Path(relative)
    if '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo') or any(p.endswith('.egg-info') for p in path.parts):
        return 'generated Python metadata'
    if path.parts[:2] == ('frontend', 'dist') and len(path.parts) > 2 and external_frontend_data(path.parts[2]):
        return 'separately delivered local research data'
    return exclusion(path.as_posix())


def app_version(root):
    value = json.loads((Path(root) / 'release/version.json').read_text('utf8'))['app_version']
    if not isinstance(value, str) or not value or any(c not in '0123456789.' for c in value):
        raise ValueError('Invalid app_version')
    return value


def freeze_inputs(root, target):
    root, target = Path(root).resolve(), Path(target).resolve()
    if target.exists():
        raise RuntimeError('Build snapshot must be new')
    target.mkdir(parents=True)
    files, excluded = [], []

    def copy(path):
        relative = path.relative_to(root).as_posix()
        reason = skip_reason(relative)
        if reason:
            excluded.append({'path': relative, 'reason': reason})
            return
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise RuntimeError('Build input must be an ordinary checkout file: ' + relative)
        raw = path.read_bytes()
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        shutil.copymode(path, destination)
        digest = hashlib.sha256(raw).hexdigest()
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Build input changed during snapshot: ' + relative)
        files.append({'path': relative, 'bytes': len(raw), 'sha256': digest})

    for relative in SOURCE_TREES + DATA_TREES:
        base = root / relative
        if not base.is_dir() or base.is_symlink():
            raise RuntimeError('Required ordinary input directory missing: ' + relative)
        for path in sorted(base.rglob('*')):
            if path.is_file():
                copy(path)
    for relative in DATA_FILES:
        copy(root / relative)
    for path in sorted((root / 'resources/vocal_tract/native').glob('*.speaker')):
        copy(path)
    # Freeze build/entry imports too. Scripts are analysis input, not blanket app data.
    for path in sorted((root / 'scripts').rglob('*.py')):
        copy(path)
    manifest = {'schema': 'ptb-macos-build-inputs/1', 'app_version': app_version(target),
                'files': files, 'excluded': excluded,
                'policy': 'explicit resource roots; shared release_content_policy; no local-data, PDF scans or MFA runtime'}
    (target / 'macos-input-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    return manifest


def data_arguments(snapshot):
    snapshot = Path(snapshot)
    items = [(snapshot / p, p) for p in SOURCE_TREES + DATA_TREES]
    items += [(snapshot / p, str(Path(p).parent)) for p in DATA_FILES]
    items += [(p, 'resources/vocal_tract/native') for p in sorted((snapshot / 'resources/vocal_tract/native').glob('*.speaker'))]
    items += [(snapshot / 'scripts' / p, 'scripts') for p in SCRIPT_DATA]
    items += [(snapshot / 'macos-input-manifest.json', '.')]
    return [value for path, destination in items for value in ('--add-data', str(path) + ':' + destination)]


def verify_snapshot(snapshot, manifest):
    for row in manifest['files']:
        path = Path(snapshot) / row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('Frozen build input changed: ' + row['path'])
