"""Compare two actual onefile archives; check the lean profile's payload.

Read-only with respect to both EXEs. Writes an inventory, omissions, large
components and remaining byte-identical DLLs for a reviewable size audit.
Run the frozen runtime verifier separately: archive presence is not execution.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader


def inventory(path):
    archive = CArchiveReader(str(path))
    rows = {name.replace('\\', '/'): {
        'compressed': info[1], 'uncompressed': info[2], 'type': info[4],
        'archive_name': name,
    } for name, info in archive.toc.items()}
    return archive, rows


def category(name):
    if name.startswith('PyQt6/Qt6/resources/') and '.debug.' in name:
        return 'Qt debug resources'
    if name.startswith('PyQt6/Qt6/qml/'):
        return 'QML imports and plugins'
    if name.startswith('PyQt6/Qt6/translations/qtwebengine_locales/'):
        return 'WebEngine UI locales'
    if name.startswith('Qt') and name.endswith('.dll'):
        return 'Qt libraries'
    if name.startswith('PyQt6/'):
        return 'Other Qt resources and bindings'
    if name.startswith('frontend/'):
        return 'Frontend assets'
    return 'Other application and Python dependencies'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    old, before = inventory(options.baseline)
    new, after = inventory(options.exe)
    config = json.loads(new.extract(after['local-preview.json']['archive_name']))
    assert config['bundle_profile'] == 'widgets-webengine-zh-en/1'
    assert not config['portable']
    assert not any(n.startswith('PyQt6/Qt6/qml/') for n in after)
    assert not any(n.startswith('PyQt6/Qt6/resources/') and '.debug.' in n for n in after)
    locales = {Path(n).stem for n in after if n.startswith('PyQt6/Qt6/translations/qtwebengine_locales/')}
    assert locales == {'en-US', 'en-GB', 'zh-CN', 'zh-TW'}, locales
    for name in ('Qt6WebEngineCore.dll', 'Qt6Quick.dll', 'Qt6Qml.dll',
                 'PyQt6/Qt6/bin/QtWebEngineProcess.exe', 'PyQt6/Qt6/bin/opengl32sw.dll',
                 'PyQt6/Qt6/resources/icudtl.dat', 'PyQt6/Qt6/resources/qtwebengine_resources.pak',
                 'PyQt6/Qt6/resources/qtwebengine_devtools_resources.pak',
                 'PyQt6/Qt6/resources/v8_context_snapshot.bin',
                 'third_party/licenses/ipa-plus/OFL-PTBIPAPlus.txt',
                 'third_party/licenses/ipa-plus/OFL-Noto.txt'):
        assert name in after, 'Required resource missing: ' + name
    # Prevent accidentally including the workspace, raw research materials or
    # editable environment configuration. Compiled modules are inside PYZ.
    prohibited = []
    for name in after:
        if any(p in {'.git', 'node_modules', '__pycache__', '.venv', 'output', 'dist'}
               for p in name.split('/') if not name.startswith('frontend/dist/')):
            prohibited.append(name)
        if Path(name).name in {'.env', '.env.local', '.env.production', 'id_rsa', 'id_ed25519'}:
            prohibited.append(name)
        if Path(name).suffix.lower() in {'.pdf', '.mp3', '.mp4', '.wav'}:
            if not (name.startswith('frontend/dist/assets/SYN-EGG-44100-') and name.endswith('.wav')):
                prohibited.append(name)
    assert not prohibited, prohibited
    groups = defaultdict(lambda: {'before': 0, 'after': 0})
    for state, entries in [('before', before), ('after', after)]:
        for name, info in entries.items():
            groups[category(name)][state] += info['compressed']
    for row in groups.values():
        row['saved'] = row['before'] - row['after']
    duplicates = defaultdict(list)
    for name, info in after.items():
        if name.lower().endswith('.dll') and info['uncompressed'] > 128 * 1024:
            digest = hashlib.sha256(new.extract(info['archive_name'])).hexdigest()
            duplicates[digest].append(name)
    pyz = new.open_embedded_archive('PYZ.pyz')
    before_bytes, after_bytes = options.baseline.stat().st_size, options.exe.stat().st_size
    assert after_bytes < before_bytes, (before_bytes, after_bytes)
    report = {
        'success': True, 'scope': 'Static archive audit; frozen execution evidence reported separately',
        'baseline': str(options.baseline.resolve()), 'exe': str(options.exe.resolve()),
        'before_bytes': before_bytes, 'after_bytes': after_bytes,
        'saved_bytes': before_bytes - after_bytes,
        'saved_percent': (before_bytes - after_bytes) / before_bytes * 100,
        'before_sha256': hashlib.sha256(options.baseline.read_bytes()).hexdigest(),
        'after_sha256': hashlib.sha256(options.exe.read_bytes()).hexdigest(),
        'before_entries': len(before), 'after_entries': len(after),
        'groups_compressed_bytes': dict(groups), 'retained_locales': sorted(locales),
        'prohibited_payload': prohibited,
        'removed': [{'name': n, **i} for n, i in before.items() if n not in after],
        'largest': [{'name': n, **i} for n, i in sorted(after.items(), key=lambda p: -p[1]['compressed'])[:30]],
        'identical_dll_groups_retained': [v for v in duplicates.values() if len(v) > 1],
        'verification_modules': [n for n in pyz.toc if n.startswith(('verify_', 'pytest', '_pytest', 'm16_test'))],
        'inventory': after,
    }
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k in {
        'success', 'before_bytes', 'after_bytes', 'saved_bytes', 'saved_percent',
        'before_entries', 'after_entries', 'groups_compressed_bytes', 'identical_dll_groups_retained',
    }}, ensure_ascii=False))


if __name__ == '__main__':
    main()
