"""Release WebEngine resources and Chinese/English UI locales only.

Retain the helper executable, ICU, V8, release resources (including DevTools),
and PyInstaller's standard binary dependency discovery. Unknown resources
are retained. This does not filter fonts, user text, or frontend content.
"""
import json
import os
from pathlib import Path

from PyInstaller.utils.hooks.qt import add_qt6_dependencies, pyqt6_library_info

if pyqt6_library_info.version is None or pyqt6_library_info.version < [6, 2, 2]:
    raise RuntimeError('The lean Qt profile requires an importable Qt >= 6.2.2')
if pyqt6_library_info.is_debug_build:
    raise RuntimeError('The lean Qt profile requires release Qt libraries')

hiddenimports, binaries, datas = add_qt6_dependencies(__file__)
web_binaries, web_datas = pyqt6_library_info.collect_qtwebengine_files()
binaries += web_binaries
omitted = []
for source, target in web_datas:
    root = Path(source)
    if not root.is_dir():
        datas.append((source, target))
        continue
    for file in sorted(root.rglob('*')):
        if not file.is_file():
            continue
        reason = None
        if root.name == 'resources' and (file.name.endswith('.debug.pak') or file.name.endswith('.debug.bin')):
            release = file.with_name(file.name.replace('.debug.', '.'))
            if not release.is_file():
                raise RuntimeError('Missing release counterpart for ' + str(file))
            reason = 'Debug-only resource; release counterpart retained'
        elif root.name == 'qtwebengine_locales' and file.suffix == '.pak' and file.stem not in {'en-US', 'en-GB', 'zh-CN', 'zh-TW'}:
            reason = 'WebEngine UI locale outside Chinese/English profile'
        if reason:
            omitted.append({'path': str(file), 'bytes': file.stat().st_size, 'reason': reason})
        else:
            datas.append((str(file), str(Path(target) / file.relative_to(root).parent)))
audit = Path(os.environ['PTB_BUNDLE_AUDIT_DIR']) / 'omitted-webengine.json'
audit.write_text(json.dumps(omitted, indent=2), encoding='utf-8')
