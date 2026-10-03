"""The desktop uses QWidget + QWebEngineView, with no QQml engine.

Keep PyInstaller's binary dependency discovery, including QtQml/QtQuick DLLs
required by WebEngine. Do not collect the optional QML import/plugin tree:
doing so also pulls in QtQuick3D/Physics/Particles and their dependencies.
Only used by the explicitly selected local-preview lean profile.
"""
import json
import os
from pathlib import Path

from PyInstaller.utils.hooks.qt import add_qt6_dependencies, pyqt6_library_info

hiddenimports, binaries, datas = add_qt6_dependencies(__file__)
qml_binaries, qml_datas = pyqt6_library_info.collect_qtqml_files()
audit = Path(os.environ['PTB_BUNDLE_AUDIT_DIR']) / 'omitted-qml.json'
audit.write_text(json.dumps({
    'reason': 'No QQml engine or QML document in the QWidget/WebEngine desktop',
    'binaries': qml_binaries, 'datas': qml_datas,
}, indent=2), encoding='utf-8')
