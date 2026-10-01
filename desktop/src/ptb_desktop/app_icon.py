"""Use the same bundled K2 asset as the workbench home and sidebar."""
import sys
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication


def configure_app_icon(dist: Path) -> QIcon:
    # Vite emits this imported asset with a content hash in both source and EXE builds.
    candidates = sorted((dist / 'assets').glob('k2-*.png'))
    icon = QIcon(str(candidates[0])) if candidates else QIcon()
    app = QApplication.instance()
    if app is not None and not icon.isNull():
        app.setWindowIcon(icon)
    if sys.platform == 'win32':
        import ctypes
        # Set before showing the first window; this is process-local, not a registry change.
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('PhoneticToolbox.Desktop.3')
    return icon
