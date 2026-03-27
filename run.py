import sys
import os
import shutil
import tempfile
import time
from pathlib import Path

# Workaround for DLL load failed error with PyQt6 and OpenCV
try:
    import cv2
except ImportError:
    pass

# Ensure the project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from phonetic_toolbox.gui.main_window import MainWindow
from phonetic_toolbox.gui.dialogs.lip_feature_analysis_standalone import (
    main as lip_standalone_main,
)
from phonetic_toolbox.utils import get_resource_path


def _set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Nephelium.PhoneticToolbox.v2"
        )
    except Exception:
        return

def _looks_like_phonetic_mei_dir(path: Path) -> bool:
    try:
        names = {child.name.lower() for child in path.iterdir()}
    except Exception:
        return False
    return (
        "phonetic_toolbox" in names
        or "phonetic_export" in names
        or "phonetictoolbox.ico" in names
    )

def _cleanup_old_mei_dirs(max_age_seconds: int = 2 * 24 * 60 * 60) -> None:
    cutoff = time.time() - max_age_seconds
    temp_root = Path(tempfile.gettempdir())
    try:
        candidates = [p for p in temp_root.iterdir() if p.is_dir() and p.name.startswith("_MEI")]
    except Exception:
        return
    for item in candidates:
        try:
            if item.stat().st_mtime > cutoff:
                continue
        except Exception:
            continue
        if not _looks_like_phonetic_mei_dir(item):
            continue
        try:
            shutil.rmtree(item)
        except Exception:
            continue

def main():
    _cleanup_old_mei_dirs()
    _set_windows_app_id()
    if "--lip-standalone" in sys.argv:
        sys.argv = [sys.argv[0]] + [
            arg for arg in sys.argv[1:] if arg != "--lip-standalone"
        ]
        sys.exit(lip_standalone_main())
    app = QApplication(sys.argv)
    icon_path = get_resource_path("PhoneticToolbox.ico")
    app.setWindowIcon(QIcon(icon_path))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
