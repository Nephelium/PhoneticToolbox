import sys
import logging
import os
import importlib.util
from pathlib import Path
from typing import Optional

def _ensure_pkg_path() -> None:
    try:
        import PhoneticToolbox  # noqa: F401
        return
    except Exception:
        pass
    # 当在包目录内以脚本方式运行时，将上一级目录加入 sys.path
    pkg_dir = Path(__file__).resolve().parent
    parent = pkg_dir.parent
    pstr = str(parent)
    if pstr not in sys.path:
        sys.path.insert(0, pstr)

_ensure_pkg_path()

try:
    from .models.state import AppState
except ImportError:
    from PhoneticToolbox.models.state import AppState

def _ensure_qt_dll_path() -> None:
    try:
        spec = importlib.util.find_spec("PyQt6")
        if spec and spec.submodule_search_locations:
            pkg_dir = Path(list(spec.submodule_search_locations)[0])
            qtbin = pkg_dir / "Qt6" / "bin"
            if qtbin.exists():
                os.add_dll_directory(str(qtbin))
    except Exception:
        pass

_ensure_qt_dll_path()
from PyQt6 import QtWidgets, uic, QtGui
try:
    from .controllers import MainController
except ImportError:
    from PhoneticToolbox.controllers import MainController

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    direct = base / relative
    if direct.exists():
        return direct
    pkg_base = base / "PhoneticToolbox"
    cand = pkg_base / relative
    return cand if cand.exists() else Path(__file__).parent / relative


def load_ui(name: str) -> QtWidgets.QWidget:
    ui_file = resource_path(f"views/{name}.ui")
    return uic.loadUi(str(ui_file))


def main(argv: Optional[list[str]] = None) -> int:
    setup_logging()
    app = QtWidgets.QApplication(argv or sys.argv)
    
    icon_path = resource_path("PhoneticToolbox.ico")
    if icon_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))

    state = AppState()
    main_window = load_ui("ui_mainwindow")

    controller = MainController(main_window, state)
    controller.init()

    main_window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
