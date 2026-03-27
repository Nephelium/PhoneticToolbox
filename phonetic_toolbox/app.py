import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from .gui.main_window import MainWindow
from .utils import get_resource_path


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

def main():
    _set_windows_app_id()
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(get_resource_path("PhoneticToolbox.ico")))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
