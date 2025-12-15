import sys
import logging
import os
import importlib.util
import multiprocessing
import atexit
import shutil
import tempfile
import re
from pathlib import Path
from typing import Optional

def _ensure_pkg_path() -> None:
    """Ensure all package paths are in sys.path for both dev and frozen environments."""
    # For frozen app (PyInstaller), add _MEIPASS to sys.path
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        meipass = sys._MEIPASS
        if meipass not in sys.path:
            sys.path.insert(0, meipass)
        # Also ensure the directory containing the exe is in path
        exe_dir = os.path.dirname(sys.executable)
        if exe_dir not in sys.path:
            sys.path.insert(0, exe_dir)
        return
    
    # For development mode
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
    # Also add current directory
    current = str(pkg_dir)
    if current not in sys.path:
        sys.path.insert(0, current)

_ensure_pkg_path()

# Store current _MEIPASS for cleanup on exit
_current_meipass: Optional[str] = None

def _cleanup_current_meipass() -> None:
    """Clean up the current session's _MEIPASS folder on exit."""
    global _current_meipass
    if _current_meipass and os.path.exists(_current_meipass):
        try:
            shutil.rmtree(_current_meipass, ignore_errors=True)
            logging.info(f"Cleaned up current _MEIPASS: {_current_meipass}")
        except Exception as e:
            logging.warning(f"Failed to clean up _MEIPASS: {e}")

def _cleanup_old_mei_folders() -> None:
    """Clean up old _MEI* folders from previous sessions that weren't properly cleaned."""
    if not getattr(sys, 'frozen', False):
        return
    
    temp_dir = tempfile.gettempdir()
    mei_pattern = re.compile(r'^_MEI\d+$')
    current_mei = getattr(sys, '_MEIPASS', None)
    
    try:
        for item in os.listdir(temp_dir):
            if mei_pattern.match(item):
                mei_path = os.path.join(temp_dir, item)
                # Skip the current session's folder
                if current_mei and os.path.normpath(mei_path) == os.path.normpath(current_mei):
                    continue
                # Check if the folder is old (not in use by another process)
                try:
                    # Try to remove - if it's in use, it will fail
                    shutil.rmtree(mei_path, ignore_errors=False)
                    logging.info(f"Cleaned up old _MEI folder: {mei_path}")
                except PermissionError:
                    # Folder is likely in use by another instance
                    pass
                except Exception as e:
                    logging.debug(f"Could not remove {mei_path}: {e}")
    except Exception as e:
        logging.warning(f"Error during _MEI cleanup: {e}")

def _setup_mei_cleanup() -> None:
    """Set up cleanup for PyInstaller temporary folders."""
    global _current_meipass
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        _current_meipass = sys._MEIPASS
        # Register cleanup on normal exit
        atexit.register(_cleanup_current_meipass)
        # Clean up old folders from previous sessions
        _cleanup_old_mei_folders()

_setup_mei_cleanup()

# Try absolute imports first (for frozen app/script usage), then relative (for package usage)
try:
    from models.state import AppState
except ImportError:
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

def _setup_mfa_path() -> None:
    """Ensure MFA binaries are in PATH, especially for bundled app."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        mfa_bin = os.path.join(sys._MEIPASS, 'mfa_bin')
        if os.path.exists(mfa_bin):
            # Prepend to PATH to ensure these versions are used
            os.environ["PATH"] = mfa_bin + os.pathsep + os.environ["PATH"]
            logging.info(f"Added bundled mfa_bin to PATH: {mfa_bin}")
            
            # Critical for Windows Python 3.8+: Add to DLL search path
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(mfa_bin)
                    logging.info(f"Added mfa_bin to DLL directory: {mfa_bin}")
                except Exception as e:
                    logging.warning(f"Could not add dll directory: {e}")

            # FORCE PRELOAD DLLs to avoid WinError 1114
            # This is a brute-force fix for dependency issues in frozen apps
            import glob
            import ctypes
            dlls = glob.glob(os.path.join(mfa_bin, "*.dll"))
            logging.info(f"Found {len(dlls)} DLLs in mfa_bin. Attempting to preload...")
            
            # Try loading twice to handle dependencies order
            for i in range(2):
                loaded = 0
                for dll in dlls:
                    try:
                        ctypes.CDLL(dll)
                        loaded += 1
                    except Exception:
                        pass
                logging.info(f"Preload pass {i+1}: Loaded {loaded}/{len(dlls)} DLLs")

    else:
        # In dev mode, we might want to ensure they are found if not in PATH
        pass

if not getattr(sys, 'frozen', False):
    _ensure_qt_dll_path()
_setup_mfa_path() # Run this early

from PyQt6 import QtWidgets, uic, QtGui

# Explicitly try to import MainController
# In frozen app, 'controllers' should be available at root.
# In dev mode (package), it might be relative.
try:
    import controllers
    MainController = controllers.MainController
except ImportError:
    try:
        from . import controllers
        MainController = controllers.MainController
    except ImportError:
        try:
            import PhoneticToolbox.controllers as controllers
            MainController = controllers.MainController
        except ImportError as e:
            # If all fail, we must log it because the app will crash
            logging.critical(f"Failed to import controllers: {e}")
            raise

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


def main() -> None:
    try:
        setup_logging()
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle("Fusion")
        
        # Load Main Window
        ui_path = resource_path("views/ui_mainwindow.ui")
        if not ui_path.exists():
            QtWidgets.QMessageBox.critical(None, "Error", f"UI file not found: {ui_path}")
            return

        window = uic.loadUi(str(ui_path))
        
        # Set window icon (for taskbar)
        icon_path = resource_path("PhoneticToolbox.ico")
        if icon_path.exists():
            app.setWindowIcon(QtGui.QIcon(str(icon_path)))
            window.setWindowIcon(QtGui.QIcon(str(icon_path)))
        
        # Initialize State & Controller
        state = AppState()
        controller = MainController(window, state)
        controller.init()

        window.show()
        sys.exit(app.exec())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
        multiprocessing.freeze_support()
        main()
