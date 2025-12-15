import sys
import os

# --- CRITICAL: SETUP PATH FOR DLLs BEFORE ANYTHING ELSE ---
# This must run before any other imports, especially before multiprocessing or MFA imports.
# It ensures that child processes (spawned by MFA) can find the necessary DLLs (Kaldi, OpenBLAS, etc.)
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
    
    # Locate mfa_bin (contains Kaldi DLLs)
    # PyInstaller 6+ in onedir mode might put things in _internal, or root.
    # We check multiple locations to be safe.
    candidates = [
        os.path.join(base_dir, 'mfa_bin'),
        os.path.join(base_dir, '_internal', 'mfa_bin'),
    ]
    # If running in onefile mode (less likely here but possible), checks sys._MEIPASS
    if hasattr(sys, '_MEIPASS'):
        candidates.append(os.path.join(sys._MEIPASS, 'mfa_bin'))

    mfa_bin_path = None
    for c in candidates:
        if c and os.path.isdir(c):
            mfa_bin_path = c
            break
            
    if mfa_bin_path:
        # 1. Add to PATH (for subprocesses and general DLL search)
        os.environ["PATH"] = mfa_bin_path + os.pathsep + os.environ["PATH"]
        
        # 2. Add DLL directory (specifically for Python 3.8+ DLL loading)
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(mfa_bin_path)
            except Exception:
                pass

    # --- FIX LLVMLITE DLL LOADING (CRITICAL) ---
    # llvmlite.dll is often placed in the root of the frozen app or _internal.
    # We must add these to DLL search path too, otherwise Numba/llvmlite will fail.
    if hasattr(os, 'add_dll_directory'):
        try:
            # Recursively add all directories in _internal that contain DLLs
            # This is brute-force but ensures we find llvmlite.dll, libsndfile.dll, etc.
            
            # 1. Add base dir
            os.add_dll_directory(base_dir)
            
            # 2. Add _internal root
            internal_dir = os.path.join(base_dir, '_internal')
            if os.path.isdir(internal_dir):
                os.add_dll_directory(internal_dir)
                
                # 3. Recursive search
                for root, dirs, files in os.walk(internal_dir):
                    for file in files:
                        if file.lower().endswith(".dll"):
                            try:
                                os.add_dll_directory(root)
                                # Once we add a dir, we can break loop for this dir
                                break 
                            except Exception:
                                pass
        except Exception:
            pass
    # Also add to PATH just in case
    os.environ["PATH"] = base_dir + os.pathsep + os.environ["PATH"]
    try:
        internal_dir = os.path.join(base_dir, '_internal')
        if os.path.isdir(internal_dir):
            dll_dirs = set()
            for root, dirs, files in os.walk(internal_dir):
                if any(f.lower().endswith('.dll') for f in files):
                    dll_dirs.add(root)
            env_path_parts = [*dll_dirs, os.environ["PATH"]]
            os.environ["PATH"] = os.pathsep.join(env_path_parts)
    except Exception:
        pass

    os.environ["BLAS_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMBA_DISABLE_JIT"] = "1"

    import importlib.abc

    class _BlockNumbaFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "numba" or fullname.startswith("numba."):
                raise ImportError("numba is disabled in this frozen application")
            return None

    sys.meta_path.insert(0, _BlockNumbaFinder())

# --- END CRITICAL SETUP ---

import logging
from PyQt6 import QtWidgets, QtCore
import multiprocessing

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle("Fusion")
        
        # Apply dark theme to match the main app
        GLOBAL_DARK_STYLESHEET = """
        QWidget {
            background-color: #121212;
            color: #ffffff;
        }
        QDialog {
            background-color: #121212;
        }
        QLineEdit, QTextEdit, QPlainTextEdit, QListWidget, QTreeWidget, QTableWidget {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #333333;
            selection-background-color: #3d3d3d;
        }
        QPushButton {
            background-color: #333333;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #444444;
        }
        """
        app.setStyleSheet(GLOBAL_DARK_STYLESHEET)
        
        # Import the dialog
        try:
            from views.auto_annotation import AutoAnnotationDialog
        except ImportError as e:
            QtWidgets.QMessageBox.critical(None, "Import Error", f"Failed to import AutoAnnotationDialog: {e}")
            return

        window = AutoAnnotationDialog()
        window.setWindowTitle("自动标注 (Auto Annotation Standalone)")
        window.show()
        
        sys.exit(app.exec())
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
