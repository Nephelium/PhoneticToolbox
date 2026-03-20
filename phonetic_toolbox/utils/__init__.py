import re
import sys
import os

def get_resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    
    Args:
        relative_path: Path relative to the project root (e.g. "PhoneticToolbox.ico" or "Phonetic_Export/index.html")
    """
    # 1. Try PyInstaller temp folder (bundled resources)
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
        full_path = os.path.join(base_path, relative_path)
        if os.path.exists(full_path):
            return full_path

    # 2. Try executable folder (external resources next to exe)
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
        full_path = os.path.join(base_path, relative_path)
        if os.path.exists(full_path):
            return full_path

    # 3. Try development path (relative to project root)
    # This file is in phonetic_toolbox/utils/__init__.py
    # Project root is 2 levels up: phonetic_toolbox/utils -> phonetic_toolbox -> root
    # Wait, __init__.py is in utils, so it is phonetic_toolbox/utils/__init__.py
    # dirname is phonetic_toolbox/utils
    # .. is phonetic_toolbox
    # .. is root
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    full_path = os.path.join(base_path, relative_path)
    
    return full_path

def parse_float_list(s: str) -> list[float]:
    """
    Parse a string of comma or space separated numbers into a list of floats.
    """
    try:
        return [float(x) for x in re.split(r"[,，\s]+", s.strip()) if x != ""]
    except:
        return []
