"""Read PE imports and report actual Windows dependency resolution."""
import ctypes
import json
import os
from pathlib import Path
import pefile

root = Path(__file__).resolve().parents[2]
qtbin = root / '.venv/p01-pyqt6/Lib/site-packages/PyQt6/Qt6/bin'
handle = os.add_dll_directory(str(qtbin))
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
kernel.GetProcAddress.restype = ctypes.c_void_p
kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
items = []
for entry in pefile.PE(str(qtbin / 'Qt6Core.dll')).DIRECTORY_ENTRY_IMPORT:
    name = entry.dll.decode()
    try:
        library = ctypes.WinDLL(name)
        path = ctypes.create_unicode_buffer(32768)
        kernel.GetModuleFileNameW(library._handle, path, len(path))
        missing = [imp.name.decode() for imp in entry.imports
                   if imp.name and not kernel.GetProcAddress(library._handle, imp.name)]
        if missing or name.lower().startswith(('icu', 'msvcp', 'vcruntime')):
            items.append({'dll': name, 'resolved': path.value, 'missing_exports': missing})
    except OSError as error:
        items.append({'dll': name, 'error': str(error)})
output = root / 'output/validation/p01/dll-diagnostics.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(items, indent=2) + '\n', encoding='utf-8')
print(json.dumps(items, indent=2))
