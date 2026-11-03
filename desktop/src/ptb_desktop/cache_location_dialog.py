"""Native folder selection before the portable payload is unpacked; no Qt."""
import ctypes as c
from ctypes import wintypes as w
from pathlib import Path


def choose_directory(reason=''):
    from .native_i18n import NativeLocale
    from .platform_paths import user_data_root
    language=NativeLocale(user_data_root()/'language.json')
    shell = c.WinDLL('shell32', use_last_error=True)
    ole = c.OleDLL('ole32')
    user = c.WinDLL('user32', use_last_error=True)
    ole.CoInitializeEx.argtypes = (c.c_void_p, w.DWORD)
    ole.CoInitializeEx.restype = c.c_long
    ole.CoUninitialize.argtypes = ()
    ole.CoTaskMemFree.argtypes = (c.c_void_p,)
    user.MessageBoxW.argtypes = (w.HWND, w.LPCWSTR, w.LPCWSTR, w.UINT)
    user.MessageBoxW.restype = c.c_int
    message=language.text('cache.chooseBody')
    if reason:
        message=language.text('cache.unavailable',reason=language.error(reason),message=message)
    if user.MessageBoxW(None, message, language.text('cache.chooseTitle'), 0x1 | 0x40) != 1:
        return None
    result = ole.CoInitializeEx(None, 2)  # New-style picker requires an STA.
    if result not in (0, 1):
        raise OSError(language.text('cache.pickerFailed'))
    try:
        class BrowseInfo(c.Structure):
            _fields_ = [('owner', w.HWND), ('root', c.c_void_p),
                        ('display', w.LPWSTR), ('title', w.LPCWSTR),
                        ('flags', w.UINT), ('callback', c.c_void_p),
                        ('parameter', w.LPARAM), ('image', c.c_int)]
        shell.SHBrowseForFolderW.argtypes = (c.POINTER(BrowseInfo),)
        shell.SHBrowseForFolderW.restype = c.c_void_p
        shell.SHGetPathFromIDListEx.argtypes = (c.c_void_p, w.LPWSTR, w.DWORD, w.DWORD)
        shell.SHGetPathFromIDListEx.restype = w.BOOL
        display = c.create_unicode_buffer(32768)
        info = BrowseInfo(None, None, c.cast(display, w.LPWSTR), language.text('cache.chooseFolder'),
                          0x1 | 0x40 | 0x10, None, 0, 0)
        item = shell.SHBrowseForFolderW(c.byref(info))
        if not item:
            return None
        try:
            path = c.create_unicode_buffer(32768)
            if not shell.SHGetPathFromIDListEx(item, path, len(path), 0):
                raise OSError(language.text('cache.folderInvalid'))
            return Path(path.value)
        finally:
            ole.CoTaskMemFree(item)
    finally:
        ole.CoUninitialize()
