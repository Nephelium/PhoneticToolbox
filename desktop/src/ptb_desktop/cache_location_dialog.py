"""Native folder selection before the portable payload is unpacked; no Qt."""
import ctypes as c
from ctypes import wintypes as w
from pathlib import Path


def choose_directory(reason=''):
    shell = c.WinDLL('shell32', use_last_error=True)
    ole = c.OleDLL('ole32')
    user = c.WinDLL('user32', use_last_error=True)
    ole.CoInitializeEx.argtypes = (c.c_void_p, w.DWORD)
    ole.CoInitializeEx.restype = c.c_long
    ole.CoUninitialize.argtypes = ()
    ole.CoTaskMemFree.argtypes = (c.c_void_p,)
    user.MessageBoxW.argtypes = (w.HWND, w.LPCWSTR, w.LPCWSTR, w.UINT)
    user.MessageBoxW.restype = c.c_int
    message = ('请选择免安装版的缓存保存位置。\n\n'
               '程序会在所选位置创建 PhoneticToolbox-cache 文件夹，'
               '存放运行文件、更新下载及普通浏览器缓存。少量图形缓存可在设置中查看和清理。\n'
               '替换 EXE 更新后继续使用此目录。停止使用时，关闭程序后可将它一起删除。\n'
               '设置、草稿、录音和研究资料仍保存在原位置。\n\n'
               '点击“确定”选择位置，点击“取消”退出。')
    if reason:
        message = '原缓存位置暂时无法使用：\n' + reason + '\n\n' + message
    if user.MessageBoxW(None, message, '选择缓存保存位置', 0x1 | 0x40) != 1:
        return None
    result = ole.CoInitializeEx(None, 2)  # New-style picker requires an STA.
    if result not in (0, 1):
        raise OSError('文件夹选择器未能启动，请关闭程序后重试。')
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
        info = BrowseInfo(None, None, c.cast(display, w.LPWSTR), '选择缓存保存位置，将创建 PhoneticToolbox-cache 文件夹',
                          0x1 | 0x40 | 0x10, None, 0, 0)
        item = shell.SHBrowseForFolderW(c.byref(info))
        if not item:
            return None
        try:
            path = c.create_unicode_buffer(32768)
            if not shell.SHGetPathFromIDListEx(item, path, len(path), 0):
                raise OSError('所选位置无法保存文件，请选择本机磁盘中的文件夹。')
            return Path(path.value)
        finally:
            ole.CoTaskMemFree(item)
    finally:
        ole.CoUninitialize()
