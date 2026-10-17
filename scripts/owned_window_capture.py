"""Capture only an owned native window, independent of foreground focus."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess


def capture_owned_window(root_pid, target):
    from PIL import Image
    user, gdi = ctypes.windll.user32, ctypes.windll.gdi32
    user.SetProcessDPIAware()
    rows = json.loads(subprocess.check_output(['powershell.exe', '-NoProfile', '-Command',
        'Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId | ConvertTo-Json -Compress'], encoding='utf8'))
    owned = {root_pid}
    while True:
        children = {row['ProcessId'] for row in rows if row['ParentProcessId'] in owned}
        if children <= owned:
            break
        owned.update(children)
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    windows = []
    def visit(handle, _):
        pid, rect = wintypes.DWORD(), wintypes.RECT()
        user.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        if pid.value in owned and user.IsWindowVisible(handle):
            user.GetWindowRect(handle, ctypes.byref(rect))
            title = ctypes.create_unicode_buffer(256)
            user.GetWindowTextW(handle, title, len(title))
            width, height = rect.right - rect.left, rect.bottom - rect.top
            if width > 800 and height > 500 and title.value.startswith('PhoneticToolbox'):
                windows.append((width * height, handle, pid.value, title.value, width, height))
        return True
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(visit)
    user.EnumWindows(callback, 0)
    if not windows:
        raise RuntimeError('No owned visible application window')
    _, handle, pid, title, width, height = max(windows)
    user.GetWindowDC.argtypes = [wintypes.HWND]; user.GetWindowDC.restype = wintypes.HDC
    user.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    gdi.CreateCompatibleDC.argtypes = [wintypes.HDC]; gdi.CreateCompatibleDC.restype = wintypes.HDC
    gdi.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]; gdi.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]; gdi.SelectObject.restype = wintypes.HANDLE
    gdi.DeleteObject.argtypes = [wintypes.HANDLE]; gdi.DeleteDC.argtypes = [wintypes.HDC]
    class Header(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('width', wintypes.LONG), ('height', wintypes.LONG),
                    ('planes', wintypes.WORD), ('bits', wintypes.WORD), ('compression', wintypes.DWORD),
                    ('image_size', wintypes.DWORD), ('xppm', wintypes.LONG), ('yppm', wintypes.LONG),
                    ('used', wintypes.DWORD), ('important', wintypes.DWORD)]
    gdi.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                            wintypes.LPVOID, ctypes.POINTER(Header), wintypes.UINT]
    dc = user.GetWindowDC(handle); memory = gdi.CreateCompatibleDC(dc)
    bitmap = gdi.CreateCompatibleBitmap(dc, width, height); previous = gdi.SelectObject(memory, bitmap)
    try:
        if not user.PrintWindow(handle, memory, 2):
            raise RuntimeError('Owned PrintWindow capture failed; no desktop fallback')
        header = Header(ctypes.sizeof(Header), width, -height, 1, 32, 0, width * height * 4, 0, 0, 0, 0)
        pixels = ctypes.create_string_buffer(width * height * 4)
        if gdi.GetDIBits(memory, bitmap, 0, height, pixels, ctypes.byref(header), 0) != height:
            raise RuntimeError('Owned window bitmap unavailable')
        Image.frombytes('RGB', (width, height), pixels.raw, 'raw', 'BGRX').save(Path(target))
    finally:
        gdi.SelectObject(memory, previous); gdi.DeleteObject(bitmap); gdi.DeleteDC(memory); user.ReleaseDC(handle, dc)
    return dict(hwnd=handle, pid=pid, rootPid=root_pid, title=title, width=width, height=height,
                method='PrintWindow(PW_RENDERFULLCONTENT), owned PID/title; no desktop capture')
