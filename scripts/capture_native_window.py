"""Capture one owned HWND via PrintWindow, never the desktop behind it."""
import ctypes
from ctypes import wintypes as wt


def capture_window(handle):
    from PIL import Image
    user, gdi = ctypes.WinDLL('user32'), ctypes.WinDLL('gdi32')
    for dll, name, args, result in (
        (user, 'GetWindowRect', [wt.HWND, ctypes.POINTER(wt.RECT)], wt.BOOL),
        (user, 'GetWindowDC', [wt.HWND], wt.HDC),
        (user, 'ReleaseDC', [wt.HWND, wt.HDC], ctypes.c_int),
        (user, 'PrintWindow', [wt.HWND, wt.HDC, wt.UINT], wt.BOOL),
        (gdi, 'CreateCompatibleDC', [wt.HDC], wt.HDC),
        (gdi, 'CreateCompatibleBitmap', [wt.HDC, ctypes.c_int, ctypes.c_int], wt.HBITMAP),
        (gdi, 'SelectObject', [wt.HDC, wt.HGDIOBJ], wt.HGDIOBJ),
        (gdi, 'DeleteObject', [wt.HGDIOBJ], wt.BOOL),
        (gdi, 'DeleteDC', [wt.HDC], wt.BOOL),
        (gdi, 'GetDIBits', [wt.HDC, wt.HBITMAP, wt.UINT, wt.UINT, ctypes.c_void_p, ctypes.c_void_p, wt.UINT], ctypes.c_int),
    ):
        function = getattr(dll, name)
        function.argtypes, function.restype = args, result

    class Header(ctypes.Structure):
        _fields_ = [('size', wt.DWORD), ('width', wt.LONG), ('height', wt.LONG),
                    ('planes', wt.WORD), ('bits', wt.WORD), ('compression', wt.DWORD),
                    ('image_bytes', wt.DWORD), ('xppm', wt.LONG), ('yppm', wt.LONG),
                    ('colours', wt.DWORD), ('important', wt.DWORD)]

    bounds = wt.RECT()
    assert user.GetWindowRect(wt.HWND(handle), ctypes.byref(bounds))
    width, height = bounds.right - bounds.left, bounds.bottom - bounds.top
    source = user.GetWindowDC(wt.HWND(handle))
    destination = gdi.CreateCompatibleDC(source)
    bitmap = gdi.CreateCompatibleBitmap(source, width, height)
    previous = gdi.SelectObject(destination, bitmap)
    try:
        assert user.PrintWindow(wt.HWND(handle), destination, 2), 'PrintWindow failed'
        header = Header(ctypes.sizeof(Header), width, -height, 1, 32, 0, width * height * 4, 0, 0, 0, 0)
        data = ctypes.create_string_buffer(width * height * 4)
        gdi.SelectObject(destination, previous)
        assert gdi.GetDIBits(destination, bitmap, 0, height, data, ctypes.byref(header), 0) == height
        return Image.frombytes('RGB', (width, height), data.raw, 'raw', 'BGRX')
    finally:
        gdi.SelectObject(destination, previous)
        gdi.DeleteObject(bitmap)
        gdi.DeleteDC(destination)
        user.ReleaseDC(wt.HWND(handle), source)
