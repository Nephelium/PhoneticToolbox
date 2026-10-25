"""Window-local native frame integration and full-screen state.

Qt owns the compact title row; Windows retains resizing and snap styles.
The optional DWM colour path needs Windows 11; no system setting is changed.
"""
import ctypes
import re
import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def _cocoa_window_messages():
    """Typed Objective-C calls on the GUI thread; no optional Python runtime."""
    library=ctypes.CDLL('/usr/lib/libobjc.A.dylib')
    selector=library.sel_registerName;selector.argtypes=[ctypes.c_char_p];selector.restype=ctypes.c_void_p
    address=ctypes.cast(library.objc_msgSend,ctypes.c_void_p).value
    pointer=ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)(address)
    mask=ctypes.CFUNCTYPE(ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p)(address)
    toggle=ctypes.CFUNCTYPE(None,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)(address)
    return library,pointer,mask,toggle,selector(b'window'),selector(b'styleMask'),selector(b'toggleFullScreen:')


def macos_window_fullscreen(handle,*,leave=False):
    """Read the actual NSWindow state behind Qt's NSView, optionally exit it.

    Cocoa can finish entering its native Space after Qt has already processed
    an exit request. Qt's cached WindowFullScreen flag alone misses that race.
    NSWindowStyleMaskFullScreen is the public AppKit flag (1 << 14).
    """
    if sys.platform!='darwin':return False
    library,pointer,mask,toggle,window_selector,mask_selector,toggle_selector=_cocoa_window_messages()
    window=pointer(int(handle),window_selector)
    if not window:return False
    active=bool(mask(window,mask_selector)&(1<<14))
    if active and leave:toggle(window,toggle_selector,None)
    return active


def configure_integrated_frame(handle):
    """Keep Windows resizing/snap styles while Qt owns the title row."""
    from ctypes.wintypes import HWND
    user = ctypes.WinDLL('user32')
    getter = user.GetWindowLongPtrW
    setter = user.SetWindowLongPtrW
    getter.argtypes, getter.restype = [HWND, ctypes.c_int], ctypes.c_ssize_t
    setter.argtypes, setter.restype = [HWND, ctypes.c_int, ctypes.c_ssize_t], ctypes.c_ssize_t
    setter(HWND(handle), -16, getter(HWND(handle), -16) | 0x000f0000)
    position = user.SetWindowPos
    position.argtypes = [HWND, HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    position(HWND(handle), HWND(0), 0, 0, 0, 0, 0x37)


def integrated_frame_event(message, window):
    from ctypes.wintypes import HWND, MSG, RECT
    msg = MSG.from_address(int(message))
    user = ctypes.WinDLL('user32')
    # FramelessWindowHint already handles WM_NCCALCSIZE in Qt. Let the
    # platform plugin update its cached frame margins across minimize/restore.
    # Consuming it here bypasses the platform-window lookup needed for that cache.
    if msg.message != 0x84 or window.isMaximized() or window.isFullScreen():
        return None
    bounds = RECT()
    get_bounds = user.GetWindowRect
    get_bounds.argtypes = [HWND, ctypes.POINTER(RECT)]
    if not get_bounds(msg.hWnd, ctypes.byref(bounds)):
        return None
    x, y = ctypes.c_short(msg.lParam & 0xffff).value, ctypes.c_short((msg.lParam >> 16) & 0xffff).value
    margin = round(5 * window.devicePixelRatioF())
    left, right = x < bounds.left + margin, x >= bounds.right - margin
    top, bottom = y < bounds.top + margin, y >= bounds.bottom - margin
    hit = (13 if left else 14 if right else 12) if top else (16 if left else 17 if right else 15) if bottom else 10 if left else 11 if right else None
    return (True, hit) if hit is not None else None


def valid_colour(value):
    return isinstance(value, str) and re.fullmatch(r'#[0-9a-fA-F]{6}', value) is not None


def colour_ref(value):
    if not valid_colour(value):
        raise ValueError('Expected an opaque hexadecimal colour')
    red, green, blue = (int(value[i:i + 2], 16) for i in (1, 3, 5))
    return red | green << 8 | blue << 16


def apply_caption_colours(handle, background, foreground):
    if not valid_colour(background) or not valid_colour(foreground):
        return False
    if sys.platform != 'win32' or sys.getwindowsversion().build < 22000 or not handle:
        return False
    try:
        from ctypes.wintypes import DWORD, HWND
        setter = ctypes.WinDLL('dwmapi').DwmSetWindowAttribute
        setter.argtypes = [HWND, DWORD, ctypes.c_void_p, DWORD]
        setter.restype = ctypes.c_long
        # Native caption button glyphs follow the caption's light/dark mode.
        dark = ctypes.c_int(sum(int(background[i:i + 2], 16) * weight
                               for i, weight in ((1, .299), (3, .587), (5, .114))) < 128)
        caption, text = DWORD(colour_ref(background)), DWORD(colour_ref(foreground))
        outcomes = [setter(HWND(handle), attribute, ctypes.byref(value), ctypes.sizeof(value))
                    for attribute, value in ((20, dark), (35, caption), (36, text))]
        return all(result == 0 for result in outcomes)
    except (OSError, AttributeError):
        return False
