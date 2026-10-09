"""Lightweight branded Win32 preparation card; no Qt, image or font dependency."""
import ctypes as c
from ctypes import wintypes as w
import os
from pathlib import Path
import threading
import time


LABELS = {
    'starting': '正在启动 PhoneticToolbox',
    'checking': '正在检查运行文件',
    'verify-science': '正在检查科学计算组件',
    'verify-host': '正在检查界面组件',
    'verify-apps': '正在检查程序文件',
    'loading': '正在加载主界面',
    'slow-loading': '主界面仍在加载，请稍候',
    'science': '正在准备科学计算组件',
    'host': '正在准备界面组件',
    'apps': '正在准备程序文件',
    'repair': '正在修复不完整的运行文件',
    'waiting': '正在等待另一个窗口完成准备',
}
STEPS = {'science': 1, 'host': 2, 'apps': 3,
         'verify-science': 1, 'verify-host': 2, 'verify-apps': 3}


def _rgb(value):
    return int(value[0:2], 16) | int(value[2:4], 16) << 8 | int(value[4:6], 16) << 16


class _Paint(c.Structure):
    _fields_ = [('dc', w.HDC), ('erase', w.BOOL), ('rect', w.RECT),
                ('restore', w.BOOL), ('update', w.BOOL), ('reserved', w.BYTE * 32)]


class _Vertex(c.Structure):
    _fields_ = [('x', w.LONG), ('y', w.LONG), ('r', w.WORD),
                ('g', w.WORD), ('b', w.WORD), ('a', w.WORD)]


class _Gradient(c.Structure):
    _fields_ = [('first', w.ULONG), ('second', w.ULONG)]


class _Canvas:
    """DPI-scaled, double-buffered GDI drawing with owned native handles."""
    def __init__(self, user, dpi):
        self.user, self.dpi, self.fonts, self.resolved_fonts = user, dpi, {}, {}
        self.gdi = c.WinDLL('gdi32', use_last_error=True)
        self.msimg = c.WinDLL('msimg32', use_last_error=True)
        def api(dll, name, result, *arguments):
            function = getattr(dll, name)
            function.restype, function.argtypes = result, arguments
        api(user, 'GetClientRect', w.BOOL, w.HWND, c.POINTER(w.RECT))
        api(user, 'DrawTextW', c.c_int, w.HDC, w.LPCWSTR, c.c_int, c.POINTER(w.RECT), w.UINT)
        api(user, 'DrawIconEx', w.BOOL, w.HDC, c.c_int, c.c_int, w.HICON, c.c_int, c.c_int, w.UINT, w.HBRUSH, w.UINT)
        api(user, 'FillRect', c.c_int, w.HDC, c.POINTER(w.RECT), w.HBRUSH)
        api(self.gdi, 'CreateCompatibleDC', w.HDC, w.HDC)
        api(self.gdi, 'CreateCompatibleBitmap', w.HBITMAP, w.HDC, c.c_int, c.c_int)
        api(self.gdi, 'SelectObject', w.HGDIOBJ, w.HDC, w.HGDIOBJ)
        api(self.gdi, 'GetStockObject', w.HGDIOBJ, c.c_int)
        api(self.gdi, 'DeleteObject', w.BOOL, w.HGDIOBJ)
        api(self.gdi, 'DeleteDC', w.BOOL, w.HDC)
        api(self.gdi, 'CreateSolidBrush', w.HBRUSH, w.DWORD)
        api(self.gdi, 'CreatePen', w.HGDIOBJ, c.c_int, c.c_int, w.DWORD)
        api(self.gdi, 'RoundRect', w.BOOL, w.HDC, *([c.c_int] * 6))
        api(self.gdi, 'CreateRoundRectRgn', w.HRGN, *([c.c_int] * 6))
        api(self.gdi, 'SelectClipRgn', c.c_int, w.HDC, w.HRGN)
        api(self.gdi, 'BitBlt', w.BOOL, w.HDC, c.c_int, c.c_int, c.c_int, c.c_int, w.HDC, c.c_int, c.c_int, w.DWORD)
        api(self.gdi, 'SetTextColor', w.DWORD, w.HDC, w.DWORD)
        api(self.gdi, 'SetBkMode', c.c_int, w.HDC, c.c_int)
        api(self.gdi, 'GetTextFaceW', c.c_int, w.HDC, c.c_int, w.LPWSTR)
        api(self.gdi, 'CreateFontW', w.HFONT, *([c.c_int] * 5), *([w.DWORD] * 8), w.LPCWSTR)
        api(self.msimg, 'GradientFill', w.BOOL, w.HDC, c.POINTER(_Vertex), w.ULONG, c.POINTER(_Gradient), w.ULONG, w.ULONG)

    def px(self, value):
        return round(value * self.dpi / 96)

    def rect(self, x, y, width, height):
        return w.RECT(self.px(x), self.px(y), self.px(x + width), self.px(y + height))

    def font(self, face, size, weight):
        key = face, size, weight
        if key not in self.fonts:
            # Requested Windows fonts; normal OS font fallback remains available
            # when an optional CJK font is absent. Never install fonts globally.
            self.fonts[key] = self.gdi.CreateFontW(-self.px(size), 0, 0, 0, weight,
                                                   0, 0, 0, 1, 0, 0, 5, 0, face)
            if not self.fonts[key]:
                raise c.WinError(c.get_last_error())
        return self.fonts[key]

    def text(self, dc, value, box, *, size=17, english=False, color='52657E', weight=400, right=False):
        face = 'Times New Roman' if english else 'KaiTi'
        previous = self.gdi.SelectObject(dc, self.font(face, size, weight))
        try:
            actual = c.create_unicode_buffer(128)
            self.gdi.GetTextFaceW(dc, len(actual), actual)
            self.resolved_fonts[face] = actual.value
            self.gdi.SetBkMode(dc, 1)
            self.gdi.SetTextColor(dc, _rgb(color))
            rectangle = self.rect(*box)
            self.user.DrawTextW(dc, value, -1, c.byref(rectangle), 0x824 | (2 if right else 0))
        finally:
            self.gdi.SelectObject(dc, previous)

    def rounded(self, dc, box, radius, fill, border=None):
        rectangle = self.rect(*box)
        brush = self.gdi.CreateSolidBrush(_rgb(fill))
        pen = self.gdi.CreatePen(0, max(1, self.px(1)), _rgb(border)) if border else self.gdi.GetStockObject(8)
        old_brush, old_pen = self.gdi.SelectObject(dc, brush), self.gdi.SelectObject(dc, pen)
        try:
            self.gdi.RoundRect(dc, rectangle.left, rectangle.top, rectangle.right, rectangle.bottom,
                               self.px(radius * 2), self.px(radius * 2))
        finally:
            self.gdi.SelectObject(dc, old_brush); self.gdi.SelectObject(dc, old_pen)
            self.gdi.DeleteObject(brush)
            if border:
                self.gdi.DeleteObject(pen)

    def gradient(self, dc, box, first, second, *, vertical=False):
        rectangle = self.rect(*box)
        def vertex(x, y, color):
            return _Vertex(x, y, *(int(color[i:i + 2], 16) * 256 for i in (0, 2, 4)), 0)
        vertices = (_Vertex * 2)(vertex(rectangle.left, rectangle.top, first),
                                  vertex(rectangle.right, rectangle.bottom, second))
        mesh = _Gradient(0, 1)
        self.msimg.GradientFill(dc, vertices, 2, c.byref(mesh), 1, int(vertical))

    def draw(self, hwnd, target, icon, state, motion):
        rectangle = w.RECT()
        self.user.GetClientRect(hwnd, c.byref(rectangle))
        width, height = rectangle.right, rectangle.bottom
        dc = self.gdi.CreateCompatibleDC(target)
        bitmap = self.gdi.CreateCompatibleBitmap(target, width, height)
        if not dc or not bitmap:
            if dc: self.gdi.DeleteDC(dc)
            if bitmap: self.gdi.DeleteObject(bitmap)
            raise c.WinError(c.get_last_error())
        old = self.gdi.SelectObject(dc, bitmap)
        try:
            self.gradient(dc, (0, 0, 620, 312), 'FFFFFF', 'F9FBFF', vertical=True)
            self.rounded(dc, (0, 0, 620, 312), 17, 'FBFCFF', 'DCE5F0')
            self.user.DrawIconEx(dc, self.px(36), self.px(32), icon, self.px(72), self.px(72), 0, None, 3)
            self.text(dc, 'PhoneticToolbox', (128, 34, 450, 43), english=True, size=34, color='233A59')
            self.text(dc, '语音分析与实验工具箱', (130, 81, 420, 26), size=18, color='60748F')
            self.gradient(dc, (36, 126, 548, 1), 'D9E5F4', 'F0F4FA')
            stage, label, percent, known = state
            self.text(dc, label, (36, 150, 453, 31), size=22, color='263D5C')
            self.text(dc, f'{percent}%' if known else '…', (486, 145, 98, 39),
                      size=32, english=True, color='387ED0', right=True)
            self.text(dc, '正在检查运行文件并加载界面，请耐心等待。', (36, 189, 548, 25), size=16)
            self.rounded(dc, (36, 232, 548, 10), 5, 'E6EDF7')
            if known:
                start, length = 36, 548 * percent / 100
            else:
                start, length = 36 + 448 * motion, 100
            if length > 0:
                region_box = self.rect(start, 232, max(length, 1), 10)
                region = self.gdi.CreateRoundRectRgn(region_box.left, region_box.top, region_box.right,
                                                     region_box.bottom, self.px(10), self.px(10))
                self.gdi.SelectClipRgn(dc, region)
                try:
                    self.gradient(dc, (start, 232, max(length, 1), 10), '87C6F4', '397FD2')
                finally:
                    self.gdi.SelectClipRgn(dc, None); self.gdi.DeleteObject(region)
            self.text(dc, '主界面就绪后自动关闭此提示', (36, 264, 450, 25), size=15, color='71829A')
            step = STEPS.get(stage)
            self.text(dc, f'{step} / 3' if step else '', (510, 264, 74, 25),
                      size=17, english=True, color='71829A', right=True)
            self.gdi.BitBlt(target, 0, 0, width, height, dc, 0, 0, 0x00CC0020)
        finally:
            self.gdi.SelectObject(dc, old); self.gdi.DeleteObject(bitmap); self.gdi.DeleteDC(dc)

    def close(self):
        for font in self.fonts.values():
            self.gdi.DeleteObject(font)
        self.fonts.clear()


class PreparationWindow:
    def __init__(self, *, offscreen=False, dpi=None, icon_path=None):
        self.hwnd = None; self.thread = None; self.failure = None
        self.message = ''; self.percent = 0
        self.stopped = threading.Event(); self.visible = threading.Event()
        self.quiet = bool(os.environ.get('PTB_OWNED_BOOTSTRAP_LOG'))
        self.offscreen, self.test_dpi, self.icon_path = offscreen, dpi, icon_path
        if dpi is not None and (not offscreen or dpi not in (96, 120, 144, 192, 240)):
            raise ValueError('Explicit DPI is only for an owned offscreen preview')
        self.state = ('starting', LABELS['starting'], 0, False)
        self.resolved_fonts = {}

    def update(self, stage, done, total):
        self.message = LABELS.get(stage, '正在准备运行文件')
        self.percent = min(100, max(0, int(100 * done / total))) if total else 0
        self.state = (stage, self.message, self.percent, total > 0)
        if self.thread is None and not self.quiet and not self.stopped.is_set():
            self.thread = threading.Thread(target=self._run, name='ptb-preparation-window', daemon=True)
            self.thread.start()

    def _run(self):
        try:
            self._window()
        except Exception as error:
            self.failure = error; self.visible.set()

    def _window(self):
        user = c.WinDLL('user32', use_last_error=True)
        kernel = c.WinDLL('kernel32', use_last_error=True)
        user.SetThreadDpiAwarenessContext.argtypes = (c.c_void_p,)
        user.SetThreadDpiAwarenessContext.restype = c.c_void_p
        prior_dpi = user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
        canvas = _Canvas(user, self.test_dpi or user.GetDpiForSystem())
        self.dpi = canvas.dpi
        kernel.GetModuleHandleW.restype = w.HMODULE
        instance = kernel.GetModuleHandleW(None)
        user.LoadImageW.argtypes = (w.HINSTANCE, w.LPCWSTR, w.UINT, c.c_int, c.c_int, w.UINT)
        user.LoadImageW.restype = w.HANDLE
        user.DestroyIcon.argtypes = (w.HICON,)
        # The application artwork already lives in the EXE icon resources.
        # Source previews receive the same ICO explicitly, without adding Qt.
        resource = str(Path(self.icon_path)) if self.icon_path else c.cast(c.c_void_p(1), w.LPCWSTR)
        icon = user.LoadImageW(None if self.icon_path else instance, resource, 1, canvas.px(72), canvas.px(72), 0x10 if self.icon_path else 0)
        if not icon:
            canvas.close()
            if prior_dpi: user.SetThreadDpiAwarenessContext(prior_dpi)
            raise RuntimeError('应用图标未能读取。')
        icons = [icon]
        proc_type = c.WINFUNCTYPE(c.c_ssize_t, w.HWND, w.UINT, w.WPARAM, w.LPARAM)
        user.DefWindowProcW.argtypes = (w.HWND, w.UINT, w.WPARAM, w.LPARAM)
        user.DefWindowProcW.restype = c.c_ssize_t
        user.BeginPaint.argtypes = (w.HWND, c.POINTER(_Paint)); user.BeginPaint.restype = w.HDC
        user.EndPaint.argtypes = (w.HWND, c.POINTER(_Paint))
        user.GetWindowRect.argtypes = (w.HWND, c.POINTER(w.RECT))
        user.SetWindowPos.argtypes = (w.HWND, w.HWND, c.c_int, c.c_int, c.c_int, c.c_int, w.UINT)
        user.InvalidateRect.argtypes = (w.HWND, c.POINTER(w.RECT), w.BOOL)
        user.UpdateWindow.argtypes = (w.HWND,)
        user.SetWindowRgn.argtypes = (w.HWND, w.HRGN, w.BOOL)
        user.PeekMessageW.argtypes = (c.POINTER(w.MSG), w.HWND, w.UINT, w.UINT, w.UINT)
        user.TranslateMessage.argtypes = (c.POINTER(w.MSG),)
        user.DispatchMessageW.argtypes = (c.POINTER(w.MSG),)
        user.DispatchMessageW.restype = c.c_ssize_t
        user.SystemParametersInfoW.argtypes = (w.UINT, w.UINT, c.c_void_p, w.UINT)
        animate = w.BOOL(True)
        user.SystemParametersInfoW(0x1042, 0, c.byref(animate), 0)
        motion = 0.5
        @proc_type
        def window_proc(hwnd, msg, wp, lp):
            nonlocal canvas, icon
            try:
                if msg == 0x10: return 0  # Preparation has atomic publication.
                if msg == 0x14: return 1  # Full double-buffered painting.
                if msg == 0xF:
                    paint = _Paint(); dc = user.BeginPaint(hwnd, c.byref(paint))
                    try: canvas.draw(hwnd, dc, icon, self.state, motion)
                    finally: user.EndPaint(hwnd, c.byref(paint))
                    return 0
                if msg in (0x317, 0x318):
                    canvas.draw(hwnd, w.HDC(wp), icon, self.state, motion); return 0
                if msg == 0x84:
                    bounds = w.RECT(); user.GetWindowRect(hwnd, c.byref(bounds))
                    y = c.c_short((lp >> 16) & 0xffff).value - bounds.top
                    return 2 if 0 <= y < canvas.px(126) else 1
                if msg == 0x2E0 and self.test_dpi is None:
                    canvas.close(); canvas = _Canvas(user, wp & 0xffff); self.dpi = canvas.dpi
                    scaled_icon = user.LoadImageW(None if self.icon_path else instance, resource, 1,
                                                   canvas.px(72), canvas.px(72), 0x10 if self.icon_path else 0)
                    if scaled_icon:
                        icons.append(scaled_icon); icon = scaled_icon
                    bounds = c.cast(lp, c.POINTER(w.RECT)).contents
                    user.SetWindowPos(hwnd, None, bounds.left, bounds.top, canvas.px(620), canvas.px(312), 0x14)
                    region = canvas.gdi.CreateRoundRectRgn(0, 0, canvas.px(620) + 1, canvas.px(312) + 1,
                                                           canvas.px(34), canvas.px(34))
                    if not user.SetWindowRgn(hwnd, region, True): canvas.gdi.DeleteObject(region)
                    user.InvalidateRect(hwnd, None, False)
                    return 0
            except Exception as error:
                self.failure = error; self.stopped.set()
                return 0
            return user.DefWindowProcW(hwnd, msg, wp, lp)
        class WindowClass(c.Structure):
            _fields_ = [('style',w.UINT),('proc',proc_type),('classExtra',c.c_int),('windowExtra',c.c_int),
                        ('instance',w.HINSTANCE),('icon',w.HICON),('cursor',w.HANDLE),('background',w.HBRUSH),
                        ('menu',w.LPCWSTR),('name',w.LPCWSTR)]
        class_name = f'PTBPreparation-{id(self):x}'
        cls = WindowClass(0x20000, window_proc, 0, 0, instance, icon, None, None, None, class_name)
        user.RegisterClassW.argtypes = (c.POINTER(WindowClass),)
        user.CreateWindowExW.argtypes = (w.DWORD,w.LPCWSTR,w.LPCWSTR,w.DWORD,c.c_int,c.c_int,c.c_int,c.c_int,w.HWND,w.HMENU,w.HINSTANCE,c.c_void_p)
        user.CreateWindowExW.restype = w.HWND
        user.DestroyWindow.argtypes = (w.HWND,)
        user.UnregisterClassW.argtypes = (w.LPCWSTR,w.HINSTANCE)
        registered = False
        try:
            if not user.RegisterClassW(c.byref(cls)): raise c.WinError(c.get_last_error())
            registered = True
            width, height = canvas.px(620), canvas.px(312)
            x = -10000 if self.offscreen else (user.GetSystemMetrics(0) - width) // 2
            y = (user.GetSystemMetrics(1) - height) // 2
            self.hwnd = user.CreateWindowExW(0x40000, class_name, 'PhoneticToolbox · 正在启动',
                                             0x90000000, x, y, width, height, None, None, instance, None)
            if not self.hwnd: raise c.WinError(c.get_last_error())
            # Windows owns the successful region; it is released with the HWND.
            region = canvas.gdi.CreateRoundRectRgn(0, 0, width + 1, height + 1, canvas.px(34), canvas.px(34))
            if not user.SetWindowRgn(self.hwnd, region, True): canvas.gdi.DeleteObject(region)
            user.UpdateWindow(self.hwnd); self.visible.set()
            message = w.MSG(); previous = None
            while not self.stopped.wait(.04):
                while user.PeekMessageW(c.byref(message), None, 0, 0, 1):
                    user.TranslateMessage(c.byref(message)); user.DispatchMessageW(c.byref(message))
                state = self.state
                if not state[3] and animate.value:
                    phase = time.monotonic() % 2.4 / 1.2
                    motion = phase if phase <= 1 else 2 - phase
                else: motion = .5
                if state != previous or not state[3]:
                    user.InvalidateRect(self.hwnd, None, False); user.UpdateWindow(self.hwnd)
                    self.resolved_fonts = dict(canvas.resolved_fonts)
                previous = state
        finally:
            if self.hwnd: user.DestroyWindow(self.hwnd)
            if registered: user.UnregisterClassW(class_name, instance)
            canvas.close()
            for owned_icon in icons: user.DestroyIcon(owned_icon)
            if prior_dpi: user.SetThreadDpiAwarenessContext(prior_dpi)

    def close(self):
        self.stopped.set()
        if self.thread: self.thread.join(3)
        if self.thread and self.thread.is_alive():
            raise RuntimeError('准备提示窗未能正常关闭。')
        if self.failure:
            raise RuntimeError('启动提示窗未能完成绘制。') from self.failure
