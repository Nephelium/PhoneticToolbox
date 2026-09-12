"""Capture only after this window has left the desktop compositor."""
import sys
from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication


def capture_hidden(window):
    screen = window.screen()
    visible, state = window.isVisible(), window.windowState()
    window.hide()
    try:
        # hide/processEvents alone can still grab the preceding compositor frame,
        # including the just-closed confirmation dialog. Yield a full timer turn.
        loop = QEventLoop()
        QTimer.singleShot(250, loop.quit)
        loop.exec()
        if sys.platform == 'win32':
            import ctypes
            result = ctypes.WinDLL('dwmapi').DwmFlush()
            if result != 0:
                raise RuntimeError('桌面尚未完成隐藏，请重试截图。')
        pixmap = screen.grabWindow(0)
        if pixmap.isNull():
            raise RuntimeError('截图失败。')
        return pixmap
    finally:
        window.setWindowState(state)
        if visible:
            window.show()
            window.activateWindow()
