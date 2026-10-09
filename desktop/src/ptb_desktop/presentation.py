"""Preserve the normal surface before the first native maximize transition."""
from PyQt6 import sip
from PyQt6.QtCore import QObject, QEvent, QTimer, Qt
from PyQt6.QtGui import QColor, QPalette, QPixmap
from PyQt6.QtWidgets import QLabel, QWidget
from PyQt6.QtQuickWidgets import QQuickWidget
from PyQt6.QtWebEngineCore import QWebEngineScript


class FirstMaximizePresentation(QObject):
    def __init__(self, view, page):
        super().__init__(view)
        self.view, self.page = view, page
        self.used = False
        self.active = False
        self.frames = 0
        self.document_ready = False
        self.generation = 0
        self.waiting = False
        self.quick = None
        self.cover = QLabel(view)
        self.cover.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cover.setScaledContents(True)
        self.cover.setAutoFillBackground(True)
        self.cover.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.cover.hide()
        self.watchdog = QTimer(self)
        self.watchdog.setSingleShot(True)
        self.watchdog.timeout.connect(self.timed_out)
        view.installEventFilter(self)

    def set_color(self, color):
        palette = self.cover.palette()
        palette.setColor(QPalette.ColorRole.Window, color)
        palette.setColor(QPalette.ColorRole.WindowText, QColor('#eeeeee' if color.lightness() < 128 else '#333333'))
        self.cover.setPalette(palette)

    def attach(self):
        # WebEngine creates its Quick child lazily. Use its public Quick API;
        # never request a native winId for this offscreen rendering widget.
        for child in self.view.findChildren(QWidget):
            if not child.inherits('QQuickWidget'):
                continue
            quick = sip.cast(child, QQuickWidget)
            if self.quick is quick:
                return
            self.quick = quick
            if self.active:
                self.cover.raise_()
                quick.update()
            return

    def prepare(self, *, capture=True):
        if self.used:
            return
        self.used = True
        # Called before Windows changes the client size. Grabbing after Resize
        # can read back a newly allocated black texture, or stall the GUI thread.
        picture = self.view.grab() if capture and not self.page.isLoading() else QPixmap()
        self.cover.setPixmap(picture)
        if picture.isNull() or not self.valid_picture(picture.toImage()):
            self.cover.clear()
            self.cover.setText('正在打开工作台…')
        self.active = True
        self.frames = 0
        self.document_ready = False
        self.cover.setGeometry(self.view.rect())
        self.cover.show()
        self.cover.raise_()
        self.attach()
        # Present the cover at the normal geometry, before the OS animation.
        self.cover.repaint()

    def begin(self):
        if not self.used:
            # Programmatic/non-Windows paths may arrive after Resize. Never
            # capture that surface; an opaque themed cover is safe at this point.
            self.prepare(capture=False)
        if not self.active or self.waiting:
            return
        self.waiting = True
        self.watchdog.start(6000)
        self.restart()
        self.view.update()

    def restart(self):
        self.generation += 1
        self.document_ready = False
        self.frames = 0
        self.check_document(self.generation)

    def check_document(self, generation):
        if not self.active or generation != self.generation:
            return
        zoom = self.page.zoomFactor()
        width, height = round(self.view.width()/zoom), round(self.view.height()/zoom)
        # A DOM being present says nothing about the Chromium surface size.
        # Two animation frames at this viewport must precede the Qt frame check.
        # ApplicationWorld keeps the marker away from application/page scripts.
        code = f'''(()=>{{
            if (!document.getElementById("app")?.childElementCount ||
                innerWidth !== {width} || innerHeight !== {height}) return false;
            const key = "__ptbFirstMaximize", token = {generation};
            let barrier = window[key];
            if (!barrier || barrier.token !== token) {{
                barrier = window[key] = {{token, ready:false}};
                requestAnimationFrame(()=>requestAnimationFrame(()=>{{
                    if (innerWidth === {width} && innerHeight === {height}) barrier.ready = true;
                }}));
            }}
            return barrier.ready;
        }})()'''
        self.page.runJavaScript(code, QWebEngineScript.ScriptWorldId.ApplicationWorld,
            lambda ready: self.document_checked(generation, ready))

    def document_checked(self, generation, ready):
        if not self.active or generation != self.generation:
            return
        if not ready or self.page.isLoading():
            QTimer.singleShot(50, lambda: self.check_document(generation))
            return
        self.document_ready = True
        self.frames = 0
        self.view.update()
        QTimer.singleShot(0, lambda: self.check_picture(generation))

    @staticmethod
    def valid_picture(picture):
        if picture.isNull():
            return False
        colors = [picture.pixelColor(x, y) for x in range(0, picture.width(), max(1, picture.width()//24))
                  for y in range(0, picture.height(), max(1, picture.height()//16))]
        # Our darkest theme has nonblack panels. A freshly cleared all-black
        # GPU target must not cause the native cover to disappear.
        return sum(max(c.red(), c.green(), c.blue()) >= 12 for c in colors) > len(colors)/20

    def check_picture(self, generation):
        if not self.active or generation != self.generation:
            return
        self.attach()
        if self.quick is None or self.quick.size() != self.view.size():
            self.frames = 0
        else:
            # An opaque QWidget above the Quick child can suppress ordinary Qt
            # paints. Explicit framebuffer rendering works even while covered;
            # waiting for two update()/afterFrameEnd signals can otherwise hang.
            picture = self.quick.grabFramebuffer()
            ratio = self.quick.devicePixelRatioF()
            sized = abs(picture.width()-round(self.view.width()*ratio)) <= 1 and abs(picture.height()-round(self.view.height()*ratio)) <= 1
            self.frames = self.frames+1 if self.document_ready and sized and self.valid_picture(picture) else 0
            if self.frames >= 2:
                self.finish()
                return
        QTimer.singleShot(16, lambda: self.check_picture(generation))

    def timed_out(self):
        if not self.active:
            return
        # A timeout is not evidence of a visible Chromium frame. Keep an opaque
        # surface and allow the native titlebar to restore/close the window.
        self.cover.clear()
        self.cover.setText('正在调整窗口显示，可还原窗口后重试…')
        self.view.update()
        if self.quick is not None:
            self.quick.update()

    def finish(self):
        self.active = False
        self.waiting = False
        self.generation += 1
        self.watchdog.stop()
        self.cover.hide()
        self.cover.setPixmap(QPixmap())

    def cancel(self):
        self.finish()
        # A restore before the first valid enlarged frame must not consume the
        # protection for the next attempt.
        self.used = False

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.ChildAdded:
            QTimer.singleShot(0, self.attach)
        elif event.type() == QEvent.Type.Resize and self.active:
            self.frames = 0
            self.cover.setGeometry(self.view.rect())
            self.cover.raise_()
            if self.waiting:
                self.restart()
        return False
