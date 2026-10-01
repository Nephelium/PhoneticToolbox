"""M09-only four-corner capture. Screen pixels stay in memory until accepted."""
import math
import sys
from PyQt6.QtCore import Qt, QEventLoop, QTimer, QPointF, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QDialog


def crop_corners(pixmap, points, width, height):
    """Map logical overlay coordinates to cropped physical image coordinates."""
    if len(points) != 4 or width <= 0 or height <= 0:
        raise ValueError('请依次选择四个角点。')
    xy = [(p.x() / width * pixmap.width(), p.y() / height * pixmap.height()) for p in points]
    cross = []
    for i in range(4):
        a, b, c = xy[i], xy[(i + 1) % 4], xy[(i + 2) % 4]
        cross.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
    if not all(v > 1 for v in cross):
        raise ValueError('角点需按左上、右上、右下、左下围成非交叉区域，请撤销或重新选择。')
    left=max(0,math.floor(min(x for x,y in xy)));top=max(0,math.floor(min(y for x,y in xy)))
    right=min(pixmap.width(),math.ceil(max(x for x,y in xy)));bottom=min(pixmap.height(),math.ceil(max(y for x,y in xy)))
    if right-left < 2 or bottom-top < 2:
        raise ValueError('所选区域过小，请重新选择。')
    cropped=pixmap.copy(left,top,right-left,bottom-top)
    cropped.setDevicePixelRatio(1)
    return cropped,[{'x':(x-left)/(right-left),'y':(y-top)/(bottom-top)} for x,y in xy]


class CornerCapture(QDialog):
    def __init__(self, pixmap, geometry):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setObjectName('m09-corner-capture')
        self.pixmap=pixmap;self.points=[];self.result_capture=None;self.error=''
        self.setGeometry(geometry);self.setCursor(Qt.CursorShape.CrossCursor)

    def paintEvent(self,event):
        painter=QPainter(self);painter.drawPixmap(self.rect(),self.pixmap)
        painter.setPen(QPen(QColor('#2979ff'),2))
        if len(self.points)>1:painter.drawPolyline(QPolygonF(self.points+([self.points[0]] if len(self.points)==4 else [])))
        for i,p in enumerate(self.points):
            painter.setBrush(QColor('#2979ff'));painter.drawEllipse(p,5,5);painter.drawText(p+QPointF(9,-9),str(i+1))
        painter.fillRect(QRectF(0,0,self.width(),64),QColor(20,24,32,235));painter.setPen(QColor('white'))
        painter.drawText(18,25,'语谱图四角：左上 → 右上 → 右下 → 左下   %s / 4' % len(self.points))
        painter.drawText(18,49,self.error or '选四角后按 Enter 完成 · Backspace / 右键撤销 · R 重新选择 · Esc 取消')

    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.RightButton:self.undo()
        elif event.button()==Qt.MouseButton.LeftButton and len(self.points)<4:
            self.points.append(event.position());self.error='';self.update()

    def undo(self):
        if self.points:self.points.pop()
        self.error='';self.update()

    def keyPressEvent(self,event):
        if event.key()==Qt.Key.Key_Escape:self.reject()
        elif event.key() in (Qt.Key.Key_Backspace,Qt.Key.Key_Delete):self.undo()
        elif event.key()==Qt.Key.Key_R:self.points=[];self.error='';self.update()
        elif event.key() in (Qt.Key.Key_Return,Qt.Key.Key_Enter):
            try:self.result_capture=crop_corners(self.pixmap,self.points,self.width(),self.height())
            except ValueError as exc:self.error=str(exc);self.update();return
            self.accept()
        else:super().keyPressEvent(event)


def capture_spectrogram(window):
    screen=window.screen();visible,state=window.isVisible(),window.windowState()
    window.hide()
    try:
        loop=QEventLoop();QTimer.singleShot(250,loop.quit);loop.exec()
        if sys.platform=='win32':
            import ctypes
            if ctypes.WinDLL('dwmapi').DwmFlush()!=0:raise RuntimeError('桌面尚未完成隐藏，请重试截图。')
        pixmap=screen.grabWindow(0)
        if pixmap.isNull():raise RuntimeError('截图失败。')
        dialog=CornerCapture(pixmap,screen.geometry())
        return dialog.result_capture if dialog.exec()==QDialog.DialogCode.Accepted else None
    finally:
        window.setWindowState(state)
        if visible:window.show();window.activateWindow()
