"""Actual Qt screenshot flow over an owned real-recording spectrogram window."""
import argparse,json,sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication,QWidget
from PyQt6.QtCore import Qt,QTimer,QPoint,QPointF
from PyQt6.QtGui import QPainter,QPixmap,QColor
from PyQt6.QtTest import QTest
from ptb_desktop.m09_capture import CornerCapture,capture_spectrogram,crop_corners

p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('output',type=Path);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
app=QApplication([]);image=QPixmap(str(args.image.resolve()));assert not image.isNull()
class Background(QWidget):
    def paintEvent(self,event):
        painter=QPainter(self);painter.fillRect(self.rect(),QColor('white'));painter.drawPixmap(100,100,self.width()-200,self.height()-200,image)
background=Background(None,Qt.WindowType.FramelessWindowHint|Qt.WindowType.WindowStaysOnTopHint);background.setGeometry(app.primaryScreen().geometry());background.showFullScreen()
toolbox=QWidget();toolbox.setWindowTitle('P17 M09 owned test toolbox');toolbox.resize(400,300);toolbox.show();QTest.qWait(500)
records=[]
def interact(cancel=False):
    dialog=app.activeModalWidget()
    if not isinstance(dialog,CornerCapture):QTimer.singleShot(20,lambda:interact(cancel));return
    assert not toolbox.isVisible()
    if cancel:QTest.keyClick(dialog,Qt.Key.Key_Escape);return
    w,h=dialog.width(),dialog.height()
    # Invalid crossing order must remain open and allow R to recover.
    for x,y in [(100,100),(w-100,h-100),(w-100,100),(100,h-100)]:QTest.mouseClick(dialog,Qt.MouseButton.LeftButton,pos=QPoint(x,y))
    QTest.keyClick(dialog,Qt.Key.Key_Return);assert dialog.error and dialog.isVisible()
    QTest.keyClick(dialog,Qt.Key.Key_R);assert len(dialog.points)==0
    points=[(110,110),(w-110,110),(w-110,h-110),(110,h-110)]
    for x,y in points:QTest.mouseClick(dialog,Qt.MouseButton.LeftButton,pos=QPoint(x,y))
    QTest.keyClick(dialog,Qt.Key.Key_Backspace);assert len(dialog.points)==3
    QTest.mouseClick(dialog,Qt.MouseButton.RightButton,pos=QPoint(110,h-110));assert len(dialog.points)==2
    for x,y in points[2:]:QTest.mouseClick(dialog,Qt.MouseButton.LeftButton,pos=QPoint(x,y))
    assert dialog.isVisible() and len(dialog.points)==4
    QTest.keyClick(dialog,Qt.Key.Key_Return)
try:
    for mode,cancel in [('normal',False),('maximized',True),('minimized',True)]:
        if mode=='normal':toolbox.showNormal()
        elif mode=='maximized':toolbox.showMaximized()
        else:toolbox.showMinimized()
        import ctypes
        ctypes.windll.user32.ShowWindow(int(background.winId()),5);ctypes.windll.user32.SetWindowPos(int(background.winId()),-1,0,0,0,0,0x0001|0x0002|0x0040)
        background.raise_();background.activateWindow();QTest.qWait(800);print('native',app.platformName(),background.isVisible(),int(background.winId()),ctypes.windll.user32.IsWindowVisible(int(background.winId())),ctypes.windll.user32.GetForegroundWindow(),background.geometry().getRect(),toolbox.screen().name(),flush=True);before=toolbox.windowState();visible=toolbox.isVisible();QTimer.singleShot(300,lambda c=cancel:interact(c));result=capture_spectrogram(toolbox)
        assert toolbox.windowState()==before and toolbox.isVisible()==visible
        if cancel:assert result is None
        else:
            pix,corners=result;assert all(abs(c['x']-x)<=1/pix.width()+1e-9 and abs(c['y']-y)<=1/pix.height()+1e-9 for c,(x,y) in zip(corners,[(0,0),(1,0),(1,1),(0,1)]));img=pix.toImage();assert all(abs(img.pixelColor(x*img.width()//11,y*img.height()//11).red()-img.pixelColor(x*img.width()//11,y*img.height()//11).blue())<=1 for x in range(1,11) for y in range(1,11)), 'Captured background must be the owned grayscale spectrogram';pix.save(str(args.output/'accepted-real-spectrogram.png'))
        records.append(dict(mode=mode,cancel=cancel,restored=True))
    # Physical-pixel mapping is tested independently of the actual monitor DPR.
    for ratio in [1,1.25,1.5,2]:
        pm=QPixmap(1000,600);pm.setDevicePixelRatio(ratio)
        cropped,corners=crop_corners(pm,[QPointF(100/ratio,100/ratio),QPointF(800/ratio,80/ratio),QPointF(850/ratio,500/ratio),QPointF(120/ratio,520/ratio)],1000/ratio,600/ratio)
        assert cropped.width()==750 and cropped.height()==440 and abs(corners[0]['y']-20/440)<1e-12
    report=dict(success=True,native_qscreen_dpr=app.primaryScreen().devicePixelRatio(),screen_geometry=[background.width(),background.height()],source=str(args.image),checks=records,mapping_ratios=[1,1.25,1.5,2],invalid_quad_undo_reset_enter=True)
    (args.output/'capture-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False))
finally:toolbox.close();background.close()
