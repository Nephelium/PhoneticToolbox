"""M09 coordinate/error cleanup structure, without reading the desktop."""
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QPointF,Qt
from ptb_desktop import m09_capture


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('ratio',[1,1.25,1.5,2])
def test_crop_uses_physical_pixels(app,ratio):
    pixmap=QPixmap(1000,600);pixmap.setDevicePixelRatio(ratio)
    points=[QPointF(x/ratio,y/ratio) for x,y in [(100,100),(800,80),(850,500),(120,520)]]
    cropped,corners=m09_capture.crop_corners(pixmap,points,1000/ratio,600/ratio)
    assert (cropped.width(),cropped.height())==(750,440)
    assert corners[0]['y']==pytest.approx(20/440)


@pytest.mark.parametrize('points',[[(0,0),(100,100),(100,0),(0,100)],[(0,0),(0,0),(0,0),(0,0)],[(0,0)]])
def test_invalid_quad_rejected(app,points):
    with pytest.raises(ValueError):m09_capture.crop_corners(QPixmap(100,100),[QPointF(*p) for p in points],100,100)


@pytest.mark.parametrize('state',[Qt.WindowState.WindowNoState,Qt.WindowState.WindowMaximized,Qt.WindowState.WindowMinimized])
def test_failed_grab_restores_state_without_screen_read(app,monkeypatch,state):
    class Window:
        visible=True
        def screen(self):return self
        def isVisible(self):return self.visible
        def windowState(self):return state
        def hide(self):self.visible=False
        def grabWindow(self,id):return QPixmap()
        def setWindowState(self,value):self.restored=value
        def show(self):self.visible=True
        def activateWindow(self):self.activated=True
    window=Window()
    # Avoid platform compositor access; only exercise error/finally control flow.
    monkeypatch.setattr(m09_capture.sys,'platform','test')
    with pytest.raises(RuntimeError,match='截图失败'):m09_capture.capture_spectrogram(window)
    assert window.visible and window.restored==state and window.activated
