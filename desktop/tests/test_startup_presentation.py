import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PyQt6.QtCore import QObject,pyqtSignal
from PyQt6.QtWidgets import QApplication
from ptb_desktop.startup_presentation import StartupPresentation


class View(QObject):
    loadStarted=pyqtSignal()
    loadFinished=pyqtSignal(bool)
    def __init__(self):
        super().__init__();self.loading=False;self.callbacks=[]
    def page(self):return self
    def isLoading(self):return self.loading
    def runJavaScript(self,code,callback):self.callbacks.append(callback)


def test_delayed_page_and_font_readiness_keeps_the_card_until_ready():
    app=QApplication.instance() or QApplication([])
    view=View();signals=[];guard=StartupPresentation(view,lambda:signals.append('ready'))
    try:
        view.loadFinished.emit(True);view.callbacks.pop()(False)
        assert not guard.done and signals==[]
        guard.check();view.callbacks.pop()(True)
        assert guard.done and signals==['ready']
        guard.check();assert signals==['ready']
    finally:guard.cancel()


def test_reload_and_cancel_invalidate_late_callbacks():
    app=QApplication.instance() or QApplication([])
    view=View();signals=[];guard=StartupPresentation(view,lambda:signals.append('ready'))
    view.loadFinished.emit(True);old=view.callbacks.pop()
    view.loadStarted.emit();old(True);assert signals==[]
    view.loadFinished.emit(True);latest=view.callbacks.pop()
    guard.cancel();latest(True);assert signals==[]
