"""Keep the launcher card until the workbench, fonts and two frames are ready."""
from PyQt6.QtCore import QObject,QTimer
from .startup_ready import signal_ready


class StartupPresentation(QObject):
    def __init__(self,view,notify=signal_ready):
        super().__init__(view)
        self.view,self.notify,self.done,self.pending=view,notify,False,False
        self.generation=0
        self.timer=QTimer(self);self.timer.setInterval(50)
        self.timer.timeout.connect(self.check)
        view.loadStarted.connect(self.loading)
        view.loadFinished.connect(self.loaded)
        view.destroyed.connect(self.cancel)

    def loading(self):
        self.generation+=1;self.timer.stop();self.pending=False

    def loaded(self,ok):
        if ok and not self.done:self.timer.start();self.check()

    def check(self):
        if self.done or self.pending or self.view.page().isLoading():return
        self.pending=True
        generation=self.generation
        self.view.page().runJavaScript("""(()=>{
            if(document.readyState!=='complete' || !document.querySelector('.app-shell') ||
               document.fonts.status!=='loaded' || !window.qt)return false;
            const key='__ptbStartupFrames';
            if(!window[key]){
                const state=window[key]={ready:false};
                requestAnimationFrame(()=>requestAnimationFrame(()=>{state.ready=true;}));
            }
            return window[key].ready;
        })()""",lambda ready:self.checked(generation,ready))

    def checked(self,generation,ready):
        if generation!=self.generation:return
        self.pending=False
        if self.done or not ready or self.view.page().isLoading():return
        self.done=True;self.timer.stop();self.notify()

    def cancel(self,*args):
        self.done=True;self.generation+=1;self.timer.stop()
