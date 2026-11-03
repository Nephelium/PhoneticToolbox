"""Exercise actual Chromium fullscreen requests and native host restoration."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from verification_artifacts import verification_run
from manual.capture_a_support import Capture


def run(args,out,scratch):
    from PyQt6.QtCore import QEvent,Qt,QPoint
    from PyQt6.QtTest import QTest
    c=Capture(args,out,scratch,'m13','mandarin-ipa');events=[]
    try:
        c.until('!!document.querySelector(".app-shell")')
        c.ui.prepare_qt(c.js,c.until)
        c.window.page.fullScreenRequested.connect(lambda request:events.append({'at':time.monotonic(),'request':request.toggleOn(),'state':c.window.windowState().value,'previous':getattr(c.window,'_reader_previous_state',Qt.WindowState.WindowNoState).value}))
        c.js('(()=>{const button=document.createElement("button");button.id="owned-fullscreen-check";button.textContent="Fullscreen";Object.assign(button.style,{position:"fixed",left:"30px",top:"80px",zIndex:999999});button.onclick=()=>document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();document.body.append(button)})()')
        def pointer():
            point=c.js('(()=>{const r=document.querySelector("#owned-fullscreen-check").getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
            target=c.window.view.focusProxy() or c.window.view
            position=target.mapFromGlobal(c.window.view.mapToGlobal(QPoint(round(point[0]),round(point[1]))))
            QTest.mouseClick(target,Qt.MouseButton.LeftButton,pos=position);c.pause(150)
        results=[]
        for maximized in (True,False,True):
            if maximized:c.window.showMaximized()
            else:c.window.showNormal();c.window.resize(1200,800)
            c.pause(400);before=[c.window.width(),c.window.height()]
            assert c.window.isMaximized()==maximized
            pointer();c.until('!!document.fullscreenElement')
            assert c.window.isFullScreen()
            pointer();c.until('!document.fullscreenElement')
            states=[]
            # Check eventual state and continued stability after Chromium exits.
            for _ in range(15):
                c.pause(100);states.append({'maximized':c.window.isMaximized(),'fullscreen':c.window.isFullScreen(),'state':c.window.windowState().value})
            assert all(s['maximized']==maximized and not s['fullscreen'] for s in states[5:]),states
            if not maximized:assert [c.window.width(),c.window.height()]==before
            results.append({'maximizedBefore':maximized,'sizeBefore':before,'statesAfter':states})
        c.report.update(success=True,checks=results)
    finally:
        c.report['fullscreenEvents']=events
        c.report['hostSha256']=hashlib.sha256((ROOT/'desktop/src/ptb_desktop/host.py').read_bytes()).hexdigest()
        c.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--locale',choices=('en','ja'),default='en');parser.add_argument('--frontend-dist',type=Path,default=ROOT/'output/validation/app-i18n-root/m11-capture-build');args=parser.parse_args();args.frontend_dist=args.frontend_dist.resolve(strict=True)
    with verification_run('reader-fullscreen',args.locale,'Actual isolated native Windows host plus pointer-activated Chromium fullscreen; normal/maximized restoration and stability') as (out,scratch):run(args,out,scratch)
