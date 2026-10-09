"""Cold native maximize checks in a private, owned Windows Qt window."""
import ctypes
import json
import os
import sys
import time
import traceback
from pathlib import Path


def verify(bundle, out, stage='ready', theme='light'):
    if stage not in {'shown', 'ready', 'delayed', 'restore', 'fallback', 'programmatic'} or theme not in {'light', 'dark'}:
        raise ValueError('Unknown presentation scenario')
    os.environ['QT_QPA_PLATFORM'] = 'windows'
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio')
    if '--disable-gpu' in os.environ['QTWEBENGINE_CHROMIUM_FLAGS']:
        raise ValueError('Default GPU is required')
    from PyQt6.QtCore import QEventLoop, QTimer, QPoint, Qt, qVersion, QObject, QEvent
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_workspace import prepare_workspace
    out.mkdir(parents=True, exist_ok=False)
    database, files = prepare_workspace(out/'state', bundle/'backend/migrations')
    register_scheme()
    app = QApplication(['P19-R18-owned-window-check'])
    app.setApplicationName('P19-R18-owned-QA')
    window = Workbench(bundle/'frontend/dist', test=True, jobs_path=database,
        local_files_root=files, vocal_profile=out/'vocal', vocal_resources=bundle/'resources/vocal_tract/native')
    window.setWindowTitle('PhoneticToolbox 3.0 · 窗口验证')
    window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    guard = window._first_maximize
    started = time.monotonic()
    report = dict(success=False, stage=stage, theme=theme, qt=qVersion(),
        mode='frozen' if getattr(sys, 'frozen', False) else 'source',
        flags=os.environ['QTWEBENGINE_CHROMIUM_FLAGS'], events=[], samples=[], terminations=[])
    def event(name, **kw):
        report['events'].append(dict(event=name, ms=round((time.monotonic()-started)*1000), **kw))
    window.page.renderProcessTerminated.connect(lambda status, code:report['terminations'].append([status.name, code]))
    first_resize = None
    delay_started = False
    class Events(QObject):
        def eventFilter(self, watched, evt):
            nonlocal first_resize, delay_started
            if watched is window.view and evt.type() == QEvent.Type.Resize and (window.isMaximized() or guard.active):
                if first_resize is None:
                    first_resize = guard.cover.isVisible()
                event('maximized-resize', cover=guard.cover.isVisible(), size=[window.view.width(),window.view.height()])
            if watched is guard.cover and evt.type() in {QEvent.Type.Show, QEvent.Type.Hide}:
                event('cover-'+('show' if evt.type()==QEvent.Type.Show else 'hide'), frames=guard.frames,
                    size=[guard.cover.width(),guard.cover.height()], picture=[guard.cover.pixmap().width(),guard.cover.pixmap().height()])
                if evt.type()==QEvent.Type.Show and stage in {'delayed', 'restore'} and not delay_started:
                    delay_started = True
                    # Block the real Chromium main thread after the pre-resize
                    # snapshot, simulating a slow first resized compositor frame.
                    window.page.runJavaScript('(()=>{const end=performance.now()+900;while(performance.now()<end){};return true})()',
                        lambda result:event('renderer-delay-finished', ok=result))
            return False
    events = Events(window)
    window.view.installEventFilter(events)
    guard.cover.installEventFilter(events)
    user = ctypes.windll.user32
    user.WindowFromPoint.restype = ctypes.c_void_p
    user.GetAncestor.restype = ctypes.c_void_p
    user.PostMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
    class POINT(ctypes.Structure):
        _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]
    def pause(ms):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[]
        window.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()))
        QTimer.singleShot(6000,loop.quit);loop.exec()
        assert box, 'JavaScript timeout'
        return box[0]
    def until(code):
        end=time.monotonic()+20
        while time.monotonic()<end:
            if js(code):return
            pause(30)
        raise AssertionError(code)
    def command(value):
        assert user.PostMessageW(int(window.winId()),0x112,value,0)
    def sample(cycle, elapsed, save=False):
        value=dict(cycle=cycle,ms=round(elapsed),cover=guard.active,frames=guard.frames,captured=False)
        report['samples'].append(value)
        point=window.view.mapToGlobal(QPoint(0,0));ratio=window.devicePixelRatioF()
        # No fixed time exemption. Capture only when the entire rectangle is
        # owned by this test window, so OS animations/other windows are excluded.
        for x,y in [(8,8),(window.view.width()-8,8),(8,window.view.height()-8),
                    (window.view.width()-8,window.view.height()-8),(window.view.width()//2,window.view.height()//2)]:
            hwnd=user.WindowFromPoint(POINT(round((point.x()+x)*ratio),round((point.y()+y)*ratio)))
            if user.GetAncestor(ctypes.c_void_p(hwnd or 0),2)!=int(window.winId()):return
        image=window.screen().grabWindow(0,point.x(),point.y(),window.view.width(),window.view.height()).toImage()
        colors=[image.pixelColor(x,y) for x in range(0,image.width(),max(1,image.width()//32))
                for y in range(0,image.height(),max(1,image.height()//24))]
        value.update(captured=True,black_fraction=sum(max(c.red(),c.green(),c.blue())<12 for c in colors)/len(colors),
            colors=len({c.rgb() for c in colors}))
        assert value['black_fraction'] < .95, value
        if save:image.save(str(out/f'window-{cycle}.png'))
    try:
        window.show()
        pause(30)
        if stage!='shown':
            until('!!document.querySelector(".home-page")&&document.fonts.status==="loaded"')
            assert js('(()=>{const b=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==="设置");b.click();return true})()')
            assert js('(()=>{const b=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps('浅色' if theme=='light' else '深色')+');if(!b)return false;b.click();return true})()')
            pause(500)
            js('window.qaPresentationToken="same-page";window.qaPresentationGL=document.createElement("canvas").getContext("webgl");')
        renderer=window.page.renderProcessPid()
        ordinary=window.size()
        start=time.monotonic()
        if stage=='programmatic':window.showMaximized()
        elif stage=='fallback':
            # WM_SIZE fallback with no SC_MAXIMIZE, as used by OS snap paths.
            user.ShowWindow(ctypes.c_void_p(int(window.winId())),3)
        else:command(0xf030)
        if stage=='restore':
            pause(120)
            assert guard.active
            command(0xf120)
            pause(100)
            assert not window.isMaximized() and not guard.active
            pause(1000)
            command(0xf030)
            start=time.monotonic()
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            pause(20)
            sample(0,(time.monotonic()-start)*1000)
            if stage=='delayed' and (time.monotonic()-start)*1000<700:
                assert guard.active, 'An old viewport uncovered the delayed renderer'
            if window.isMaximized() and not guard.active and time.monotonic()-start>1.3:break
        assert window.isMaximized() and not guard.active and guard.frames>=2, 'First cover did not retire on valid resized frames'
        assert first_resize, 'The cover appeared after the first WebEngine Resize'
        assert any(v['captured'] for v in report['samples']), 'No owned screen sample'
        sample(0,(time.monotonic()-start)*1000,save=True)
        until('!!document.getElementById("app")?.childElementCount&&document.fonts.status==="loaded"')
        if stage=='shown':
            renderer=window.page.renderProcessPid()
            js('window.qaPresentationToken="same-page";window.qaPresentationGL=document.createElement("canvas").getContext("webgl");')
        shows=sum(e['event']=='cover-show' for e in report['events'])
        for cycle in range(1,4):
            command(0xf120);pause(200)
            assert not window.isMaximized() and not guard.active
            assert window.size()==ordinary, 'Restore lost the normal geometry'
            # Continuous normal resizes must not trigger screenshot protection.
            for offset in [20,40,0]:
                window.resize(ordinary.width()-offset,ordinary.height()-offset);pause(25)
                assert not guard.active
            start=time.monotonic();command(0xf030)
            for target in [20,80,160,320,600]:
                remaining=target-(time.monotonic()-start)*1000
                if remaining>0:pause(round(remaining))
                sample(cycle,(time.monotonic()-start)*1000,save=target==600)
            assert window.isMaximized() and not guard.active
            assert sum(e['event']=='cover-show' for e in report['events'])==shows
        assert window.page.renderProcessPid()==renderer
        assert js('qaPresentationToken==="same-page"&&!!qaPresentationGL&&!qaPresentationGL.isContextLost()')
        assert not report['terminations']
        report.update(success=True,pre_resize_cover=first_resize,cover_shows=shows,
            renderer_preserved=True,page_preserved=True,webgl_preserved=True,
            actual_color_scheme=js('getComputedStyle(document.documentElement).colorScheme'))
    except Exception:
        report['error']=traceback.format_exc()
    finally:
        window.closing=True;window.close();window.page.deleteLater();app.processEvents()
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return 0 if report['success'] else 1


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',default='ready')
    parser.add_argument('--theme',default='light')
    args=parser.parse_args()
    from v3_local_preview_entry import configure
    raise SystemExit(verify(configure(),args.out.absolute(),args.stage,args.theme))
