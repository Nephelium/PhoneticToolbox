"""Owned hidden Qt copy/resize checks; no devices, existing database or profile.

Native mode uses the Windows QPA backend with WA_DontShowOnScreen. This checks
the OS clipboard but cannot establish physical monitor/DPI/compositor behavior.
"""
import argparse
import json
import os
import time
from pathlib import Path
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-hidden',action='store_true')
    parser.add_argument('--baseline-copy',action='store_true')
    parser.add_argument('--cycles',type=int,default=20)
    args=parser.parse_args()
    os.environ['QT_QPA_PLATFORM']='windows' if args.native_hidden else 'offscreen'
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')
    from PyQt6.QtCore import QEventLoop,QTimer,QMimeData,Qt,QPoint,qVersion
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtWebEngineCore import QWebEngineSettings
    from ptb_desktop.host import Workbench,register_scheme
    out=ROOT/'output/validation/desktop-r2'/('qt-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:6])
    out.mkdir(parents=True)
    register_scheme();app=QApplication(['PTB-desktop-r2-owned-test'])
    window=Workbench(ROOT/'frontend/dist',test=True,vocal_profile=out/'vocal',start_module='M17')
    if args.native_hidden:window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen,True)
    window.show();window.resize(1366,768)
    report={'success':False,'platform':app.platformName(),'qt':qVersion(),'flags':os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS'),
        'scope':'owned hidden/offscreen Qt with explicit client geometry on maximize; physical titlebar double-click, monitor DPI and Windows compositor remain unverified',
        'terminations':[],'checks':[],'transitions':[]}
    window.page.renderProcessTerminated.connect(lambda status,code:report['terminations'].append({'status':status.name,'code':code}))
    clipboard=app.clipboard();backup=QMimeData();mime=clipboard.mimeData()
    if mime:
        for fmt in mime.formats():backup.setData(fmt,mime.data(fmt))
    wrote_clipboard=False;last_written=None
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();values=[]
        window.page.runJavaScript(code,lambda value:(values.append(value),loop.quit()))
        QTimer.singleShot(5000,loop.quit);loop.exec()
        if not values:raise AssertionError('JavaScript response timed out')
        return values[0]
    def until(code,seconds=30):
        end=time.perf_counter()+seconds
        while time.perf_counter()<end:
            if js(code):return
            pause()
        raise AssertionError(code)
    def capture(name):
        shot=window.view.grab().toImage();assert not shot.isNull()
        pixels=[shot.pixelColor(x,y) for x in range(0,shot.width(),max(1,shot.width()//40)) for y in range(0,shot.height(),max(1,shot.height()//30))]
        black=sum(max(p.red(),p.green(),p.blue())<12 for p in pixels)/len(pixels)
        colors=len({p.rgb() for p in pixels})
        if name:shot.save(str(out/(name+'.png')))
        return {'black_fraction':black,'sample_colors':colors,'content_visible':black<.95 and colors>10}
    try:
        until('document.querySelector(".m17-body")?.dataset.loaded==="true" && document.querySelector(".m17-body")?.dataset.fontReady==="true"')
        report['defaults']={'page_background':window.page.backgroundColor().name(),
            'page_background_alpha':window.page.backgroundColor().alpha(),
            'webgl':window.page.settings().testAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled),
            'javascript_clipboard':window.page.settings().testAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard),
            'javascript_paste':window.page.settings().testAttribute(QWebEngineSettings.WebAttribute.JavascriptCanPaste),
            'renderer_pid':window.page.renderProcessPid()}
        # Never print or save clipboard contents, only equality/length evidence.
        value='中文 ḁ e\u0301 é 𝼆 V𐞀 t͡s\nʔ\tɧ'
        js('(()=>{const e=document.querySelector(".m17-editor");e.value='+json.dumps(value)+';e.dispatchEvent(new Event("input",{bubbles:true}));})()')
        location=js('(()=>{const b=[...document.querySelectorAll("button")].find(e=>e.textContent.trim()==="复制全部").getBoundingClientRect();return {x:b.x+b.width/2,y:b.y+b.height/2}})()')
        QTest.mouseClick(window.view.focusProxy() or window.view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(location['x']),round(location['y'])))
        pause(500)
        result=clipboard.text()==value
        if result:wrote_clipboard=True;last_written=value
        report['copy']={'ui_button':'QTest.mouseClick on owned view','unicode_equal':result,'length':len(value),'notice':js('document.querySelector(".m17-body")?.innerText.includes("已复制全部文字")')}
        report['copy']['browser_clipboard_api']=js('typeof navigator.clipboard?.writeText')
        report['copy']['permission_events']=list(window.m05_media.events)
        assert result!=args.baseline_copy,report['copy']
        if not args.baseline_copy:
            assert report['copy']['notice']
            for text in ['',value*1000]:
                response=json.loads(window.bridge.writeClipboard(text));assert response=={'ok':True}
                assert clipboard.text()==text
                wrote_clipboard=True;last_written=text
            before=clipboard.text()
            rejected=json.loads(window.bridge.writeClipboard('a'*2_000_001))
            assert not rejected['ok'] and clipboard.text()==before
            report['checks'].append('UI copy, empty text, combining/decomposed/non-BMP/multiline/large text round trips; oversized write rejects without clearing clipboard')
        else:report['checks'].append('baseline old built M17 copy button fails actual clipboard equality')
        # rAF heartbeat and WebGL context are owned test-only observability.
        js('window.__qaFrames=0;window.__qaResizes=0;addEventListener("resize",()=>window.__qaResizes++);window.__qaTick=()=>{window.__qaFrames++;requestAnimationFrame(window.__qaTick)};window.__qaTick();window.__qaGlCanvas=document.createElement("canvas");window.__qaGl=window.__qaGlCanvas.getContext("webgl");')
        report['webgl_context']=js('!!window.__qaGl && !window.__qaGl.isContextLost()')
        initial_pid=window.page.renderProcessPid();previous_frames=js('window.__qaFrames')
        for cycle in range(args.cycles):
            for state in ('maximized','normal'):
                started=time.perf_counter()
                if state=='maximized':
                    window.showMaximized()
                    # Hidden QPA windows do not receive the normal WM geometry.
                    # Exercise the same client resize explicitly, without claiming
                    # this covers native titlebar/DWM presentation.
                    window.resize(window.screen().availableGeometry().size())
                else:window.showNormal();window.resize(1366+(cycle%3)*100,768+(cycle%2)*80)
                samples=[]
                for target_ms in (0,16,50,150,500):
                    remaining=target_ms-(time.perf_counter()-started)*1000
                    if remaining>0:pause(max(1,round(remaining)))
                    before=time.perf_counter()
                    pixels=capture(f'{cycle}-{state}-{target_ms}ms' if cycle in (0,args.cycles-1) else None)
                    js_start=time.perf_counter()
                    live=js('({frames:window.__qaFrames,resizes:window.__qaResizes,width:innerWidth,height:innerHeight})')
                    samples.append({'target_ms':target_ms,'actual_ms':(before-started)*1000,'js_response_ms':(time.perf_counter()-js_start)*1000,**pixels,**live})
                report['transitions'].append({'cycle':cycle,'state':state,'qt_maximized':window.isMaximized(),'samples':samples})
                until('innerWidth>0 && innerHeight>0 && window.__qaFrames>'+str(previous_frames),5)
                metrics=js('({width:innerWidth,height:innerHeight,frames:window.__qaFrames,editor:!!document.querySelector(".m17-editor"),glLost:window.__qaGl?window.__qaGl.isContextLost():null})')
                previous_frames=metrics['frames']
                assert metrics['editor'] and metrics['glLost'] is not True
                assert window.page.renderProcessPid()==initial_pid
                report['transitions'][-1].update({'seconds':time.perf_counter()-started,**metrics})
                assert all(s['content_visible'] for s in samples),report['transitions'][-1]
        assert not report['terminations']
        report['checks'].append('maximize/restore loops retain renderer PID, JS/rAF, live DOM, available WebGL context and nonblack screenshots')
        report['success']=True
    finally:
        # Restore all MIME formats only while clipboard still holds our write.
        # If the user copied something concurrently, leave their new data alone.
        if wrote_clipboard and clipboard.text()==last_written:
            originals={fmt:bytes(backup.data(fmt)) for fmt in backup.formats()}
            clipboard.setMimeData(backup)
            restored=clipboard.mimeData()
            report['clipboard_restored']=all(bytes(restored.data(fmt))==value for fmt,value in originals.items())
        else:report['clipboard_restored']=not wrote_clipboard
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
        window.closing=True;window.close();window.page.deleteLater();app.processEvents()
        print(json.dumps({'success':report['success'],'output':str(out),'checks':len(report['checks'])},ensure_ascii=False))


if __name__=='__main__':main()
