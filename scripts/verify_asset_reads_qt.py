"""Verify responsive real Qt pages while an owned resource read waits six seconds."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import threading
import time

from workbench_source import configure
from verification_artifacts import verification_run,write_json


def main():
    root=Path(__file__).resolve().parents[1]
    configure(root)
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--mute-audio'
    from PyQt6.QtCore import QEventLoop,QThread,QTimer
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench,register_scheme
    from ptb_desktop.feedback_manager import FeedbackManager
    from ptb_desktop import web_assets

    register_scheme()
    app=QApplication(['Owned asynchronous resource QA'])
    with verification_run('desktop-assets','qt','Current source, visible default GPU, real manual m16 and IPA, one 6s worker read; independent native WM_NULL monitor; isolated manager with no credentials') as (out,scratch):
        report=dict(success=False,resources=[],native=[],gui=[],actions=[])
        stop,armed,entered=threading.Event(),threading.Event(),threading.Event()
        original=web_assets._read_asset
        phase='baseline';window=None;manager=None
        def read(folder,name):
            injected=armed.is_set() and name=='manual/chapters/m16.json'
            if injected:armed.clear();entered.set()
            start=time.perf_counter()
            if injected:time.sleep(6)
            try:return original(folder,name)
            finally:report['resources'].append(dict(path=name,artificial=injected,
                gui_thread=QThread.currentThread()==app.thread(),ms=round((time.perf_counter()-start)*1000,3)))
        def pause(ms=50):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code,page=None):
            loop,values=QEventLoop(),[]
            (page or window.page).runJavaScript(code,lambda value:(values.append(value),loop.quit()))
            QTimer.singleShot(1500,loop.quit);loop.exec()
            if not values:raise RuntimeError('Qt JavaScript callback did not respond')
            return values[0]
        def until(code,page=None):
            end=time.monotonic()+25
            while not js(code,page):
                if time.monotonic()>end:raise RuntimeError(code)
                pause()
        def click(selector):
            if not js('(()=>{const e=document.querySelector('+json.dumps(selector)+');if(!e)return false;e.click();return true})()'):
                raise RuntimeError(selector)
        def system(label,chart):
            if not js('(()=>{const e=[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(label)+');if(!e)return false;e.click();return true})()'):
                raise RuntimeError(label)
            until('!!document.querySelector("[data-chart='+chart+']")')
        web_assets._read_asset=read
        try:
            window=Workbench(root/'frontend/dist',test=True,vocal_profile=scratch/'vocal',start_module='M17')
            window.show()
            report['dpi']=window.devicePixelRatioF()
            report['flags']=os.environ['QTWEBENGINE_CHROMIUM_FLAGS']
            report['render_terminations']=[]
            window.page.renderProcessTerminated.connect(lambda state,code:report['render_terminations'].append([str(state),code]))
            handle=int(window.winId());user=ctypes.WinDLL('user32')
            user.SendMessageTimeoutW.argtypes=[wintypes.HWND,wintypes.UINT,wintypes.WPARAM,
                wintypes.LPARAM,wintypes.UINT,wintypes.UINT,ctypes.POINTER(ctypes.c_size_t)]
            user.SendMessageTimeoutW.restype=ctypes.c_size_t
            def monitor():
                while not stop.is_set():
                    begin=time.perf_counter();result=ctypes.c_size_t()
                    ok=bool(user.SendMessageTimeoutW(handle,0,0,0,2,200,ctypes.byref(result)))
                    report['native'].append(dict(phase=phase,ms=round((time.perf_counter()-begin)*1000,3),ok=ok))
                    stop.wait(.1)
            thread=threading.Thread(target=monitor,daemon=True);thread.start()
            last=time.perf_counter()
            def tick():
                nonlocal last
                now=time.perf_counter();report['gui'].append(dict(phase=phase,ms=round((now-last)*1000,3)));last=now
            timer=QTimer(window);timer.timeout.connect(tick);timer.start(20)
            until('document.querySelector(".m17-body")?.dataset.fontReady==="true"')
            click('#tool-sidebar button[title="使用说明"]');until('!!document.querySelector(".manual-document")')
            click('.manual-toc [data-manual-chapter="m16"]')
            until('document.querySelector(".manual-document")?.dataset.manualChapter==="m16"')
            pause(300)
            loaded=[];window.view.loadFinished.connect(lambda ok:loaded.append(ok))
            window.view.reload()
            deadline=time.monotonic()+25
            while not loaded:
                if time.monotonic()>deadline:raise RuntimeError('Owned reload timeout')
                pause()
            assert loaded[-1]
            until('document.querySelector(".m17-body")?.dataset.fontReady==="true"')
            click('#tool-sidebar button[title="使用说明"]');until('!!document.querySelector(".manual-document")')
            phase='controlled_slow_resource';armed.set()
            click('.manual-toc [data-manual-chapter="m16"]')
            deadline=time.monotonic()+3
            while not entered.is_set():
                if time.monotonic()>deadline:raise RuntimeError('Controlled resource was not requested')
                pause(20)
            # These are completed before the slow chapter can return. Other
            # static chart/font resources can still be served by the same pool.
            start=time.perf_counter()
            click('#tab-M17');system('VoQS','voqs');system('extIPA','extipa');system('IPA','ipa')
            report['actions'].append(dict(action='three IPA charts while manual read waits',ms=round((time.perf_counter()-start)*1000,3)))
            assert not any(r['artificial'] for r in report['resources']),'Actions ran after, rather than during, the read'
            click('#tab-help')
            until('document.querySelector(".manual-document")?.dataset.manualChapter==="m16"')
            pause(200)
            delayed=[r for r in report['resources'] if r['artificial']]
            assert len(delayed)==1 and not delayed[0]['gui_thread']
            assert not any(r['gui_thread'] for r in report['resources'])
            controlled=[r for r in report['native'] if r['phase']=='controlled_slow_resource']
            assert controlled and all(r['ok'] for r in controlled),controlled
            report['controlled_native_timeouts']=sum(not r['ok'] for r in controlled)
            report['controlled_gui_max_ms']=max(r['ms'] for r in report['gui'] if r['phase']=='controlled_slow_resource')
            assert report['controlled_gui_max_ms']<1000,'The GUI stalled despite asynchronous file reads'
            visible_images='[...document.querySelectorAll(".manual-document img")].filter(e=>e.getBoundingClientRect().top<innerHeight&&e.getBoundingClientRect().bottom>0)'
            until('('+visible_images+').length>0&&('+visible_images+').every(e=>e.complete&&e.naturalWidth>0)')
            report['visible_images']=js('('+visible_images+').map(e=>({width:e.naturalWidth,height:e.naturalHeight,complete:e.complete}))')
            js('window.__assetFrames=false;requestAnimationFrame(()=>requestAnimationFrame(()=>window.__assetFrames=true));true')
            until('window.__assetFrames===true')
            pause(100)
            window.view.grab().save(str(out/'recording-manual-after.png'))

            # The same handler serves the independent manager's HTML entry and
            # WebChannel bootstrap. An empty owned identity sends no requests.
            phase='manager';manager=FeedbackManager(root/'frontend/dist',root=scratch/'manager',tray=False)
            manager.show()
            until('!!document.querySelector(".manager-connect")',manager.page)
            report['manager_connect_visible']=True
            manager.view.grab().save(str(out/'manager-after.png'))
            assert not report['render_terminations']

            # WebEngine may destroy a real request while its worker is waiting.
            # Closing its window/profile must neither wait for the worker nor
            # deliver a late result into the deleted request. The independent
            # manager continues to use the shared resource adapter meanwhile.
            phase='cancel_pending';armed.set();entered.clear()
            js('fetch("ptbapp://app/manual/chapters/m16.json?owned_cancel=1").catch(()=>{});true')
            deadline=time.monotonic()+3
            while not entered.is_set():
                if time.monotonic()>deadline:raise RuntimeError('Cancellation resource was not requested')
                pause(20)
            stop.set();thread.join(timeout=1);timer.stop();timer=None
            start=time.perf_counter()
            window.closing=True;window.close();window.page.deleteLater();pause(50)
            window.profile.deleteLater();pause(50);window.deleteLater();pause(50);window=None
            report['close_pending_ms']=round((time.perf_counter()-start)*1000,3)
            assert report['close_pending_ms']<1000
            assert sum(r['artificial'] for r in report['resources'])==1,'Close waited for the pending read'
            report['manager_responsive_after_close']=bool(js('!!document.querySelector(".manager-connect")',manager.page))
            assert report['manager_responsive_after_close']
            deadline=time.monotonic()+8
            while sum(r['artificial'] for r in report['resources'])<2:
                if time.monotonic()>deadline:raise RuntimeError('Cancelled read did not finish')
                pause()
            report['success']=True
        except Exception as error:
            report['error']=repr(error);raise
        finally:
            stop.set()
            if 'thread' in locals():thread.join(timeout=1)
            if 'timer' in locals() and timer is not None:timer.stop()
            if manager:
                manager.quitting=True;manager.close();manager.page.deleteLater();pause(100)
                manager.profile.deleteLater();pause(100);manager.deleteLater();pause(100)
            if window:
                window.closing=True;window.close();window.page.deleteLater();pause(100)
                window.profile.deleteLater();pause(100);window.deleteLater();pause(100)
            web_assets._read_pool().waitForDone(10000)
            web_assets._read_asset=original
            write_json(out/'report.json',report)
            print(json.dumps(dict(out=str(out),success=report['success'],actions=report['actions'],
                native_timeouts=report.get('controlled_native_timeouts'),gui_max_ms=report.get('controlled_gui_max_ms'),error=report.get('error')),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
