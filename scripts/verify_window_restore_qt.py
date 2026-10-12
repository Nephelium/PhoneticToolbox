"""Actual Windows minimize/restore transitions, pixels and retained page state."""
import ctypes
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PARTS=('scripts','desktop/src','backend/src','packages/phonetic_core/src')
for part in PARTS:sys.path.insert(0,str(ROOT/part))
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/part) for part in PARTS)
os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--mute-audio'


def main():
    from ctypes.wintypes import HWND
    from PyQt6.QtCore import QEventLoop,QTimer,Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench,register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files
    from verification_artifacts import verification_run
    from capture_native_window import capture_window

    with verification_run('window-restore','qt','Windows native ShowWindow minimize/restore; isolated existing-schema DB/profile; no camera or microphone') as (out,scratch):
        db=scratch/'jobs.sqlite3'
        with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
        cache=scratch/'cache';cache.mkdir();initialize_local_files(cache)
        register_scheme();app=QApplication(['owned-window-restore'])
        window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=scratch/'vocal')
        window.move(30,30);window.show()
        report=dict(success=False,cases=[],terminations=[],buttonChecks=[])
        window.page.renderProcessTerminated.connect(lambda status,code:report['terminations'].append([status.name,code]))
        user=ctypes.WinDLL('user32');show=user.ShowWindow;show.argtypes=[HWND,ctypes.c_int]
        def pause(ms=200):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            loop,values=QEventLoop(),[]
            window.page.runJavaScript(code,lambda v:(values.append(v),loop.quit()));QTimer.singleShot(8000,loop.quit);loop.exec()
            assert values,'JavaScript timeout';return values[0]
        def until(code):
            deadline=time.monotonic()+40
            while time.monotonic()<deadline:
                if js(code):return
                pause()
            raise AssertionError(code)
        def click(label):
            js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")||b.textContent.trim())==='+json.dumps(label)+')?.click()');pause()
        def visible_frame(name):
            picture=capture_window(int(window.winId()))
            samples=[picture.getpixel((x,y))[:3] for x in range(15,picture.width-15,max(1,picture.width//30)) for y in range(15,picture.height-15,max(1,picture.height//20))]
            fraction=sum(max(c)>=12 for c in samples)/len(samples)
            picture.save(out/(name+'.png'))
            return fraction
        try:
            until("!!document.querySelector('.window-controls')&&document.fonts.status==='loaded'")
            js("window.__restoreVerification=Math.random().toString(36)")
            for mode in ('light','dark'):
                click('设置');click('外观');click('浅色' if mode=='light' else '深色');click('首页')
                window.showNormal();pause(250)
                # Keyboard activation also exercises Chromium's focus-visible outline.
                js("document.querySelector('[aria-label=最大化窗口]').focus()")
                QTest.keyClick(window.view.focusProxy(),Qt.Key.Key_Return);pause(1200)
                until("!!document.querySelector('[aria-label=还原窗口]')")
                assert window.isMaximized()
                button=js("(()=>{const b=document.querySelector('[aria-label=还原窗口]');return {outline:getComputedStyle(b).outlineStyle,shadow:getComputedStyle(b).boxShadow};})()")
                assert button==dict(outline='none',shadow='none'),button
                area=window.screen().availableGeometry();frame=window.frameGeometry()
                assert abs(frame.width()-area.width())<=2 and abs(frame.height()-area.height())<=2,(frame,area)
                report['buttonChecks'].append(dict(mode=mode,**button))
                for module in ('首页','参数显示','唇形提取'):
                    click(module)
                    if module=='唇形提取':pause(5000)
                    for maximized in (False,True):
                        if maximized:window.showMaximized()
                        else:window.showNormal()
                        pause(1200)
                        for cycle in range(3):
                            name=str(len(report['cases']))+'-'+str(maximized)+'-'+str(cycle)
                            baseline=visible_frame(name+'-before');assert baseline>.7,baseline
                            before=js("({tab:document.querySelector('[role=tab][aria-selected=true]')?.id,w:innerWidth,h:innerHeight,documentToken:window.__restoreVerification})")
                            show(HWND(int(window.winId())),6);pause((100,600,2500)[cycle]);assert window.isMinimized()
                            show(HWND(int(window.winId())),9);pause(1000)
                            after=js("({tab:document.querySelector('[role=tab][aria-selected=true]')?.id,w:innerWidth,h:innerHeight,documentToken:window.__restoreVerification})")
                            fraction=visible_frame(name+'-restored')
                            if fraction<=.7:
                                window._first_maximize.quick.grabFramebuffer().save(str(out/(name+'-quick.png')))
                                pause(3000)
                                report['lateFrame']=visible_frame(name+'-late')
                            row=dict(mode=mode,module=module,maximized=maximized,cycle=cycle,before=before,after=after,nonblack=fraction,quickVisible=window.view.focusProxy().isVisible(),exposed=window.windowHandle().isExposed(),quickExposed=window._first_maximize.quick.quickWindow().isExposed(),quickVisibility=window._first_maximize.quick.quickWindow().visibility().name,guardActive=window._first_maximize.active,windowState=int(window.windowState().value))
                            report['cases'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
                            assert not window.isMinimized() and window.isMaximized()==maximized
                            assert before==after,(before,after)
                            assert fraction>.7,row
                click('设置');click('字体')
                js("(()=>{const e=document.querySelector('input[aria-label=正文基础字号]');e.value='18';e.dispatchEvent(new Event('input',{bubbles:true}));})()")
                show(HWND(int(window.winId())),6);pause(600)
                show(HWND(int(window.winId())),9);pause(1000)
                assert js("document.querySelector('input[aria-label=正文基础字号]').value==='18'"),'Font draft lost on restore'
                assert visible_frame('font-draft-'+mode)>.7
                click('取消字体编辑')
            assert not report['terminations'],report['terminations'];report['success']=True
        finally:
            js("document.querySelector('#settings-tab-fonts')?.click()");pause()
            js("[...document.querySelectorAll('.font-settings button')].find(b=>b.textContent.trim()==='取消字体编辑')?.click()");pause()
            window.close();pause(350);report['closed']=not window.isVisible()
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8');app.quit()
            print(json.dumps(dict(success=report['success'],report=str(out/'report.json')),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
