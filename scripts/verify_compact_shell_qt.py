"""Integrated Windows tabs, safe window controls and resizable reading sidebars."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARTS = ('scripts', 'desktop/src', 'backend/src', 'packages/phonetic_core/src')
for part in PARTS:sys.path.insert(0,str(ROOT/part))
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/part) for part in PARTS)
os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--mute-audio'


def main():
    from PyQt6.QtCore import QEventLoop,QPoint,QTimer,Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication,QMessageBox
    from ptb_desktop.host import Workbench,register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files
    from verification_artifacts import verification_run
    from capture_native_window import capture_window

    with verification_run('compact-shell','qt','Current production bundle; isolated existing-schema DB and profile; Windows integrated frame / reading sidebars') as (out,scratch):
        db=scratch/'jobs.sqlite3'
        with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
        cache=scratch/'cache';cache.mkdir();initialize_local_files(cache)
        register_scheme();app=QApplication(['compact-shell-owned-qa'])
        window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=scratch/'vocal',start_module='M02')
        window.move(30,30);window.show()
        report=dict(success=False,checks=[],cases=[],captures=[])
        def pause(ms=180):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            loop,results=QEventLoop(),[]
            window.page.runJavaScript(code,lambda value:(results.append(value),loop.quit()));QTimer.singleShot(8000,loop.quit);loop.exec()
            assert results,'JavaScript timeout'
            return results[0]
        def until(code):
            deadline=time.monotonic()+45
            while time.monotonic()<deadline:
                if js(code):return
                pause()
            raise AssertionError(code)
        def click(label):
            js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")||b.textContent.trim())==='+json.dumps(label)+')?.click()');pause()
        def capture(name):
            pause(350);capture_window(int(window.winId())).save(out/(name+'.png'));report['captures'].append(name)
            print('Captured '+name,flush=True)
        def drag(label,delta):
            selector='[role=separator][aria-label='+json.dumps(label,ensure_ascii=False)+']'
            until('!!document.querySelector('+json.dumps(selector)+')&&getComputedStyle(document.querySelector('+json.dumps(selector)+')).display!=="none"')
            point=js('(()=>{const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return [r.x+r.width/2,r.y+Math.min(60,r.height/2)];})()')
            target=window.view.focusProxy();start=QPoint(round(point[0]),round(point[1]))
            QTest.mousePress(target,Qt.MouseButton.LeftButton,pos=start)
            for i in range(1,7):QTest.mouseMove(target,start+QPoint(round(delta*i/6),0),delay=20)
            QTest.mouseRelease(target,Qt.MouseButton.LeftButton,pos=start+QPoint(delta,0));pause(250)
        def width(selector):return js('document.querySelector('+json.dumps(selector)+').getBoundingClientRect().width')
        try:
            until("!!document.querySelector('.window-controls')&&document.fonts.status==='loaded'")
            assert window.windowFlags()&Qt.WindowType.FramelessWindowHint
            chrome=js("(()=>{const a=document.querySelector('.topbar').getBoundingClientRect(),b=document.querySelector('.brand').getBoundingClientRect();return {top:a.top,brandTop:b.top,brands:document.querySelectorAll('#tool-sidebar .brand').length};})()")
            assert chrome==dict(top=0,brandTop=0,brands=1),chrome
            assert not js("document.body.textContent.includes('PhoneticToolbox 3.0 · 界面试用版')")
            assert not js("document.querySelector('.topbar')?.textContent.includes('开源与学术致谢')")
            click('首页');until("!!document.querySelector('.home-credits')")
            assert js("(()=>{const button=document.querySelector('.home-credits'),welcome=document.querySelector('.welcome');const b=button.getBoundingClientRect(),w=welcome.getBoundingClientRect();return button.closest('.home-page')!==null&&Math.abs(b.right-w.right)<2&&b.top<w.top+15;})()")
            click('开源与学术致谢');until("!!document.querySelector('dialog[open]')");click('关闭对话框')
            click('最大化窗口');until("!!document.querySelector('[aria-label=还原窗口]')");assert window.isMaximized()
            available=window.screen().availableGeometry();frame=window.frameGeometry()
            assert abs(frame.width()-available.width())<=2 and abs(frame.height()-available.height())<=2,(frame,available)
            report['maximized']=[frame.x(),frame.y(),frame.width(),frame.height()]
            click('还原窗口');pause();assert not window.isMaximized()
            click('最小化窗口');pause();assert window.isMinimized();window.showNormal();pause()
            window.showFullScreen();pause(350)
            screen=window.screen().geometry();frame=window.frameGeometry()
            assert abs(frame.width()-screen.width())<=2 and abs(frame.height()-screen.height())<=2,(frame,screen)
            window.showNormal();pause(350)
            # Real native border hit tests, scoped to this owned window only.
            import ctypes
            from ctypes.wintypes import HWND,RECT
            user=ctypes.WinDLL('user32');bounds=RECT()
            user.GetWindowRect.argtypes=[HWND,ctypes.POINTER(RECT)]
            user.GetWindowRect(HWND(int(window.winId())),ctypes.byref(bounds))
            send=user.SendMessageW;send.argtypes=[HWND,ctypes.c_uint,ctypes.c_size_t,ctypes.c_ssize_t];send.restype=ctypes.c_ssize_t
            for x,y,expected in [(bounds.left+2,bounds.top+2,13),(bounds.right-2,bounds.bottom-2,17),(bounds.left+2,(bounds.top+bounds.bottom)//2,10)]:
                assert send(HWND(int(window.winId())),0x84,0,(y&65535)<<16|(x&65535))==expected
            for mode in ('light','dark'):
                click('设置');click('外观');click('浅色' if mode=='light' else '深色');click('参数显示')
                rows=js("[...document.querySelectorAll('.tab-wrap')].map(e=>({active:e.classList.contains('active'),colours:[...e.querySelectorAll('button')].map(b=>getComputedStyle(b).color),background:getComputedStyle(e).backgroundColor,shadow:getComputedStyle(e).boxShadow}))")
                assert all(len(set(row['colours']))==1 for row in rows),rows
                assert all(row['shadow']=='none' for row in rows if row['active']),rows
                report['cases'].append(dict(mode=mode,tabs=rows));capture('tabs-'+mode)
            click('使用说明');until("!!document.querySelector('.manual-document')")
            before=width('.manual-navigation');drag('说明书目录宽度',70);assert width('.manual-navigation')>before+45
            first=js("document.querySelector('[data-manual-chapter]').getAttribute('data-manual-chapter')")
            js("document.querySelector('[data-manual-chapter="+first+"]').click()")
            until("!!document.querySelector('[data-manual-section]')")
            js("document.querySelector('[data-manual-section]').click()")
            until("!!document.querySelector('.manual-target-highlight')")
            assert js("getComputedStyle(document.querySelector('.manual-target-highlight')).outlineStyle==='none'")
            assert js("getComputedStyle(document.querySelector('.manual-toc button.active')).boxShadow==='none'")
            capture('manual-sidebar')
            manual_width=width('.manual-navigation')
            click('语音学论文精读');until("!!document.querySelector('.paper-body .paper-library')")
            before=width('.paper-library');drag('论文库宽度',65);assert width('.paper-library')>before+40
            paper_width=width('.paper-library');capture('paper-sidebar')
            click('使用说明');assert abs(width('.manual-navigation')-manual_width)<2
            click('语音学论文精读');assert abs(width('.paper-library')-paper_width)<2
            # A narrow window fits temporarily without replacing the saved preference.
            window.resize(850,650);pause(300);capture('narrow');window.resize(1440,900);pause(300)
            assert abs(width('.paper-library')-paper_width)<2
            # The integrated close button still asks about uncommitted font edits.
            click('设置');click('字体')
            js("(()=>{const e=document.querySelector('input[aria-label=正文基础字号]');e.value='18';e.dispatchEvent(new Event('input',{bubbles:true}));})()")
            seen=[]
            def reject_confirmation():
                for widget in app.topLevelWidgets():
                    if isinstance(widget,QMessageBox) and widget.isVisible() and widget.windowTitle()=='未保存的草稿':
                        seen.append(widget.text());widget.button(QMessageBox.StandardButton.No).click()
            confirmation=QTimer();confirmation.timeout.connect(reject_confirmation);confirmation.start(100)
            point=js("(()=>{const r=document.querySelector('[aria-label=关闭窗口]').getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2];})()")
            QTest.mouseClick(window.view.focusProxy(),Qt.MouseButton.LeftButton,pos=QPoint(round(point[0]),round(point[1])))
            deadline=time.monotonic()+8
            while not seen and time.monotonic()<deadline:pause(100)
            confirmation.stop();assert seen and window.isVisible(),seen
            click('取消字体编辑')
            report['checks']=['Integrated brand/tabs start at client top, one brand; credits at homepage content top right and absent from tab bar; old home line absent','Actual minimize/maximize/restore, work-area and full-screen bounds, native resize border hit tests','Two themes: whole-tab surface and identical label/close colour','Native Qt dragging of manual and paper sidebars; separate widths survive switching and temporary narrow window','Manual target has no outline, current TOC uses colour without ring','Trusted integrated close shows existing native unsaved confirmation; No preserves font draft and window']
            report['success']=True
        finally:
            js("document.querySelector('#settings-tab-fonts')?.click()");pause()
            js("[...document.querySelectorAll('.font-settings button')].find(b=>b.textContent.trim()==='取消字体编辑')?.click()");pause()
            window.close();pause(350);report['closed']=not window.isVisible()
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8');app.quit()
            print(json.dumps(dict(success=report['success'],report=str(out/'report.json')),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
