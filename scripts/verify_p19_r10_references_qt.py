"""Reference UI through native Qt/QWebChannel, without changing OS clipboard or opening a browser."""
import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from uuid import uuid4


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',action='store_true');args=parser.parse_args()
    os.environ['QT_QPA_PLATFORM']='windows'
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')
    from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint,qVersion
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop import host
    from ptb_worker.local_acoustic_files import initialize_local_files
    root=Path(__file__).resolve().parents[1];out=root/'output/validation/p19-r10'/('qt-'+('baseline-' if args.baseline else '')+uuid4().hex)
    out.mkdir(parents=True);db=out/'jobs.sqlite3'
    with sqlite3.connect((root/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone();source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    class Clipboard:
        value='unchanged';writes=[]
        def setText(self,value):self.value=value;self.writes.append(value)
        def text(self):return self.value
    clipboard=Clipboard();opened=[];requests=[]
    old_clipboard=host.QApplication.clipboard;old_open=host.QDesktopServices.openUrl
    host.QApplication.clipboard=lambda:clipboard
    host.QDesktopServices.openUrl=lambda url:(opened.append(url.toString()) or True)
    host.register_scheme();app=QApplication(['P19-R10-owned-QA'])
    w=host.Workbench(root/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.resize(1440,900);w.show()
    w.page.newWindowRequested.connect(lambda r:requests.append({'url':r.requestedUrl().toString(),'user':r.isUserInitiated()}))
    report={'success':False,'baseline':args.baseline,'qt':qVersion(),'checks':[],
            'scope':'Windows hidden native source host; clipboard and OS browser launch endpoints replaced with recording adapters; EXE not exercised'}
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda v:(box.append(v),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        assert box,'JavaScript timeout';return box[0]
    def until(code):
        end=time.monotonic()+35
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code)
    def button(text):
        assert js('(()=>{const b=[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(text)+');if(!b)return false;b.click();return true})()');pause()
    def native_click(expression,modifier=Qt.KeyboardModifier.NoModifier,key=False):
        point=js('(()=>{const e='+expression+';if(!e)return null;e.scrollIntoView({block:"center"});e.focus();const r=e.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
        assert point,expression;pause()
        receiver=w.view.focusProxy() or w.view
        if key:QTest.keyClick(receiver,Qt.Key.Key_Return,modifier)
        else:QTest.mouseClick(receiver,Qt.MouseButton.LeftButton,modifier,QPoint(round(point[0]),round(point[1])))
        pause(300)
    def query(text):
        js('(()=>{const e=document.querySelector("input[aria-label=搜索来源]");e.value='+json.dumps(text)+';e.dispatchEvent(new Event("input",{bubbles:true}))})()');pause()
    try:
        until('!!document.querySelector(".home-page")&&document.fonts.status==="loaded"')
        button('开源与学术致谢');query('Klatt formant synthesizer')
        until('!!document.querySelector(".reference-row")')
        citation=js('document.querySelector(".reference-row .selectable").textContent')
        native_click('document.querySelector(".reference-links button")')
        status=js('document.querySelector("input[aria-label=搜索来源]").nextElementSibling.textContent')
        native_click('document.querySelector(".reference-links a")')
        if args.baseline:
            assert not clipboard.writes and '剪贴板' in status,status
            assert not opened and requests,requests
            report['checks'].append('reproduced citation Clipboard API failure and unhandled user-initiated target=_blank link')
            report['copy_status']=status
        else:
            assert clipboard.writes==[citation] and status=='引用已复制',(clipboard.writes,status)
            assert len(opened)==1 and opened[0]==requests[0]['url'],(opened,requests)
            assert js('document.querySelector(".reference-row .selectable").textContent')==citation
            report['checks'].append('academic citation copied intact via QWebChannel and native writer; PDF handed to system browser without leaving workbench')
            native_click('document.querySelectorAll(".reference-links a")[1]',key=True)
            native_click('document.querySelectorAll(".reference-links a")[1]',Qt.KeyboardModifier.ControlModifier)
            assert len(opened)==3 and all(r['user'] for r in requests),(opened,requests)
            report['checks'].append('DOI link Enter and Ctrl+click each handed to browser exactly once')
            js('document.querySelectorAll(".reference-groups button")[1].click()');query('Vue')
            until('!!document.querySelector(".reference-row")')
            software=js('document.querySelector(".reference-row .selectable").textContent')
            native_click('document.querySelector(".reference-links button")')
            assert clipboard.writes[-1]==software
            report['checks'].append('software citation uses same desktop copy path')
            js('document.querySelector(".dialog-header button").click()');pause()
            button('语音合成');button('方法与来源')
            until('!!document.querySelector(".reference-row")')
            native_click('document.querySelector(".reference-links button")')
            assert len(clipboard.writes)==3
            js('document.querySelector(".dialog-header button").click()');pause()
            report['checks'].append('module-specific reference modal uses shared copy path')
            button('声道工作台')
            until('!!document.querySelector("iframe[title=声道工作台]")?.contentDocument?.querySelector("#aboutDialog")')
            # Use a real pointer event in the same-origin embedded document.
            point=js('(()=>{const f=document.querySelector("iframe[title=声道工作台]"),d=f.contentDocument,e=d.querySelector("#aboutDialog a");d.querySelector("#aboutDialog").showModal();e.scrollIntoView({block:"center"});const r=e.getBoundingClientRect(),fr=f.getBoundingClientRect();return [r.x+fr.x+r.width/2,r.y+fr.y+r.height/2]})()');pause()
            expected=js('document.querySelector("iframe[title=声道工作台]").contentDocument.querySelector("#aboutDialog a").href')
            QTest.mouseClick(w.view.focusProxy(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(point[0]),round(point[1])));pause(300)
            assert opened[-1]==expected,(opened,expected)
            report['checks'].append('embedded M10 model source target=_blank opens through the same native handler')
            before=len(opened);js('window.open("https://example.com/automatic-popup","_blank")');pause(300)
            assert len(opened)==before
            report['checks'].append('automatic script popup still blocked')
            w.view.grab().save(str(out/'m10-source.png'))
        report['opened']=opened;report['requests']=requests;report['clipboard_writes']=clipboard.writes;report['success']=True
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
        w.closing=True;w.close();w.page.deleteLater();app.processEvents()
        host.QApplication.clipboard=old_clipboard;host.QDesktopServices.openUrl=old_open
        print(json.dumps({'out':str(out),'success':report['success'],'checks':report['checks']},ensure_ascii=False))


if __name__=='__main__':main()
