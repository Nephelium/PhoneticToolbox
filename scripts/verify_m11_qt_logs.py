"""Reopen the completed owned Qt fixture to check history and redacted native log."""
import argparse,json,os,time
from pathlib import Path
from uuid import uuid4
os.environ.setdefault('QT_QPA_PLATFORM','offscreen');os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu')
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QEventLoop,QTimer
from ptb_desktop.host import Workbench,register_scheme

def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];fixture=Path(a.fixture).absolute();out=fixture/('logs-'+uuid4().hex[:8]);out.mkdir()
    os.environ['PTB_M11_COMPONENT_ROOT']=str(fixture/'components')
    register_scheme();app=QApplication(['M11-log-QA']);window=Workbench(root/'frontend/dist',test=True,jobs_path=fixture/'jobs.sqlite3',local_files_root=fixture/'cache',vocal_profile=out/'profile');window.resize(1440,1000);window.show()
    def pause():
        loop=QEventLoop();QTimer.singleShot(100,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];window.page.runJavaScript(code,lambda v:(box.append(v),loop.quit()));QTimer.singleShot(5000,loop.quit);loop.exec();return box[0] if box else None
    def until(code):
        start=time.monotonic()
        while time.monotonic()-start<45:
            if js(code):return
            pause()
        raise AssertionError(code+' '+str(js('document.body.innerText.slice(-2000)')))
    report=dict(success=False)
    try:
        until('!!document.querySelector("nav")');js('[...document.querySelectorAll("nav button")].find(b=>b.textContent.trim()==="MFA 自动标注").click()')
        until('document.querySelector(".mfa-results select")?.options.length>1')
        js('(()=>{const e=document.querySelector(".mfa-results select");e.selectedIndex=1;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        until('document.querySelector(".native-log")?.textContent.length>0')
        text=js('document.querySelector(".native-log").textContent');assert 'C:\\' not in text and 'D:\\' not in text and 'INFO' in text
        js('document.querySelector(".native-log").parentElement.open=true');pause();window.view.grab().save(str(out/'native-log.png'))
        report.update(success=True,checks=['actual Qt completed history reopens','native log loads through actual bridge and own task directory','host drive paths redacted'])
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');window.close();pause();app.quit();print(out);print(json.dumps(report))

if __name__=='__main__':main()
