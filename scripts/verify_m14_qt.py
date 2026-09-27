"""Actual Qt host/QWebChannel and built frontend on synthetic files; no DDL."""
import json
import os
import time
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu')
from PyQt6.QtCore import QEventLoop,QTimer
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from verify_m14_wiring import setup,ROOT


def main():
    out,db,cache=setup();inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    register_scheme();app=QApplication(['M14-owned-QA'])
    window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
    window.resize(1440,1000);window.show()
    QFileDialog.getExistingDirectory=lambda *a,**k:str(saved if '结果' in str(a) else inputs)
    report=dict(success=False,checks=[],scope='Windows actual Qt offscreen; built frontend; synthetic phonology table')
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[]
        window.page.runJavaScript(code,lambda result:(box.append(result),loop.quit()))
        QTimer.singleShot(5000,loop.quit);loop.exec();return box[0] if box else None
    def until(code,seconds=60):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' '+str(js('document.body.innerText.slice(-4000)')))
    def click(text):js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text)+')?.click()')
    try:
        import base64
        until('!!document.querySelector("nav")');click('音系归纳');until('!!document.querySelector("[aria-label=音系归纳工作区]")')
        raw=base64.b64encode((ROOT/'tests/fixtures/m14/public.xlsx').read_bytes()).decode()
        js('(()=>{const d=new DataTransfer();d.items.add(new File([Uint8Array.from(atob('+json.dumps(raw)+'),c=>c.charCodeAt(0))],"public.xlsx"));const e=document.querySelector("[aria-label=音系归纳工作区] input[type=file]");e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        until('document.body.innerText.includes("已读取 16 条记录")')
        click('编辑调值顺序与调类');until('!!document.querySelector("dialog[open]")')
        js('(()=>{const e=[...document.querySelectorAll("input")].find(e=>e.getAttribute("aria-label")==="调类 35");e.value="阳平";e.dispatchEvent(new Event("input",{bubbles:true}));})()')
        click('确认调类');click('4 · 结果');click('生成三份结果');until('document.body.innerText.includes("三份结果已完整生成")')
        click('选择目录保存三份结果');until('document.body.innerText.includes("三份结果已保存")')
        assert len(list(saved.glob('*.docx')))==2 and len(list(saved.glob('*.xlsx')))==1
        click('选择目录保存三份结果');until('document.body.innerText.includes("同名结果")')
        report['checks'].append('real Qt/QWebChannel: XLSX import, tone editing, durable three-output generation, native directory save, collision recovery')
        pause(250);window.view.grab().save(str(out/'qt.png'))
        report['success']=True
    finally:
        (out/'qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        js('window.onbeforeunload=null');window.close();pause(300);app.quit();print(out)


if __name__=='__main__':main()
