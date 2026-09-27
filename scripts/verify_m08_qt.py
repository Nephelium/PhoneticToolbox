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
from verify_m08_wiring import setup,ROOT


def main():
    out,db,cache=setup();inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    (inputs/'合成.wav').write_bytes((ROOT/'output/validation/m08-wiring/input.wav').read_bytes())
    register_scheme();app=QApplication(['M08-owned-QA'])
    window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
    window.resize(1440,1000);window.show()
    QFileDialog.getExistingDirectory=lambda *a,**k:str(saved if '结果' in str(a) else inputs)
    report=dict(success=False,checks=[],scope='Windows actual Qt offscreen; built frontend; synthetic audio')
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
        until('!!document.querySelector("nav")');click('变速变调');until('!!document.querySelector(".m08-page")');click('打开音频目录')
        until('document.querySelectorAll(".m08-page select option").length>1')
        js('(()=>{const e=document.querySelector(".m08-page select");e.value=e.options[1].value;e.dispatchEvent(new Event("change",{bubbles:true}))})()')
        until('!!document.querySelector(".m08-curve") && document.querySelector(".m08-page").getAttribute("aria-busy")==="false"');click('合成当前视野')
        until('document.querySelector(".m08-page").innerText.includes("输出 1.000 s")')
        click('保存并编号');until('document.querySelectorAll(".history li").length===1')
        until('document.querySelector(".m08-page").getAttribute("aria-busy")==="false" && (document.querySelector(".history-plot svg path")?.getAttribute("d")?.length||0)>30')
        assert len(list(saved.glob('*.wav')))==1
        report['checks'].append('real Qt transport: source decoding, F0, durable synthesis, native save and history')
        pause(250);window.view.grab().save(str(out/'qt.png'))
        report['success']=True
    finally:
        (out/'qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        js('window.onbeforeunload=null');window.close();pause(300);app.quit();print(out)


if __name__=='__main__':main()
