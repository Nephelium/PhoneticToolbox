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
from verify_m06_wiring import setup,ROOT


def main():
    out,db,cache=setup();inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    (inputs/'合成.wav').write_bytes((ROOT/'tests/fixtures/m06/source.wav').read_bytes())
    register_scheme();app=QApplication(['M06-owned-QA'])
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
        until('!!document.querySelector("nav")');click('语音合成');until('!!document.querySelector("[aria-label=语音合成工作区]")')
        js('(()=>{const e=document.querySelector("[aria-label=总时长]");e.value="0.3";e.dispatchEvent(new Event("input",{bubbles:true}))})()');click('应用时长')
        js('(()=>{const e=document.querySelector("input.ipa-text");e.value="a i";e.dispatchEvent(new Event("input",{bubbles:true}))})()');click('生成元音')
        until('document.body.innerText.includes("元音曲线已生成")');click('合成');until('document.body.innerText.includes("合成完成")');click('导出音频');until('document.body.innerText.includes("已导出合成结果")')
        assert len(list(saved.glob('*.wav')))==1
        report['checks'].append('Qt QWebChannel, generated curves, actual durable synthesis and native WAV/snapshot export')
        pause(250);window.view.grab().save(str(out/'qt.png'))
        report['success']=True
    finally:
        (out/'qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        js('window.onbeforeunload=null');window.close();pause(300);app.quit();print(out)


if __name__=='__main__':main()
