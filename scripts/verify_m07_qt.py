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
from verify_m07_wiring import setup,ROOT


def main():
    out,db,cache=setup();inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    for i in (0,1):(inputs/f'input{i}.wav').write_bytes((ROOT/f'output/validation/m07/baseline/round1/input{i}.wav').read_bytes())
    register_scheme();app=QApplication(['M07-owned-QA'])
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
        until('!!document.querySelector("nav")');click('发声类型合成');until('!!document.querySelector("[aria-label=发声类型连续统工作区]")')
        click('打开音频目录');until('document.querySelector("select[aria-label=源音频]").options.length===3')
        def set_value(label,value):
            js('(()=>{const e=document.querySelector('+json.dumps('input[aria-label="'+label+'"],select[aria-label="'+label+'"]')+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}))})()')
        for label,index in [('源音频',1),('目标音频',2)]:
            value=js('document.querySelector('+json.dumps('select[aria-label="'+label+'"]')+').options['+str(index)+'].value');set_value(label,value)
        until('[...document.querySelectorAll("button")].some(b=>b.textContent.trim()==="提取 F0"&&!b.disabled)');click('提取 F0')
        until('document.body.innerText.includes("分析完成，控制点尚未修改")')
        set_value('源 F0 第 4 点','130');click('应用编辑');until('document.body.innerText.includes("F0 编辑已应用")')
        set_value('连续统步数','3');click('生成当前');until('document.querySelectorAll(".result-row").length===1')
        click('输出位置');click('保存完整组');until('document.body.innerText.includes("已保存本组完整文件和参数清单")')
        assert len(list(saved.glob('M07-*/*.wav')))==4
        click('step01');until('document.body.innerText.includes("当前试听：")')
        report['checks'].append('actual Qt/QWebChannel source selection, analysis, explicit F0 edit, owned synthesis, four WAVs and complete snapshot save')
        pause(250);window.view.grab().save(str(out/'qt.png'))
        report['success']=True
    finally:
        (out/'qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        js('window.onbeforeunload=null');window.close();pause(300);app.quit();print(out)


if __name__=='__main__':main()
