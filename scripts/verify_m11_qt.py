"""Actual Qt/QWebChannel/workbench and real optional MFA. No test task adapter."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
from uuid import uuid4
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu')
from PyQt6.QtCore import QEventLoop,QTimer
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files
from ptb_worker.mfa.probe import generate

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',required=True);a=p.parse_args()
    out=ROOT/'output/validation/m11'/('qt-'+uuid4().hex);out.mkdir(parents=True);print(out,flush=True)
    components=out/'components';components.mkdir();shutil.copyfile(a.registry,components/'registry.json');os.environ['PTB_M11_COMPONENT_ROOT']=str(components)
    inputs=out/'public';generate(inputs,word='啊')
    (inputs/'probe.wav').rename(inputs/'中文 空格.wav');(inputs/'probe.lab').rename(inputs/'中文 空格.lab')
    nested=inputs/'子目录';generate(nested,word='啊')
    saved=out/'saved';saved.mkdir();db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    register_scheme();app=QApplication(['M11-owned-QA']);window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
    window.resize(1440,1000);window.show()
    QFileDialog.getExistingDirectory=lambda *args,**kwargs:str(inputs if 'corpus' in str(args) else saved)
    report=dict(success=False,checks=[],scope='actual Windows Qt offscreen, production frontend and local host; public synthetic data')
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];window.page.runJavaScript(code,lambda result:(box.append(result),loop.quit()));QTimer.singleShot(5000,loop.quit);loop.exec();return box[0] if box else None
    def until(code,seconds=180):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' '+str(js('document.body.innerText.slice(-5000)')))
    def click(text):js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text)+')?.click()')
    try:
        until('!!document.querySelector("nav")');click('MFA 自动标注');until('!!document.querySelector("[aria-label=\\"MFA 自动标注工作区\\"]")')
        click('组件安装与环境检查');until('document.body.innerText.includes("待发布")');report['checks'].append('formal navigation, optional component manager, honest unpublished download')
        click('收起组件管理');click('选择语料目录');until('document.body.innerText.includes("已读取 2 组音频")')
        js('(()=>{const e=document.querySelector(".parameter-row input");e.value="20";e.dispatchEvent(new Event("change",{bubbles:true}));})()');until('document.querySelectorAll(".parameter-row input")[1].value==="80"')
        js('(()=>{const e=document.querySelector(".parameter-row input");e.value="10";e.dispatchEvent(new Event("change",{bubbles:true}));const r=document.querySelectorAll(".parameter-row input")[1];r.value="40";r.dispatchEvent(new Event("input",{bubbles:true}));})()')
        click('开始对齐');until('document.querySelectorAll(".result-row").length===3',240)
        report['checks'].append('native recursive Chinese/space corpus selection, beam linkage, real persistent MFA, two TextGrids plus provenance')
        click('保存完整结果到输出目录');until('document.body.innerText.includes("已保存 3 个文件")')
        assert len(list(saved.rglob('*.TextGrid')))==2
        report['checks'].append('actual native directory save with new collision-safe subdirectory')
        js('document.documentElement.dataset.theme="light"');pause(300);window.view.grab().save(str(out/'light.png'))
        js('document.documentElement.dataset.theme="dark"');pause(300);window.view.grab().save(str(out/'dark.png'))
        window.resize(900,700);pause(300);window.view.grab().save(str(out/'compact.png'))
        assert js('document.documentElement.scrollWidth<=innerWidth')
        js('document.querySelector("[aria-label=\\"关闭 MFA 自动标注\\"]").click()');until('!!document.querySelector("dialog[open]")');click('取消关闭')
        report['checks'].append('light/dark and compact viewport, formal AppShell close guard')
        report['success']=True
    finally:
        (out/'qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        if not report['success']:window.view.grab().save(str(out/'failure.png'))
        window.close();pause(500);app.quit();print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
