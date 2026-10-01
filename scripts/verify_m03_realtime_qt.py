"""M03-D owned Qt, real local API/child, isolated copy of existing test schema."""
import json
import argparse
import os
import sqlite3
import time
import hashlib
from pathlib import Path
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
from PyQt6.QtCore import QEventLoop,QTimer,Qt
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();extra=parser.add_mutually_exclusive_group();extra.add_argument('--include-private',action='store_true');extra.add_argument('--ranges',action='store_true');options=parser.parse_args()
    out=ROOT/'output/validation/m03-ui'/('qt-realtime-'+uuid4().hex);out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone();source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    with np.load(ROOT/'tests/fixtures/m03/EGG-SYN-PCM16.npz') as a:samples=np.column_stack([a['load.egg_signal_raw'],a['load.audio_signal']])
    wavfile.write(inputs/'EGG ɑ̃˥.wav',44100,samples);wavfile.write(inputs/'mono.wav',44100,samples[:,0]);wavfile.write(inputs/'silence.wav',44100,np.zeros_like(samples))
    if os.environ.get('PTB_M03_TEST_INPUT'):(inputs/'realtime.wav').write_bytes(Path(os.environ['PTB_M03_TEST_INPUT']).read_bytes())
    originals={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
    register_scheme();app=QApplication(['M03-owned-QA']);app.setApplicationName('M03-owned-QA')
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
    w.resize(1440,1000);w.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating);w.show()
    checks=[];report=dict(success=False,checks=checks,schema_applied=[])
    QFileDialog.getExistingDirectory=lambda *a,**k:str(saved if '输出' in str(a) else inputs)
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda v:(box.append(v),loop.quit()));QTimer.singleShot(5000,loop.quit);loop.exec();return box[0] if box else None
    def pause(milliseconds=100):loop=QEventLoop();QTimer.singleShot(milliseconds,loop.quit);loop.exec()
    def until(code,seconds=60):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError('UI timeout: '+code+'; '+str(js('document.querySelector(".egg-page")?.innerText.slice(0,1800)')))
    def click(text):js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text,ensure_ascii=False)+')?.click()')
    def fill(label,value):js('(()=>{const e=document.querySelector('+json.dumps('input[aria-label="'+label+'"]')+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()')
    def select(name):js('(()=>{const e=document.querySelector(".egg-source select");e.value=[...e.options].find(o=>o.textContent==='+json.dumps(name,ensure_ascii=False)+').value;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
    def ready():until('document.querySelectorAll(".egg-four-plots .scientific-plot>svg").length===4')
    def wait_exports():until('![...document.querySelectorAll(".egg-history .task-row [role=status]")].some(e=>["排队中","运行中","正在取消"].includes(e.textContent))',120)

    def idle():until('[...document.querySelectorAll(".egg-page button")].some(b=>b.textContent==="保存 CSV / 三图"&&!b.disabled)',60)
    try:
        until('document.documentElement.style.getPropertyValue("--font").length>0');click('EGG 信号分析');until('!!document.querySelector(".egg-page")')
        click('打开 WAV 目录');until('document.querySelectorAll(".egg-source option").length>=4')
        start=time.monotonic();select('EGG ɑ̃˥.wav');idle();ready()
        checks.append(dict(check='Qt real file selection auto draws four plots',seconds=time.monotonic()-start))
        fill('EGG 选区起点',.123456);idle();assert js('document.querySelector(".audio-pane header").textContent.includes("0.3735")')
        fill('EGG 微观窗口',100);idle();checks.append('Qt parameters auto refresh without update button')
        js('document.querySelector(".egg-result-links button").click()');until('!document.querySelector("dialog")&&document.querySelector(".audio-pane header")?.textContent.includes("0.3735")')
        assert js('!document.querySelector(".egg-page p.notice")');checks.append('Qt fractional history restore stays valid')
        select('EGG ɑ̃˥.wav');idle();fill('EGG 选区时长',.12);idle()
        click('保存 CSV / 三图');wait_exports();until('[...document.querySelectorAll(".egg-result-links button")].some(b=>b.textContent.includes("CSV + 三张图"))')
        js('[...document.querySelectorAll(".egg-result-links button")].find(b=>b.textContent.includes("CSV + 三张图")).click()');until('document.querySelectorAll(".egg-export-image").length===3')
        QFileDialog.getExistingDirectory=lambda *a,**k:str(saved);click('选择目录保存完整结果');until('document.querySelector("dialog [role=status]")?.textContent.includes("已保存 5")');assert list(saved.glob('*.csv'));click('返回分析')
        checks.append('Qt single CSV and three PNG display and native save')
        click('逆滤波 IF');wait_exports();until('[...document.querySelectorAll(".egg-result-links button")].some(b=>b.textContent.includes("逆滤波"))')
        js('[...document.querySelectorAll(".egg-result-links button")].find(b=>b.textContent.includes("逆滤波")).click()');until('document.querySelectorAll("dialog .audio-transport").length===2');click('返回分析');checks.append('Qt inverse result reads both audio roles')
        if os.environ.get('PTB_M03_TEST_INPUT'):
            start=time.monotonic();select('realtime.wav');idle();ready();checks.append(dict(check='Qt actual 77s source auto draws four plots',seconds=time.monotonic()-start))
            start=time.monotonic();fill('EGG 选区起点',29.004);fill('EGG 选区时长',1.5551);idle();checks.append(dict(check='Qt actual source ROI auto update',seconds=time.monotonic()-start))
        js('document.documentElement.dataset.theme="light"');pause(300);w.view.grab().save(str(out/'realtime-light.png'));js('document.documentElement.dataset.theme="dark"');pause(300);w.view.grab().save(str(out/'realtime-dark.png'))
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==originals[p.name] for p in inputs.iterdir())
        report['success']=True
    except Exception as exc:
        report['error']=str(exc);w.view.grab().save(str(out/'failed.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True);w.close();pause(300)
if __name__=='__main__':main()
