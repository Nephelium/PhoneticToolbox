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
from PyQt6.QtCore import QEventLoop,QTimer,Qt,QRect
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();extra=parser.add_mutually_exclusive_group();extra.add_argument('--include-private',action='store_true');extra.add_argument('--ranges',action='store_true');options=parser.parse_args()
    out=ROOT/'output/validation/m03-ui'/('qt-overview-'+uuid4().hex);out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone();source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    i=np.arange(120*48000,dtype=np.int64);gain=np.where(i<len(i)//2,1,2)
    samples=np.column_stack([((i%320)*2-320)*70*gain,((i%240)*2-240)*60*gain]).astype(np.int16)
    wavfile.write(inputs/'long.wav',48000,samples)
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
    try:
        until('document.documentElement.style.getPropertyValue("--font").length>0');click('EGG 信号分析');until('!!document.querySelector(".egg-page")');click('打开 WAV 目录');until('document.querySelectorAll(".egg-source option").length===2');select('long.wav');until('!document.querySelector(".egg-source select").disabled')
        w.resize(1280,800);js('document.documentElement.dataset.theme="dark";document.querySelector(".egg-overview").scrollIntoView({block:"center"})');pause(500)
        rect=js('(()=>{const e=document.querySelector(".egg-overview"),r=e.getBoundingClientRect(),g=e.querySelector(".wave-track svg").getBoundingClientRect(),c=e.querySelector(".overview-controls").getBoundingClientRect();return [r.x,r.y,r.width,r.height,g.y-r.y,g.height,c.y-g.bottom]})()')
        assert rect[4]<15 and rect[5]==110 and rect[6]>=0;w.view.grab(QRect(*[round(x) for x in rect[:4]])).save(str(out/'overview.png'));checks.append('native Qt compact overview: waveform first, 110px height, controls underneath')
        click('适合窗口');assert js('document.querySelector(".overview-controls .mono").textContent.includes("2")');js('document.querySelector(".overview-controls input[type=checkbox]").click()');until('document.querySelectorAll(".egg-overview .wave-track svg").length===2');checks.append('native controls preserve fit-to-60s and dual-channel display');report['success']=True
    except Exception as e:
        report['error']=str(e);raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True);w.closing=True;w.close();app.processEvents()

if __name__=='__main__':main()
