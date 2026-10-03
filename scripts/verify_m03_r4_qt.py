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
    out=ROOT/'output/validation/m03-ui'/('qt-r4-'+uuid4().hex);out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone();source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    private=Path(r'C:\Users\13680\Desktop\project\音频数据\EGG测试\3.wav')
    original_hash=hashlib.sha256(private.read_bytes()).hexdigest()
    (inputs/'3.wav').write_bytes(private.read_bytes())
    originals={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
    register_scheme();app=QApplication(['M03-owned-QA']);app.setApplicationName('M03-owned-QA')
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
    w.resize(1440,1000);w.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating);w.show()
    save_target=[saved/'EGG-IF-four.png']
    QFileDialog.getSaveFileName=lambda *a,**k:(str(save_target[0]),'PNG (*.png)')
    checks=[];report=dict(success=False,checks=checks,schema_applied=[])
    QFileDialog.getExistingDirectory=lambda *a,**k:str(saved if '结果' in str(a) else inputs)
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
        click('打开 WAV 目录');until('document.querySelectorAll(".egg-source option").length>=2');select('3.wav');idle();ready()
        assert js('(()=>{const a=[...document.querySelectorAll(".module-toolbar button")].map(e=>e.textContent);return a[a.indexOf("刷新文件")+1]==="批量分析"})()')
        assert js('document.querySelectorAll(".egg-bottom-bar .selection-controls input").length')==2
        assert not js('[...document.querySelectorAll(".workbench-left button")].some(e=>e.textContent.includes("播放选区"))')
        js('(()=>{const a=document.querySelectorAll(".egg-bottom-bar .selection-controls input");for(const [i,v] of [[1,40.5],[0,40]]){a[i].value=v;a[i].dispatchEvent(new Event("input",{bubbles:true}));a[i].dispatchEvent(new Event("change",{bubbles:true}));}})()');pause(100);idle()
        assert js(r'document.querySelector("input[aria-label=\"EGG 选区时长\"]").value')=='0.5'
        fill('EGG 选区时长',.25);pause(100);idle()
        assert js('document.querySelectorAll(".egg-bottom-bar .selection-controls input")[1].value')=='40.25'
        checks.append('Qt endpoints and duration synchronized; batch moved after refresh; full bottom transport')
        for width,height in [(1440,1000),(960,720),(800,600)]:
            w.resize(width,height);pause(250)
            geometry=js('(()=>{const e=document.querySelector(".egg-bottom-bar"),r=e.getBoundingClientRect();return {top:r.top,bottom:r.bottom,view:innerHeight,width:e.clientWidth,scroll:e.scrollWidth}})()')
            assert geometry['top']>=0 and geometry['bottom']<=geometry['view']+1 and geometry['scroll']<=geometry['width']+1,geometry
            w.view.grab().save(str(out/f'r4-bottom-{width}.png'));checks.append(dict(check='Qt visible bottom bar',viewport_width=width,**geometry))
        w.resize(1440,1000);pause(150);click('批量分析');until('document.querySelector("dialog")?.innerText.includes("待处理文件")')
        def cutoff(value):js('(()=>{const e=[...document.querySelectorAll("dialog label")].find(e=>e.textContent.trim()==="低通 Hz").querySelector("input");e.value='+str(value)+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        cutoff(20);click('提交所选文件');until('document.querySelector("dialog [role=alert]")?.textContent.includes("高通")')
        cutoff(1500);w.view.grab().save(str(out/'r4-batch.png'));click('提交所选文件')
        until('!document.querySelector("dialog")');until('[...document.querySelectorAll("button")].some(e=>e.textContent==="保存本次批量结果（1）")',120)
        click('保存本次批量结果（1）');deadline=time.monotonic()+30
        while not list(saved.glob('*.ptb.json')) and time.monotonic()<deadline:pause(100)
        meta=json.loads(next(saved.glob('*.ptb.json')).read_text('utf-8'));assert meta['config']['lowpass_cutoff']==1500 and meta['config']['highpass_cutoff']==25
        assert meta['config']['roi_start']==0 and meta['config']['roi_end'] is None
        assert js(r'document.querySelector("input[aria-label=\"EGG 低通频率\"]").value')=='2000'
        checks.append('Qt invalid cutoffs rejected; real full-file batch native save retains 1500 Hz and independent single-file 2000 Hz')
        assert hashlib.sha256(private.read_bytes()).hexdigest()==original_hash
        report['original_sha256']=original_hash;report['success']=True
    except Exception as exc:
        report['error']=str(exc);w.view.grab().save(str(out/'failed.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True);w.close();pause(300)
if __name__=='__main__':main()
