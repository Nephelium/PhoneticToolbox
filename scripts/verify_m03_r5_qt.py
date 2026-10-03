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
    out=ROOT/'output/validation/m03-ui'/('qt-r5-'+uuid4().hex);out.mkdir(parents=True)
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
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile',reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe')
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
        js('(()=>{const a=document.querySelectorAll(".egg-bottom-bar .selection-controls input");for(const [i,v] of [[1,40.5],[0,40]]){a[i].value=v;a[i].dispatchEvent(new Event("input",{bubbles:true}));a[i].dispatchEvent(new Event("change",{bubbles:true}));}})()');pause(100);idle()
        assert js('document.querySelectorAll(".egg-checks input").length')==3
        js('document.querySelectorAll(".egg-checks input").forEach(e=>{if(!e.checked)e.click()})');pause(150);idle()
        until('document.querySelector(".spec-pane").innerText.includes("REAPER F0")')
        assert js('document.querySelectorAll(".spec-pane svg circle").length')>20
        checks.append('Qt real native preview with 3 F0 switches and plotted observations')
        js('document.querySelectorAll(".egg-checks input")[2].click()');pause(100);idle()
        assert not js('document.querySelector(".spec-pane").innerText.includes("REAPER F0")')
        js('document.querySelectorAll(".egg-checks input")[2].click()');pause(100);idle()
        checks.append('Qt REAPER can be hidden and restored')
        for width,height in [(1440,1000),(960,720)]:
            w.resize(width,height);pause(200)
            assert js('(()=>{const e=document.querySelector(".egg-checks");return e.scrollWidth<=e.clientWidth+1})()')
            w.view.grab().save(str(out/f'r5-preview-{width}.png'))
        checks.append('Qt F0 controls fit 2 viewport sizes')
        w.resize(1440,1000);pause(100);click('保存 CSV / 三图')
        until('[...document.querySelectorAll("button")].some(e=>e.textContent.startsWith("查看 ")&&e.textContent.includes("CSV"))',120)
        js('[...document.querySelectorAll("button")].find(e=>e.textContent.startsWith("查看 ")&&e.textContent.includes("CSV")).click()')
        until('[...document.querySelectorAll("dialog button")].some(e=>e.textContent==="选择目录保存完整结果"&&!e.disabled)')
        click('选择目录保存完整结果');deadline=time.monotonic()+30
        while not list(saved.glob('*.ptb.json')) and time.monotonic()<deadline:pause(100)
        meta=json.loads(next(saved.glob('*.ptb.json')).read_text('utf-8'))
        assert meta['f0_analysis']['praat']['floor_hz']==30 and meta['f0_analysis']['praat']['ceiling_hz']==800
        assert meta['f0_analysis']['reaper']['backend']=='native_reaper'
        assert 'F0_REAPER (Hz)' in next(saved.glob('*.csv')).read_text('utf-8')
        from PyQt6.QtGui import QImage
        pngs=list(saved.glob('*.png'));assert len(pngs)==3
        assert all(not QImage(str(p)).isNull() for p in pngs)
        checks.append('Qt native save: CSV + 3 readable PNG + 30-800 Hz actual engine metadata')
        click('返回分析');click('批量分析');until('document.querySelector("dialog")?.innerText.includes("REAPER F0")')
        w.view.grab().save(str(out/'r5-batch.png'));click('返回工作台');checks.append('Qt batch REAPER option present')
        assert hashlib.sha256(private.read_bytes()).hexdigest()==original_hash
        report['original_sha256']=original_hash;report['success']=True
    except Exception as exc:
        report['error']=str(exc);w.view.grab().save(str(out/'failed.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True);w.close();pause(300)
if __name__=='__main__':main()
