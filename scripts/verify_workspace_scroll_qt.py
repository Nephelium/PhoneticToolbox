"""P04-SCROLL: native Qt wheel events on the real shared workbench."""
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
from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint,QPointF,QRect
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();extra=parser.add_mutually_exclusive_group();extra.add_argument('--include-private',action='store_true');extra.add_argument('--ranges',action='store_true');options=parser.parse_args()
    out=ROOT/'output/validation/m03-ui'/('qt-scroll-'+uuid4().hex);out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone();source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    with np.load(ROOT/'tests/fixtures/m03/EGG-SYN-PCM16.npz') as a:samples=np.column_stack([a['load.egg_signal_raw'],a['load.audio_signal']])
    wavfile.write(inputs/'EGG ɑ̃˥.wav',44100,samples);wavfile.write(inputs/'mono.wav',44100,samples[:,0]);wavfile.write(inputs/'silence.wav',44100,np.zeros_like(samples))
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
    duration_expr='Number(document.querySelector('+json.dumps('input[aria-label="EGG 选区时长"]')+').value)'
    def wheel(selector,delta=-240,control=False):
        rect=js('(()=>{const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return [r.x+r.width/2,r.y+Math.min(60,r.height/2)];})()')
        target=w.view.focusProxy() or w.view;pos=QPoint(round(rect[0]),round(rect[1]));local=target.mapFrom(w.view,pos)
        event=QWheelEvent(QPointF(local),QPointF(target.mapToGlobal(local)),QPoint(),QPoint(0,delta),Qt.MouseButton.NoButton,Qt.KeyboardModifier.ControlModifier if control else Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
        app.sendEvent(target,event);pause(500)
    try:
        until('document.documentElement.style.getPropertyValue("--font").length>0');click('EGG 信号分析');until('!!document.querySelector(".egg-page")');click('打开 WAV 目录');until('document.querySelectorAll(".egg-source option").length===4');select('EGG ɑ̃˥.wav');until('!document.querySelector(".egg-source select").disabled');click('更新分析');ready()
        for width,height,theme in [(1280,800,'light'),(800,500,'dark')]:
            w.resize(width,height);js('document.documentElement.dataset.theme='+json.dumps(theme));pause(500);js('document.querySelector("main").scrollTop=0');pause()
            before=js(duration_expr);assert before==.5
            wheel('.cq-pane svg');assert js('document.querySelector("main").scrollTop>0')
            assert js(duration_expr)==before
            for _ in range(5):wheel('main',-600)
            assert js('document.querySelector(".egg-result-links").getBoundingClientRect().bottom<=innerHeight')
            w.view.grab().save(str(out/f'bottom-{width}-{height}.png'));checks.append(f'{width}x{height}: native wheel scrolls to bottom without zoom')
        w.resize(1280,800);js('document.querySelector("main").scrollTop=0');pause(500);wheel('.cq-pane svg',120,True);until(duration_expr+'<.5');ready();checks.append('native Ctrl-wheel zoom preserved')
        click('批量分析');until('!!document.querySelector("dialog")');w.resize(800,500);pause(500)
        bottom=js('document.querySelector(".dialog-actions").getBoundingClientRect().bottom');assert bottom<=js('innerHeight')
        wheel('.dialog-body',-600);assert js('document.querySelector(".dialog-body").scrollTop>0');assert js('document.querySelector(".dialog-actions").getBoundingClientRect().bottom')==bottom
        w.view.grab().save(str(out/'dialog.png'));checks.append('native wheel scrolls dialog body; actions remain visible');click('返回工作台')
        w.resize(1280,800)
        for factor in [1.25,1.5,2.0]:
            w.view.setZoomFactor(factor);pause(500);js('document.querySelector("main").scrollTop=0');wheel('main',-600)
            assert js('document.querySelector("main").scrollTop>0');checks.append(f'Qt page zoom {factor}: content remains scrollable')
        w.view.setZoomFactor(1);w.resize(800,500);click('声道工作台');until('!!document.querySelector(".vocal-page iframe")');pause(1000)
        js('document.querySelector("main").scrollTop=0');wheel('.vocal-page iframe',-600)
        assert js('document.querySelector("main").scrollTop>0');assert js('document.querySelector(".vocal-page").getBoundingClientRect().height>=640')
        w.view.grab().save(str(out/'vocal-scroll.png'));checks.append('M10 iframe header wheel chains to outer scroll; minimum readable height retained; no recording test')
        class SmallScreen:
            def availableGeometry(self):return QRect(0,0,720,460)
        w.fit_screen(SmallScreen());assert w.minimumWidth()==688 and w.minimumHeight()==396;assert w.width()<=688 and w.height()<=396
        w.fit_screen();assert w.minimumWidth()<=800 and w.minimumHeight()<=500;checks.append('small available-screen geometry clamps minimum and actual size; restores normal minimum')
        report['success']=True
    except Exception as e:
        report['error']=str(e);w.view.grab().save(str(out/'failed.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True);w.closing=True;w.close();app.processEvents()

if __name__=='__main__':main()
