"""M03-D owned Qt, real local API/child, isolated copy of existing test schema."""
import json
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
    out=ROOT/'output/validation/m03-ui'/('qt-'+uuid4().hex);out.mkdir(parents=True)
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
    def pause():loop=QEventLoop();QTimer.singleShot(100,loop.quit);loop.exec()
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
        until('document.documentElement.style.getPropertyValue("--font").length>0');click('EGG 信号分析');until('!!document.querySelector(".egg-page")')
        click('打开 WAV 目录');until('document.querySelectorAll(".egg-source option").length===4')
        select('mono.wav');until('document.querySelector(".egg-page [role=alert]")?.textContent.includes("双声道")');checks.append('mono rejected in real file selector')
        select('EGG ɑ̃˥.wav');until('!document.querySelector(".egg-source select").disabled');click('更新分析');ready();checks.append('four real scientific plots from owned compatibility child')
        assert js('document.querySelectorAll(".cq-pane circle").length>0')
        w.view.grab().save(str(out/'light-1440.png'));print('Qt four plots ready',flush=True)
        fill('EGG 选区时长',.12);until('document.querySelectorAll(".egg-four-plots .scientific-plot>svg").length===0');assert js('[...document.querySelectorAll(".egg-control-row button")].find(b=>b.textContent==="播放选区").disabled');click('更新分析');ready();checks.append('changed ROI hides obsolete plots and disables playback until fresh result')
        click('保存 CSV / 三图');wait_exports();until('[...document.querySelectorAll(".egg-result-links button")].some(b=>b.textContent.includes("CSV + 三张图"))')
        js('[...document.querySelectorAll(".egg-result-links button")].find(b=>b.textContent.includes("CSV + 三张图")).click()');until('document.querySelectorAll(".egg-export-image").length===3');QFileDialog.getExistingDirectory=lambda *a,**k:str(saved);click('选择目录保存完整结果');until('document.querySelector(".egg-page [role=status]")?.textContent.includes("已保存 5")');assert list(saved.rglob('*.csv'));click('返回分析');checks.append('single CSV and three PNG shown and saved through native grant')
        click('逆滤波 IF');wait_exports();until('[...document.querySelectorAll(".egg-result-links button")].some(b=>b.textContent.includes("逆滤波"))');js('[...document.querySelectorAll(".egg-result-links button")].find(b=>b.textContent.includes("逆滤波")).click()');until('document.querySelectorAll(".inverse-grid svg").length===4');until('document.querySelectorAll("dialog .audio-transport").length===2');click('选择目录保存完整结果');until('document.querySelector(".egg-page [role=status]")?.textContent.includes("已保存 3")');w.view.grab().save(str(out/'inverse.png'));click('返回分析');checks.append('IF four plots, both normalized/estimated WAV players and actual dual WAV save')
        click('批量分析');until('!!document.querySelector(".batch-files")');click('提交所选文件');until('!document.querySelector(".batch-files")');wait_exports();assert js('document.querySelector(".egg-history").innerText.includes("失败")');assert js('document.querySelector(".egg-history").innerText.includes("已完成")');checks.append('real mixed batch continues past mono failure');click('保存本次批量结果（2）');until('document.querySelector(".egg-page [role=status]")?.textContent.includes("已保存 2 个批次任务")');checks.append('batch saves all successful results with one directory grant')
        js('document.documentElement.dataset.theme="dark"');w.resize(1280,800);pause();w.view.grab().save(str(out/'dark-1280.png'))
        assert js('document.documentElement.scrollWidth<=window.innerWidth');checks.append('dark 1280 width stays inside common shell')
        click('保存参数草稿');js('[...document.querySelectorAll(".tab-close")].find(b=>b.ariaLabel==="关闭 EGG 信号分析").click()');until('!document.querySelector(".egg-page")');click('EGG 信号分析');until('document.querySelectorAll(".egg-history .task-row").length>=6');checks.append('close and reopen restores persisted jobs and parameter draft')
        assert originals=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        for p in saved.rglob('*.wav'):
            rate,data=wavfile.read(p);assert rate==44100 and len(data)==5292
        report['success']=True
    except Exception as e:
        report['error']=str(e);w.view.grab().save(str(out/'failed.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True);w.closing=True;w.close();app.processEvents()

if __name__=='__main__':main()
