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
        until('document.documentElement.style.getPropertyValue("--font").length>0');click('EGG 信号分析');until('!!document.querySelector(".egg-page")')
        click('打开 WAV 目录');until('document.querySelectorAll(".egg-source option").length===4')
        select('mono.wav');until('document.querySelector(".egg-page [role=alert]")?.textContent.includes("双声道")');checks.append('mono rejected in real file selector')
        select('EGG ɑ̃˥.wav');until('!document.querySelector(".egg-source select").disabled');click('更新分析');ready();checks.append('four real scientific plots from owned compatibility child')
        assert js('document.querySelectorAll(".cq-pane circle").length>0')
        w.view.grab().save(str(out/'light-1440.png'));print('Qt four plots ready',flush=True)
        fill('EGG 选区时长',.12);until('document.querySelectorAll(".egg-four-plots .scientific-plot>svg").length===0');assert js('[...document.querySelectorAll(".egg-control-row button")].find(b=>b.textContent==="播放选区").disabled');click('更新分析');ready();checks.append('changed ROI hides obsolete plots and disables playback until fresh result')
        click('保存 CSV / 三图');wait_exports();until('[...document.querySelectorAll(".egg-result-links button")].some(b=>b.textContent.includes("CSV + 三张图"))')
        js('[...document.querySelectorAll(".egg-result-links button")].find(b=>b.textContent.includes("CSV + 三张图")).click()');until('document.querySelectorAll(".egg-export-image").length===3');QFileDialog.getExistingDirectory=lambda *a,**k:'';click('选择目录保存完整结果');until('document.querySelector("dialog [role=status]")?.textContent.includes("已取消保存")');assert not list(saved.iterdir());QFileDialog.getExistingDirectory=lambda *a,**k:str(saved);click('选择目录保存完整结果');until('document.querySelector("dialog [role=status]")?.textContent.includes("已保存 5")');assert list(saved.rglob('*.csv'));click('返回分析');checks.append('single CSV and three PNG shown and saved through native grant')
        click('逆滤波 IF');wait_exports();until('[...document.querySelectorAll(".egg-result-links button")].some(b=>b.textContent.includes("逆滤波"))');js('[...document.querySelectorAll(".egg-result-links button")].find(b=>b.textContent.includes("逆滤波")).click()');until('document.querySelectorAll(".inverse-grid svg").length===4');until('document.querySelectorAll("dialog .audio-transport").length===2');click('选择目录保存完整结果');until('document.querySelector("dialog [role=status]")?.textContent.includes("已保存 3")');w.view.grab().save(str(out/'inverse.png'));click('返回分析');checks.append('IF four plots, both normalized/estimated WAV players and actual dual WAV save')
        click('批量分析');until('!!document.querySelector(".batch-files")');click('提交所选文件');until('!document.querySelector(".batch-files")');wait_exports();assert js('document.querySelector(".egg-history").innerText.includes("失败")');assert js('document.querySelector(".egg-history").innerText.includes("已完成")');checks.append('real mixed batch continues past mono failure');click('保存本次批量结果（2）');until('document.querySelector(".egg-page [role=status]")?.textContent.includes("已保存 2 个批次任务")');checks.append('batch saves all successful results with one directory grant')
        js('document.documentElement.dataset.theme="dark"');w.resize(1280,800);pause();w.view.grab().save(str(out/'dark-1280.png'))
        assert js('document.documentElement.scrollWidth<=window.innerWidth');checks.append('dark 1280 width stays inside common shell')
        click('保存参数草稿');js('[...document.querySelectorAll(".tab-close")].find(b=>b.ariaLabel==="关闭 EGG 信号分析").click()');until('!document.querySelector(".egg-page")');click('EGG 信号分析');until('document.querySelectorAll(".egg-history .task-row").length>=6');checks.append('close and reopen restores persisted jobs and parameter draft')
        if options.ranges:
            wavfile.write(inputs/'wide.wav',44100,np.tile(samples,(8,1)))
            originals['wide.wav']=hashlib.sha256((inputs/'wide.wav').read_bytes()).hexdigest()
            QFileDialog.getExistingDirectory=lambda *a,**k:str(inputs)
            click('打开 WAV 目录');until('document.querySelectorAll(".egg-source option").length===5')
            select('wide.wav');until('!document.querySelector(".egg-source select").disabled')
            fill('EGG 选区起点',3);fill('EGG 选区时长',.5)
            for width in (5,5000):
                fill('EGG 微观窗口',width);click('更新分析');ready();wait_exports()
                assert js('Number([...document.querySelectorAll("input")].find(e=>e.ariaLabel==="EGG 微观窗口").value)')==width
                # DOM readiness precedes Qt's compositor; allow the visible frame to paint.
                pause(500)
                w.view.grab().save(str(out/('micro-'+str(width)+'.png')))
            assert js('document.querySelector(".audio-pane small").textContent.includes("每 32 点")')
            checks.append('5 and 5000ms windows compute and render in actual Qt with explicit display stride')
        if options.include_private:
            QFileDialog.getExistingDirectory=lambda *a,**k:str(inputs)
            click('打开 WAV 目录');until('document.querySelectorAll(".egg-source option").length===4')
            confirmed=json.loads((ROOT/'output/validation/p03/confirmed-egg.json').read_text('utf-8'))
            baseline=json.loads((ROOT/'tests/fixtures/m03/manifest.json').read_text('utf-8'))
            report['private_cases']=[]
            for case in confirmed['cases']:
                original=Path(case['input']);raw=original.read_bytes();digest=hashlib.sha256(raw).hexdigest()
                assert digest==next(c for c in baseline['private_cases'] if c['id']==case['case_id'])['input_sha256']
                # Only this already-authorized input is staged into the ignored test fixture area.
                name=case['case_id']+'.wav';(inputs/name).write_bytes(raw);originals[name]=digest
                click('刷新文件');until('document.querySelectorAll(".egg-source option").length==='+str(len(originals)+1))
                select(name);until('!document.querySelector(".egg-source select").disabled')
                flipped=bool(js('document.querySelector(".egg-source").innerText.includes("左：音频")'))
                if flipped!=case['flip_channels']:click('交换声道')
                click('更新分析');ready();wait_exports()
                with sqlite3.connect(db) as conn:
                    rows=conn.execute("SELECT snapshot,result_manifest FROM jobs WHERE state='succeeded' ORDER BY created_at DESC").fetchall()
                snapshot,manifest=next((json.loads(a),json.loads(b)) for a,b in rows if name in a and json.loads(a)['config']['analysis']['mode']=='preview')
                meta_file=next(f for f in manifest['files'] if f['name']=='egg.ptb.json')
                meta=json.loads((cache/(meta_file['id']+'.bin')).read_text('utf-8'))
                assert meta['input_sha256']==digest and meta['config']['flip_channels']==case['flip_channels']
                assert meta['preview']['schema_version']=='egg-preview/1'
                assert all(abs(v)<=.750001 for v in meta['preview']['audio']['values'] if v is not None)
                assert hashlib.sha256(original.read_bytes()).hexdigest()==digest
                report['private_cases'].append(dict(id=case['case_id'],sha256=digest,samples=meta['sample_count'],roi=meta['selection'],four_plots=True))
                w.view.grab().save(str(out/(case['case_id']+'-onset-private.png')))
                # Select a loud half-second for a second, non-onset UI check.
                rate,values=wavfile.read(original);audio=values[:,0 if case['flip_channels'] else 1].astype(float)
                block=int(rate*.5);count=len(audio)//block;energies=np.abs(audio[:count*block]).reshape(count,block).mean(axis=1)
                first=int(np.argmax(energies))*block
                fill('EGG 选区起点',first/rate);fill('EGG 选区时长',.5)
                js('[...document.querySelectorAll(".egg-parameter-row input[type=checkbox]")].filter(e=>["Praat F0","GCI F0"].includes(e.parentElement.textContent.trim())).forEach(e=>{if(!e.checked)e.click()})')
                click('更新分析');ready();wait_exports()
                with sqlite3.connect(db) as conn:rows=conn.execute("SELECT snapshot,result_manifest FROM jobs WHERE state='succeeded' ORDER BY created_at DESC").fetchall()
                _,manifest=next((json.loads(a),json.loads(b)) for a,b in rows if name in a and json.loads(a)['config']['analysis']['mode']=='preview')
                meta_file=next(f for f in manifest['files'] if f['name']=='egg.ptb.json');meta=json.loads((cache/(meta_file['id']+'.bin')).read_text('utf-8'))
                assert abs(meta['selection']['start_sample']-first)<=1 and meta['input_sha256']==digest
                assert any(v is not None and v>0 for v in meta['preview']['praat']['values'])
                assert any(v is not None and abs(v)>.001 for v in meta['preview']['audio']['values'])
                report['private_cases'][-1]['voiced_roi']=meta['selection'];report['private_cases'][-1]['praat_points']=sum(v is not None for v in meta['preview']['praat']['values'])
                assert hashlib.sha256(original.read_bytes()).hexdigest()==digest
                w.view.grab().save(str(out/(case['case_id']+'-voiced-private.png')))
            checks.append('two authorized natural recordings open in actual Qt with matched hash, role and four plots')
        assert originals=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        for p in saved.rglob('*.wav'):
            rate,data=wavfile.read(p);assert rate==44100 and len(data)==5292
        report['success']=True
    except Exception as e:
        report['error']=str(e);w.view.grab().save(str(out/'failed.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True);w.closing=True;w.close();app.processEvents()

if __name__=='__main__':main()
