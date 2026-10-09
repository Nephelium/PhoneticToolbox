"""Actual offscreen Qt, local task service and native exports, isolated test store."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --mute-audio')
import json
import sqlite3
import time
import hashlib
import sys
from pathlib import Path
from uuid import uuid4
import numpy as np
import soundfile as sf
from PyQt6.QtCore import QEventLoop,QTimer,Qt
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files

ROOT=Path(__file__).resolve().parents[1]

def main():
    selected_only='--selected-only' in sys.argv
    out=ROOT/'output/validation/m01-r4'/('qt-'+uuid4().hex);inputs=out/'inputs';inputs.mkdir(parents=True)
    t=np.arange(43*16000)/16000;audio=.2*np.sin(2*np.pi*200*t);egg=.6*np.sin(2*np.pi*100*t)
    sf.write(inputs/'a.wav',np.column_stack([egg,audio]),16000,subtype='FLOAT')
    sf.write(inputs/'b.wav',np.column_stack([audio,egg]),16000,subtype='FLOAT')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    db=out/'tasks.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(db) as dest:src.backup(dest)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    register_scheme();app=QApplication(['M01-R4-owned-offscreen-QA'])
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe',vocal_profile=out/'vocal')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen,True);w.resize(1800,1000);w.show()
    chosen=[inputs];QFileDialog.getExistingDirectory=lambda *a,**k:str(chosen[0])
    report={'success':False,'checks':[],'layouts':[]};batches=[];original=w.service.request
    def request(path,method='GET',body=None):
        value=original(path,method,body)
        if path=='/api/v1/jobs/batches/create':batches.append(value['id'])
        return value
    w.service.request=request
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        values=[];loop=QEventLoop();w.page.runJavaScript(code,lambda value:(values.append(value),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        if not values:raise RuntimeError('JS timeout')
        return values[0]
    def until(code,seconds=90):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' / '+str(js('[...document.querySelectorAll(".m01-results,.m02-page .error-banner,[role=status]")].map(e=>e.textContent)')))
    def click(text):
        query='[...document.querySelectorAll("button")].find(b=>b.offsetParent&&!b.disabled&&(b.getAttribute("aria-label")??b.textContent.trim())==='+json.dumps(text)+')'
        until('!!'+query);js(query+'.click()');pause()
    def set_select(label,value):
        js('(()=>{const e=document.querySelector('+json.dumps('select[aria-label="'+label+'"]')+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
    def active_error():return js('[...document.querySelectorAll(".error-banner,[role=alert]")].filter(e=>e.offsetParent).map(e=>e.textContent)')
    try:
        until('document.querySelector(".host-badge")?.textContent==="本地桌面"');click('参数估计');click('选择音频目录')
        until('document.querySelectorAll(".m01-file-entry").length===2');js('document.querySelector(".m01-file-list .file-row").click()');until('!!document.querySelector(".wave-track")')
        click('选择输出参数');click('全不选')
        for key in ('pF0','rF0','Intensity'):js('document.querySelector(".parameter-grid input[value='+key+']").click()')
        click('应用到草稿');js('document.querySelector(".m01-joint input[type=checkbox]").click()');pause()
        set_select('EGG参数保存方式','cycles')
        js('document.querySelectorAll(".m01-file-list .file-row")[1].click()');until('!!document.querySelector(".wave-track")')
        set_select('当前文件分析声道','0')
        height=js('document.querySelector(".m01-file-progress").getBoundingClientRect().height')
        if selected_only:
            assert js('[...document.querySelectorAll(".m01-batch-bar button")].find(b=>b.textContent==="分析选中文件").disabled')
            js('document.querySelectorAll(".m01-file-entry input[type=checkbox]")[1].click()');pause()
            click('分析选中文件');until('document.querySelector(".m01-page")?.textContent.includes("已保存2个结果文件")',240)
            assert not (inputs/'a.xlsx').exists() and not (inputs/'a.ptb.sqlite').exists()
        else:
            click('开始全列表分析');until('document.querySelector(".m01-page")?.textContent.includes("已保存4个结果文件")',240)
        assert abs(js('document.querySelector(".m01-file-progress").getBoundingClientRect().height')-height)<1
        assert js('document.querySelector("progress[aria-label=单个音频处理进度]").value')==1
        batch=w.service.get('/api/v1/jobs/batches/'+batches[-1]);assert batch['summary']['complete'],batch
        if selected_only:assert batch['summary']['total']==1 and batch['audio_names']==['b.wav'],batch
        for name in (('b',) if selected_only else ('a','b')):
            conn=sqlite3.connect(inputs/(name+'.ptb.sqlite'))
            values=conn.execute('SELECT "F0 - Praat" FROM params WHERE Time_s BETWEEN 1 AND 42').fetchall()
            assert abs(np.nanmedian(np.array(values,float))-200)<1
            assert abs(np.nanmedian(np.array(conn.execute('SELECT gF0 FROM egg_cycles').fetchall(),float))-100)<1
            assert conn.execute('SELECT COUNT(*) FROM params').fetchone()[0]==8600
            conn.close()
        report['checks'].append('Two actual 43s stereo tasks beyond old 2M sample gate, reverse-channel override, raw EGG cycle tables, native REAPER, XLSX/SQLite auto-save and retained progress')
        if selected_only:
            report['checks']=['M01-R5 single selected file skips unmarked audio and preserves its reversed audio/EGG channels at new batch index 0']
            js('document.querySelectorAll(".m01-file-entry input[type=checkbox]")[0].click()');pause();click('分析选中文件')
            until('document.querySelector(".m01-page")?.textContent.includes("已保存4个结果文件")',240)
            batch=w.service.get('/api/v1/jobs/batches/'+batches[-1]);assert batch['summary']['complete'] and batch['audio_names']==['a.wav','b.wav'],batch
            assert (inputs/'a.xlsx').exists() and (inputs/'a.ptb.sqlite').exists()
            report['checks'].append('M01-R5 multiple checked files submit in list order and save both formats')
        for width,height in ((1800,1000),(1000,760),(650,800)):
            w.resize(width,height);pause(300)
            assert js('document.querySelector(".m01-file-progress").getBoundingClientRect().width>0')
            assert js('(()=>{const e=document.querySelector(".m01-joint");return e.scrollWidth<=e.clientWidth+1;})()')
            report['layouts'].append([width,height])
        w.resize(1600,1000)
        for mode in ('light','dark'):
            js('document.documentElement.dataset.theme='+json.dumps(mode));pause(200);w.view.grab().save(str(out/('m01-'+mode+'.png')))
        if selected_only:
            assert all(hashlib.sha256((inputs/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
            report['success']=True
            return
        click('参数显示');click('选择音频目录');click('a.wav')
        until('document.querySelectorAll(".m02-parameters input").length>=6')
        for name in ('CQ','SQ','F0 - GCI'):js('document.querySelector('+json.dumps('.m02-parameters input[aria-label="'+name+'"]')+').click()')
        click('将 3 项分配到图窗');until('document.querySelectorAll(".m02-page .parameter-curve[d*=L]").length===3')
        for filename in ('a.xlsx','a.ptb.sqlite'):
            js('(()=>{const e=document.querySelector(".m02-page .signal-panel>label select");e.value=[...e.options].find(o=>o.textContent==='+json.dumps(filename)+').value;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            until('document.querySelectorAll(".m02-page .parameter-curve[d*=L]").length===3&&!document.querySelector(".m02-page [aria-busy=true]")',240)
            assert not active_error(),active_error()
        report['checks'].append('Actual M02 opens both native export formats and draws CQ/SQ/GCI F0 at independent event times')
        w.view.grab().save(str(out/'m02-native.png'))
        # Open the already-generated max-duration evidence with its managed view reader.
        candidates=[p.parent for p in (ROOT/'output/validation/m01-r4').glob('long-*/verification.json')
            if json.loads(p.read_text('utf-8')).get('seconds')==1800 and json.loads(p.read_text('utf-8')).get('success')]
        assert candidates,'Run verify_m01_r4_long.py --seconds 1800 first'
        long=max(candidates,key=lambda p:(p/'verification.json').stat().st_mtime)
        report['long_evidence']=str(long.relative_to(ROOT))
        if long.exists():
            chosen[0]=long;click('选择音频目录');click('input.wav');until('!!document.querySelector(".m02-page .wave-track")&&!document.querySelector(".m02-page>p[role=status]")',240)
            for filename in ('result.ptb.sqlite','result.xlsx'):
                js('(()=>{const e=document.querySelector(".m02-page .signal-panel>label select");e.value=[...e.options].find(o=>o.textContent==='+json.dumps(filename)+').value;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
                until('document.querySelectorAll(".m02-page .parameter-curve[d*=L]").length===3&&!document.querySelector(".m02-page [aria-busy=true]")',240)
                assert not active_error(),active_error()
            report['checks'].append('Actual Qt previews 30min stereo and reads both large SQLite and XLSX via bounded window queries')
            js('(()=>{const e=[...document.querySelectorAll(".m02-toolbar label")].find(e=>e.textContent.includes("时间窗长度")).querySelector("input");e.value="0.1";e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            pause(300)
            until('document.querySelectorAll(".m02-page .parameter-curve[d*=L]").length===3&&!document.querySelector(".m02-page [aria-busy=true]")',60)
            until('(document.querySelector(".m02-page .parameter-curve[data-parameter=CQ]")?.getAttribute("d").match(/L/g)??[]).length>=10',60)
            pause(500)
            assert js('!document.querySelector(".m02-page [aria-busy=true]")')
            report['checks'].append('30min result zooms to 100ms with fresh CQ/SQ/GCI F0 view before enabling export')
            w.view.grab().save(str(out/'m02-30min.png'))
        assert all(hashlib.sha256((inputs/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
        report['success']=True
    except Exception as error:
        report['error']=str(error);w.view.grab().save(str(out/'failed.png'));raise
    finally:
        w.closing=True;w.close();app.processEvents();report['service_exit_code']=w.service.exit_code
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8');print(out,flush=True)

if __name__=='__main__':main()
