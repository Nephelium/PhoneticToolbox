"""M01-R3 actual Qt/native grants, scientific tasks, exports and parent slicing."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --mute-audio')
import hashlib
import io
import json
import math
import sqlite3
import time
from pathlib import Path
from uuid import uuid4
import numpy as np
import soundfile as sf
from openpyxl import load_workbook
from PyQt6.QtCore import QEventLoop,QTimer,Qt
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.io.lip import encode_lip
from ptb_worker.store import SQLiteJobStore

ROOT=Path(__file__).resolve().parents[1]


def grid(first='默认',second='test'):
    return f'''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 1
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "word"
        xmin = 0
        xmax = 1
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.4
            text = "{first}"
        intervals [2]:
            xmin = 0.4
            xmax = 1
            text = "{second}"
'''


def snapshot(folder):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}


def read_pair(folder,base):
    wb=load_workbook(folder/(base+'.xlsx'),read_only=True,data_only=False)
    try:excel=[list(row) for row in wb.active.values]
    finally:wb.close()
    conn=sqlite3.connect(':memory:')
    try:
        conn.deserialize((folder/(base+'.ptb.sqlite')).read_bytes())
        assert conn.execute('PRAGMA integrity_check').fetchone()==('ok',)
        columns=[r[1] for r in conn.execute('PRAGMA table_info(params)')]
        rows=[list(row) for row in conn.execute('SELECT * FROM params ORDER BY rowid')]
    finally:conn.close()
    assert columns==excel[0] and len(rows)==len(excel)-1
    for a,b in zip(rows,excel[1:]):
        for x,y in zip(a,b):
            if isinstance(x,(int,float)):
                assert math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12)
            else:assert (x or '')==(y or '')
    return columns,rows


def main():
    out=ROOT/'output/validation/m01-r3'/('qt-'+uuid4().hex)
    inputs,grids,lips,exports=[out/name for name in ('inputs','grids','lips','exports')]
    for folder in (inputs,grids,lips,exports):folder.mkdir(parents=True)
    t=np.arange(16000)/16000
    for index,name in enumerate(('a','b')):
        sf.write(inputs/(name+'.wav'),.25*np.sin(2*np.pi*(150+index*30)*t),16000,subtype='PCM_16')
        (inputs/(name+'.TextGrid')).write_text(grid(),encoding='utf-8')
        (grids/(name+'.TextGrid')).write_text(grid('甲','乙'),encoding='utf-8')
        for folder,value in ((inputs,1.),(lips,5.+index)):
            (folder/(name+'.lip.json')).write_bytes(encode_lip({'relative_times':[0.,1.],
                'area':[value,value],'outer_width':[2.,2.],'open':[3.,3.],'circularity':[4.,4.],
                'metadata':{'lip_manual_offset':0.}}))
    before=[snapshot(folder) for folder in (inputs,grids,lips)]
    (exports/'a.xlsx').write_bytes(b'previous a result')
    (exports/'b.ptb.sqlite').write_bytes(b'previous b result')
    db=ROOT/'output/validation/p06/local-state.sqlite3';SQLiteJobStore(db).check_schema()
    config=json.loads((ROOT/'output/validation/m01/workbench-local.json').read_text('utf-8'))
    register_scheme();app=QApplication(['M01-R3-owned-offscreen-QA'])
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,
        local_files_root=ROOT/'output/validation/m01'/config['cache'],
        reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe',vocal_profile=out/'vocal')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen,True);w.resize(1800,1000);w.show()
    chosen=[inputs];QFileDialog.getExistingDirectory=lambda *a,**k:str(chosen[0]) if chosen[0] else ''
    report={'success':False,'checks':[],'layouts':[],'schema_applied':[],
            'scope':'Windows actual Qt offscreen/native files/local scientific tasks; synthetic 1s audio, annotations and lip vectors'}
    batches=[];original_request=w.service.request
    def request(path,method='GET',body=None):
        result=original_request(path,method,body)
        if path=='/api/v1/jobs/batches/create':batches.append(result['id'])
        return result
    w.service.request=request
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();values=[];w.page.runJavaScript(code,lambda value:(values.append(value),loop.quit()))
        QTimer.singleShot(6000,loop.quit);loop.exec()
        if not values:raise RuntimeError('JS timeout')
        return values[0]
    def until(code,seconds=60):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        details=js('({pages:[...document.querySelectorAll("main>.module-frame")].map(e=>({label:e.getAttribute("aria-label"),classes:e.className})),waves:document.querySelectorAll(".wave-track").length,statuses:[...document.querySelectorAll("[role=status]")].map(e=>e.textContent),alerts:[...document.querySelectorAll("[role=alert]")].map(e=>e.textContent)})')
        raise AssertionError(code+' / '+str(details))
    def click(text):
        code='[...document.querySelectorAll("button")].find(b=>b.offsetParent&&!b.disabled&&(b.getAttribute("aria-label")??b.textContent.trim())==='+json.dumps(text)+')'
        until('!!'+code);js(code+'.click()');pause()
    def directory(button,folder):
        chosen[0]=folder;click(button)
        until('![...document.querySelectorAll("button")].some(b=>b.textContent.trim()==="正在读取…")')
    def selected_name(kind):
        return js('document.querySelector('+json.dumps('select[aria-label='+kind+'关联]')+').selectedOptions[0].textContent')
    def result_json(job):
        file=next(f for f in job['result_manifest']['files'] if f['name'].endswith('.ptb.json'))
        raw=b''.join(w.service.binary(f'/api/v1/jobs/local-results/{file["id"]}?offset={offset}&size={min(1048576,file["size_bytes"]-offset)}') for offset in range(0,file['size_bytes'],1048576))
        assert hashlib.sha256(raw).hexdigest()==file['sha256']
        return json.loads(raw)
    try:
        until('document.querySelector(".host-badge")?.textContent==="本地桌面"');click('参数估计');directory('选择音频目录',inputs)
        until('document.querySelectorAll(".m01-file-entry").length===2');js('document.querySelector(".m01-file-list .file-row").click()')
        until('!!document.querySelector(".m01-slicing")&&!!document.querySelector(".wave-track")')
        assert selected_name('TextGrid')=='a.TextGrid' and selected_name('唇形')=='a.lip.json'
        directory('选择TextGrid目录',grids);until('document.querySelector(".textgrid-interval")?.textContent.includes("甲")')
        directory('选择唇形目录',lips);assert selected_name('唇形')=='a.lip.json'
        directory('选择TextGrid目录',None);assert js('document.querySelector(".textgrid-interval")?.textContent.includes("甲")')
        click('TextGrid使用音频目录');until('document.querySelector(".textgrid-interval")?.textContent.includes("默认")');directory('选择TextGrid目录',grids)
        report['checks'].append('Native default same-folder associations; independent TextGrid/lip overrides; cancellation and reset preserve other links')
        click('选择输出参数');click('全不选')
        for key in ('LipArea','LipWidth','LipOpen','LipCirc'):js('document.querySelector(".parameter-grid input[value='+key+']").click()')
        click('应用到草稿')
        js('[...document.querySelectorAll("label")].find(e=>e.textContent.includes("结果与WAV同目录")).querySelector("input").click()')
        directory('选择结果目录',exports);click('开始全列表分析')
        until('document.querySelector(".m01-page")?.textContent.includes("已保存4个结果文件")',180)
        batch=w.service.get('/api/v1/jobs/batches/'+batches[-1]);assert batch['summary']['complete']
        jobs=[w.service.get('/api/v1/jobs/'+item['job_id']) for item in batch['summary']['items']]
        frames=[]
        for index,job in enumerate(jobs):
            wire=result_json(job);assert wire['metadata']['computation_revision']=='acoustic/2'
            columns,rows=read_pair(exports,('a' if index==0 else 'b')+' (2)')
            numeric={c['key']:c for c in wire['numeric']};labels={c['key']:c['label'] for c in wire['numeric']};texts={c['key']:c['values'] for c in wire['text']}
            assert columns==[labels.get(key,key) for key in wire['column_order']]
            assert len(rows)==len(wire['times_s'])
            for i,row in enumerate(rows):
                for key,value in zip(wire['column_order'],row):
                    expected=wire['times_s'][i] if key=='Time_s' else texts[key][i] if key in texts else numeric[key]['values'][i]
                    if isinstance(expected,(int,float)):assert math.isclose(value,expected,rel_tol=1e-12,abs_tol=1e-12)
                    else:assert (value or '')==(expected or '')
            assert any(math.isclose(row[columns.index('唇面积')],5.+index,abs_tol=1e-12) for row in rows if row[columns.index('唇面积')] is not None)
            frames.append(len(rows))
        assert not list(exports.glob('*.json'));saved_before=snapshot(exports);click('保存已完成结果')
        until('document.querySelector(".m01-page")?.textContent.includes("已保存4个结果文件")');assert snapshot(exports)==saved_before
        report['checks'].append('Two actual scientific tasks; four exported artifacts with paired (2) names; every XLSX/SQLite time/value/label equals internal JSON; repeat save unchanged')
        report['frames']=frames;report['analysis_batches']=batches[:]
        js('[...document.querySelectorAll("label")].find(e=>e.textContent.includes("同时切分参数结果")).querySelector("input").click()')
        click('保存当前层切分音频')
        until('document.querySelector(".m01-page")?.textContent.includes("已保存6个结果文件")',180)
        batch=w.service.get('/api/v1/jobs/batches/'+batches[-1]);assert batch['summary']['complete']
        segment_job=w.service.get('/api/v1/jobs/'+batch['summary']['items'][0]['job_id']);metadata=result_json(segment_job)
        assert metadata['parent_result_sha256'] and all(s['parameter_status']=='included' for s in metadata['segments'])
        wavs=list(exports.glob('*.wav'));assert len(wavs)==2
        for wav in wavs:
            columns,rows=read_pair(exports,wav.stem);assert rows and 'Source_Time_s' in columns
        assert not list(exports.glob('*.json')) and (exports/'a.xlsx').read_bytes()==b'previous a result' and (exports/'b.ptb.sqlite').read_bytes()==b'previous b result'
        report['checks'].append('Actual parent-result lookup and two annotated segments succeed after JSON is omitted from user exports; six WAV/SQLite/XLSX artifacts; internal segment provenance retained')
        report['export_names']=sorted(p.name for p in exports.iterdir());report['segment_batch']=batches[-1]
        for width,height in ((1800,1000),(900,700),(450,700)):
            w.resize(width,height);pause(300)
            assert js('(()=>{const e=document.querySelector(".m01-directory-bar");return e.scrollWidth<=e.clientWidth+1;})()')
            report['layouts'].append({'width':width,'height':height})
        w.resize(1440,900)
        for mode in ('light','dark'):
            js('document.documentElement.dataset.theme='+json.dumps(mode));pause(300);w.view.grab().save(str(out/('m01-'+mode+'.png')))
        click('参数显示');directory('选择音频目录',inputs);click('a.wav')
        until('!!document.querySelector(".m02-page .wave-track")&&!document.querySelector(".m02-page>p[role=status]")')
        directory('选择参数目录',exports)
        until('[...document.querySelectorAll(".m02-page .signal-panel>label select option")].some(o=>o.textContent==="a (2).ptb.sqlite")&&!document.querySelector(".m02-page>p[role=status]")')
        for name in ('a (2).ptb.sqlite','a (2).xlsx'):
            js('(()=>{const e=document.querySelector(".m02-page .signal-panel>label select");e.value=[...e.options].find(o=>o.textContent==='+json.dumps(name)+').value;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            until('document.querySelectorAll(".m02-parameters input").length>=4&&!document.querySelector(".m02-page>p[role=status]")')
            assert not js('document.querySelector(".m02-page .error-banner")?.textContent')
        click('新建图窗');js('document.querySelector(".m02-parameters input").click()');click('将 1 项分配到图窗')
        until('!!document.querySelector(".m02-page .parameter-curve")?.getAttribute("d")')
        report['checks'].append('Actual M02 native reader opens both exported SQLite and XLSX, then draws a parameter curve')
        w.view.grab().save(str(out/'m02-exported-pair.png'))
        assert [snapshot(folder) for folder in (inputs,grids,lips)]==before
        report['sources_unchanged']=True;report['success']=True
    except Exception as error:
        report['error']=str(error);w.view.grab().save(str(out/'failed.png'));raise
    finally:
        w.closing=True;w.close();app.processEvents();report['service_exit_code']=w.service.exit_code
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8');print(out,flush=True)


if __name__=='__main__':main()
