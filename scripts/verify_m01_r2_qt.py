"""M01-R2 actual offscreen Qt + owned local tasks, using an authorized WAV copy."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --mute-audio')
import hashlib
import json
import pickle
import time
from pathlib import Path
from uuid import uuid4
import numpy as np
import soundfile as sf
from PyQt6.QtCore import QEventLoop,QTimer
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.io.lip import encode_lip,decode_lip
from ptb_worker.store import SQLiteJobStore

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'output/validation/m01-m02-r1/real-qt-b2322971b8aa4e7b8b86e176e328536a/recursive/甲/15-范皓云-男-1.wav'


def snapshot(folder):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}


def main():
    out=ROOT/'output/validation/m01-r2'/('qt-'+uuid4().hex)
    inputs=out/'inputs';lips=out/'lips';exports=out/'exports'
    for folder in (inputs,lips,exports):folder.mkdir(parents=True)
    source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    duration=sf.info(SOURCE).duration
    for name in ('a','b'):
        (inputs/(name+'.wav')).write_bytes(SOURCE.read_bytes())
        (inputs/(name+'.TextGrid')).write_bytes(SOURCE.with_suffix('.TextGrid').read_bytes())
    vectors={'area':[1.,1.],'outer_width':[2.,2.],'open':[3.,3.],'circularity':[4.,4.]}
    old={'absolute_timestamps':np.array([100.,100.+duration]),**{k:np.array(v) for k,v in vectors.items()},
         'metadata':{'audio_first_frame_time':np.float64(100.),'lip_manual_offset':0.}}
    (inputs/'a.pkl').write_bytes(pickle.dumps(old))
    (inputs/'a_timestamps.pkl').write_bytes(pickle.dumps({'start_time':99.}))
    (lips/'b.lip.json').write_bytes(encode_lip({'relative_times':[0.,duration],**vectors,'metadata':{'lip_manual_offset':0.}}))
    before_input,before_lip=snapshot(inputs),snapshot(lips)
    db=ROOT/'output/validation/p06/local-state.sqlite3';SQLiteJobStore(db).check_schema()
    config=json.loads((ROOT/'output/validation/m01/workbench-local.json').read_text('utf-8'))
    register_scheme();app=QApplication(['M01-R2-owned-offscreen-QA'])
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,
        local_files_root=ROOT/'output/validation/m01'/config['cache'],
        reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe',vocal_profile=out/'vocal')
    w.resize(1800,1000);w.show();folder=[inputs]
    QFileDialog.getExistingDirectory=lambda *a,**k:str(folder[0])
    report={'success':False,'checks':[],'layouts':[],'schema_applied':[],
        'scope':'Windows actual Qt offscreen and local scientific tasks; two byte-identical copies of authorized real WAV; synthetic lip vectors only'}
    imported=[];original_import=w.service.import_input
    def observed_import(raw,name,role):
        if role=='lip':
            data=decode_lip(raw);imported.append({'name':name,'sha256':hashlib.sha256(raw).hexdigest(),
                'metadata':data['metadata'],'open':data['open']})
        return original_import(raw,name,role)
    w.service.import_input=observed_import
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();values=[]
        w.page.runJavaScript(code,lambda value:(values.append(value),loop.quit()))
        QTimer.singleShot(6000,loop.quit);loop.exec()
        if not values:raise RuntimeError('JS timeout')
        return values[0]
    def until(code,seconds=60):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' / '+str(js('document.querySelector(".m01-page")?.textContent')))
    def click(text):
        code='[...document.querySelectorAll("button")].find(b=>b.offsetParent&&!b.disabled&&b.textContent.trim()==='+json.dumps(text)+')'
        until('!!'+code);js(code+'.click()');pause()
    def select(selector,value):
        js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
    try:
        until('document.querySelector(".host-badge")?.textContent==="本地桌面"')
        click('参数估计');click('选择音频目录')
        until('document.querySelectorAll(".m01-file-entry").length===2')
        js('document.querySelector(".m01-file-list .file-row").click()')
        until('!!document.querySelector(".m01-slicing")&&!!document.querySelector(".wave-track")')
        assert js('document.querySelector(".m01-slicing").closest(".workbench-left")!==null')
        assert js('document.querySelector("select[aria-label=唇形关联]").selectedOptions[0].textContent')=='a.pkl'
        assert not js('document.querySelector(".association-help,.m01-intervals")')
        report['checks'].append('Actual authorized WAV/TextGrid reads; left slicing section; old PKL default association without compatibility panel')
        click('选择输出参数')
        for width,height in ((1800,1000),(1440,900)):
            w.resize(width,height);pause(300)
            columns=js('getComputedStyle(document.querySelector(".parameter-grid")).gridTemplateColumns.split(" ").length')
            assert columns==4
            report['layouts'].append({'width':width,'height':height,'columns':columns})
        w.view.grab().save(str(out/'parameters-four-columns.png'))
        click('全不选')
        for key in ('LipArea','LipWidth','LipOpen','LipCirc'):
            js('document.querySelector(".parameter-grid input[value='+key+']").click()')
        click('应用到草稿');w.resize(1800,1000)
        folder[0]=lips;click('关联唇形')
        until('document.querySelector(".m01-page")?.textContent.includes("2 个已匹配")')
        js('document.querySelectorAll(".m01-file-list .file-row")[1].click()')
        until('document.querySelector("select[aria-label=唇形关联]").selectedOptions[0].textContent==="b.lip.json"')
        select('select[aria-label=唇形关联]','');click('刷新列表')
        until('[...document.querySelectorAll("button")].some(b=>b.textContent.trim()==="刷新列表"&&!b.disabled)')
        assert js('document.querySelector("select[aria-label=唇形关联]").value')==''
        click('关联唇形')
        until('document.querySelector("select[aria-label=唇形关联]").selectedOptions[0].textContent==="b.lip.json"')
        report['checks'].append('Actual native directory grants batch-match mixed PKL/JSON; single cancellation survives refresh; bulk restores matched entry')
        js('[...document.querySelectorAll("label")].find(e=>e.textContent.includes("结果与WAV同目录")).querySelector("input").click()')
        folder[0]=exports;click('选择结果目录');click('开始全列表分析')
        until('document.querySelector(".m01-page")?.textContent.includes("已保存6个结果文件")',180)
        payloads=[json.loads(p.read_text('utf-8')) for p in sorted(exports.glob('*.ptb.json'))]
        assert len(payloads)==2 and len(imported)==2
        assert [item['name'] for item in imported]==['a.lip.json','b.lip.json']
        assert imported[0]['metadata']['audio_first_frame_time']==100.
        cols=[{c['key']:c for c in payload['numeric']} for payload in payloads]
        for key in ('LipArea','LipWidth','LipOpen','LipCirc'):
            assert cols[0][key]['values']==cols[1][key]['values']
            assert cols[0][key]['nonfinite']==cols[1][key]['nonfinite']
            assert any(v is not None for v in cols[0][key]['values'])
        assert payloads[0]['times_s']==payloads[1]['times_s']
        report['checks'].append('Actual two-file scientific batch succeeds; six saved files; four lip columns and time axes exactly equal for normalized old/new inputs')
        report['imports']=imported;report['frames']=len(payloads[0]['times_s'])
        for mode in ('light','dark'):
            js('document.documentElement.dataset.theme='+json.dumps(mode));pause(300)
            w.view.grab().save(str(out/('m01-'+mode+'.png')))
        for height in (660,1000):
            w.resize(1440,height);pause(300)
            assert js('(()=>{const e=document.querySelector(".m01-slicing"),p=e.closest(".workbench-left");p.scrollTop=p.scrollHeight;return e.getBoundingClientRect().bottom<=p.getBoundingClientRect().bottom+1;})()')
        assert snapshot(inputs)==before_input and snapshot(lips)==before_lip
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash
        report['sources_unchanged']=True;report['success']=True
    except Exception as error:
        report['error']=str(error);w.view.grab().save(str(out/'failed.png'));raise
    finally:
        w.closing=True;w.close();app.processEvents()
        report['service_exit_code']=w.service.exit_code
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
        print(out,flush=True)


if __name__=='__main__':main()
