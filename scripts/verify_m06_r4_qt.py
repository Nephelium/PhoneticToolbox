"""R4 extension to the actual Qt harness; no production environment changes."""
import hashlib
import io
import json
import sqlite3
import numpy as np
from PyQt6.QtWidgets import QFileDialog
from verify_m06_wiring import read


def verify(window,out,db,js,click,until,pause,report):
    def set_value(label,value):
        selector=json.dumps('[aria-label="'+label+'"]')
        js('(()=>{const e=document.querySelector('+selector+');e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event(e.tagName==="SELECT"?"change":"input",{bubbles:true}))})()')
        pause(100)
    def latest():
        with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as conn:
            job=conn.execute('SELECT id FROM jobs ORDER BY created_at DESC LIMIT 1').fetchone()[0]
        task=window.service.get('/api/v1/jobs/'+job);assert task['state']=='succeeded',task
        values={f['name']:read(window.service,f) for f in task['result_manifest']['files']}
        return job,values,json.loads(values['m06.ptb.json'])
    report['r4_tasks']=[];report['r4_layouts']=[]
    window.resize(1920,1000);pause(150)
    set_value('合成方法','world');set_value('F0 提取算法','harvest');click('提取参数')
    until('document.body.innerText.includes("F0 提取完成")')
    for method in ['world','psola']:
        set_value('合成方法',method);set_value('重合成音高','curve')
        set_value('曲线覆盖',220);click('应用覆盖');set_value('总时长',.9);click('应用时长');pause(100)
        if method=='world':set_value('谱包络频率比例',1.1);set_value('非周期幅度比例',.8)
        else:set_value('F0 提取算法','praat_cc')
        click('重合成音频');until('document.querySelector(".m06-page").getAttribute("aria-busy")==="false"')
        assert '合成完成' in js('document.body.innerText')
        job,values,meta=latest();assert meta['computation_revision']=='m06-'+method+'/1'
        assert meta['config']['curves']['F0']['override']==220
        assert meta['sample_count']==14400
        with np.load(io.BytesIO(values['analysis.npz']),allow_pickle=False) as data:
            assert data['target_f0_hz'].shape==data['target_times_s'].shape
        saved=out/('r4-'+method+'-saved');saved.mkdir()
        QFileDialog.getExistingDirectory=lambda *a,**k:str(saved)
        click('导出音频');until('document.body.innerText.includes("已导出合成结果")')
        for name,raw in values.items():assert (saved/name).read_bytes()==raw,name
        report['r4_tasks'].append(dict(method=method,job=job,diagnostics=meta['diagnostics'],
                                      files={n:hashlib.sha256(v).hexdigest() for n,v in values.items()}))
        for width,height in [(1920,1000),(1440,900),(900,700)]:
            window.resize(width,height);pause(180)
            for theme in ['light','dark']:
                click('设置');pause(100);click('浅色' if theme=='light' else '深色');pause(100);click('语音合成');pause(350)
                m=js('''(()=>{const r=document.querySelector('.m06-page'),b=e=>{const v=e.getBoundingClientRect();return {x:v.x,y:v.y,w:v.width,b:v.bottom}};return {root:b(r),bar:b(r.querySelector('.m06-transport')),work:b(r.querySelector('.module-workbench')),left:b(r.querySelector('.workbench-left')),center:b(r.querySelector('.workbench-center')),right:b(r.querySelector('.workbench-right')),overflow:r.scrollWidth-r.clientWidth,players:r.querySelectorAll('.transport-controls').length}})()''')
                assert m['overflow']<=1 and m['players']==1,m
                assert m['bar']['b']<=m['root']['b']+1 and m['bar']['y']>=m['work']['b']-1,m
                if m['work']['w']<=1060:assert m['right']['y']>=max(m['left']['b'],m['center']['b'])-1,m
                background=js('getComputedStyle(document.documentElement).getPropertyValue("--app").trim()')
                if theme=='dark':assert background!=report['r4_layouts'][-1]['background']
                report['r4_layouts'].append(dict(method=method,width=width,height=height,theme=theme,background=background,measurement=m))
                if width!=1440:window.view.grab().save(str(out/f'r4-qt-{method}-{width}-{theme}.png'))
        window.resize(1920,1000);pause(150)
    report['checks'].append('R4 actual Qt WORLD/PSOLA edited F0 and duration, 2 native four-file saves byte readback, 12 layouts and source metadata')
