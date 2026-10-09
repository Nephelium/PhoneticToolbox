"""Actual Qt host/QWebChannel and built frontend on synthetic files; no DDL."""
import json
import os
import time
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu')
from PyQt6.QtCore import QEventLoop,QTimer
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from verify_m06_wiring import setup,ROOT,read


def main():
    out,db,cache=setup();inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    (inputs/'合成.wav').write_bytes((ROOT/'tests/fixtures/m06/source.wav').read_bytes())
    register_scheme();app=QApplication(['M06-owned-QA'])
    manifest=json.loads((ROOT/'resources/manifests/acoustic.json').read_text('utf8'))
    window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile',reaper_binary=ROOT/manifest['resources'][0]['validation_source'])
    window.resize(1920,1000);window.show()
    QFileDialog.getExistingDirectory=lambda *a,**k:str(saved if '结果' in str(a) else inputs)
    report=dict(success=False,checks=[],scope='Windows actual Qt offscreen; built frontend; synthetic audio')
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[]
        window.page.runJavaScript(code,lambda result:(box.append(result),loop.quit()))
        QTimer.singleShot(5000,loop.quit);loop.exec();return box[0] if box else None
    def until(code,seconds=60):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' '+str(js('document.body.innerText.slice(-4000)')))
    def click(text):js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text)+')?.click()')
    try:
        until('!!document.querySelector("nav")');click('语音合成');until('!!document.querySelector("[aria-label=语音合成工作区]")')
        if os.environ.get('M06_R5_ONLY'):
            from verify_m06_r5_qt import verify
            verify(window,out,db,js,click,until,pause,report)
            report['success']=True
            return
        js('(()=>{const e=document.querySelector("[aria-label=总时长]");e.value="0.3";e.dispatchEvent(new Event("input",{bubbles:true}))})()');click('应用时长')
        js('(()=>{const e=document.querySelector("input.ipa-text");e.value="a i";e.dispatchEvent(new Event("input",{bubbles:true}))})()');click('生成元音')
        until('document.body.innerText.includes("元音曲线已生成")')
        def preset(name):
            js('(()=>{const e=document.querySelector("[aria-label=发声类型预设]");e.value='+json.dumps(name)+';e.dispatchEvent(new Event("change",{bubbles:true}))})()')
            click('应用预设');pause(50);click('应用并覆盖');pause(100)
        preset('假声')
        assert js('document.querySelector('+json.dumps('[aria-label="F0 下限"]')+').value')=='50'
        preset('假声')
        assert js('document.querySelector('+json.dumps('[aria-label="F0 下限"]')+').value')=='50'
        click('合成音频');until('document.body.innerText.includes("合成完成")');click('导出音频');until('document.body.innerText.includes("已导出合成结果")')
        assert len(list(saved.glob('*.wav')))==1
        report['checks'].append('Qt QWebChannel, generated curves, actual durable synthesis and native WAV/snapshot export')
        meta=json.loads((saved/'m06.ptb.json').read_text('utf8'))
        assert meta['computation_revision']=='klatt/2'
        assert meta['config']['f0_transform']['offset_hz']==180
        assert meta['config']['curves']['F0']['points']==[[0.0,300.0],[0.3,300.0]]
        preset('嘎裂');assert '下移 50.0 Hz' in js('document.body.innerText')
        preset('耳语');assert js('document.querySelector('+json.dumps('[aria-label="F0 下限"]')+').value')=='50'
        report['checks'].append('R2 Qt: noncumulative falsetto, real shifted F0 in exported synthesis snapshot, creaky shift and neutral restoration')

        report['layouts']=[]
        for width,height in [(1920,1000),(2560,1400)]:
            window.resize(width,height);pause(200)
            for theme in ['light','dark']:
                js('document.documentElement.dataset.theme='+json.dumps(theme));pause(100)
                for mode in ['波形','语谱图']:
                    click(mode);pause(100)
                    for key in ['Home','End']:
                        js('document.querySelectorAll(".m06-page [role=separator]").forEach(e=>e.dispatchEvent(new KeyboardEvent("keydown",{key:'+json.dumps(key)+',bubbles:true})))');pause(150)
                        for collapsed in [False,True]:
                            if collapsed:click('收起合成与任务')
                            pause(100)
                            m=js("""(()=>{const root=document.querySelector('.m06-page'),s=root.querySelector('.curve-editor svg'),b=(root.querySelector('.wave-track svg')??root.querySelector('.spectrum-plot')).getBoundingClientRect(),p=s.createSVGPoint();const x=n=>{p.x=n;p.y=0;return p.matrixTransform(s.getScreenCTM()).x};return {edges:[x(72)-b.left,x(s.viewBox.baseVal.width-24)-b.right],columns:[root,...root.querySelectorAll('.workbench-left,.workbench-center,.workbench-right-body')].filter(e=>e.clientWidth).map(e=>[e.clientWidth,e.scrollWidth,e.clientHeight,e.scrollHeight])}})()""")
                            assert m and all(abs(v)<=1.1 for v in m['edges']),m
                            assert all(sw<=w+1 and sh<=h+1 for w,sw,h,sh in m['columns']),m
                            report['layouts'].append(dict(width=width,height=height,theme=theme,mode=mode,key=key,collapsed=collapsed,measurement=m))
                            if collapsed:click('展开合成与任务')
        window.resize(1920,1000);pause(200)
        js('(()=>{const e=document.querySelector("[aria-label=总时长]");e.value="0.6";e.dispatchEvent(new Event("input",{bubbles:true}))})()');click('应用时长');pause(100)
        assert js('document.querySelector(".curve-editor svg").dataset.end')=='0.6'
        report['checks'].append('32 actual Qt layout/axis combinations; existing audio duration changes immediately')
        click('打开音频目录');until('document.querySelector(".source-picker").options.length>1')
        js('(()=>{const e=document.querySelector(".source-picker");e.selectedIndex=1;e.dispatchEvent(new Event("change",{bubbles:true}))})()')
        until('document.body.innerText.includes("音频已加载")');click('提取参数')
        until('document.body.innerText.includes("参数提取完成")')
        report['r3_extractions']=[]
        import sqlite3
        for method in ['praat_cc','praat_ac','reaper']:
            js('(()=>{const e=document.querySelector("[aria-label=F0提取算法]")||document.querySelector('+json.dumps('[aria-label="F0 提取算法"]')+');e.value='+json.dumps(method)+';e.dispatchEvent(new Event("change",{bubbles:true}))})()')
            pause(80);click('提取参数');pause(150)
            until('document.querySelector(".m06-page").getAttribute("aria-busy")==="false"')
            assert '参数提取完成' in js('document.body.innerText')
            with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as conn:
                job_id=conn.execute("SELECT id FROM jobs ORDER BY created_at DESC LIMIT 1").fetchone()[0]
            task=window.service.get('/api/v1/jobs/'+job_id)
            assert task['state']=='succeeded',task
            file=next(f for f in task['result_manifest']['files'] if f['name']=='m06.ptb.json')
            result=json.loads(read(window.service,file))
            assert result['config']['f0_method']==method
            assert result['diagnostics']['actual_f0_backend']==('native_reaper' if method=='reaper' else method)
            report['r3_extractions'].append(dict(method=method,job=job_id,diagnostics=result['diagnostics']))
        click('语谱图');pause(100)
        report['r3_layouts']=[]
        for width,height in [(1920,1000),(900,700)]:
            window.resize(width,height);pause(200)
            m=js("""(()=>{const r=document.querySelector('.m06-page'),b=e=>{const p=e.getBoundingClientRect();return [p.x,p.y,p.width,p.height,p.bottom]};return {root:b(r),work:b(r.querySelector('.module-workbench')),left:b(r.querySelector('.workbench-left')),center:b(r.querySelector('.workbench-center')),right:b(r.querySelector('.workbench-right')),bar:b(r.querySelector('.m06-transport')),label:b(r.querySelector('.preview-controls label')),select:b(r.querySelector('.preview-controls select')),players:r.querySelectorAll('.transport-controls').length,volume:!!r.querySelector('.m06-transport .volume')}})()""")
            assert m['players']==1 and m['volume']
            assert m['bar'][1]>=m['work'][4]-1 and m['bar'][4]<=m['root'][4]+1,m
            assert abs(m['label'][1]-m['select'][1])<1,m
            if m['work'][2]<=1060:assert m['right'][1]>=max(m['left'][4],m['center'][4])-1,m
            report['r3_layouts'].append(m)
            window.view.grab().save(str(out/f'r3-qt-{width}.png'))
        report['checks'].append('R3 actual Qt CC/AC/native REAPER extraction and metadata, shared full-width transport and inline spectrum label at two sizes')

        window.resize(1920,1000);pause(100)
        pause(250);window.view.grab().save(str(out/'qt.png'))
        report['success']=True
    finally:
        (out/'qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        js('window.onbeforeunload=null');window.close();pause(300);app.quit();print(out)


if __name__=='__main__':main()
