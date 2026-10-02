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
from verify_m06_wiring import setup,ROOT


def main():
    out,db,cache=setup();inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    (inputs/'合成.wav').write_bytes((ROOT/'tests/fixtures/m06/source.wav').read_bytes())
    register_scheme();app=QApplication(['M06-owned-QA'])
    window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
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
        pause(250);window.view.grab().save(str(out/'qt.png'))
        report['success']=True
    finally:
        (out/'qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        js('window.onbeforeunload=null');window.close();pause(300);app.quit();print(out)


if __name__=='__main__':main()
