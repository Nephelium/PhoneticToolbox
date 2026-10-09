"""M07-R1 actual Windows Qt: hidden test window, native gestures, muted audio."""
import hashlib
import json
import os
import time
os.environ.setdefault('QT_QPA_PLATFORM','windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --disable-gpu-compositing --mute-audio')
from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from verify_m07_wiring import setup,ROOT


def main():
    out,db,cache=setup();inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    for i in (0,1):(inputs/f'input{i}.wav').write_bytes((ROOT/f'output/validation/m07/baseline/round1/input{i}.wav').read_bytes())
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    register_scheme();app=QApplication(['M07-R1-owned-QA'])
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
    w.resize(1920,1080);w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.show();w.page.setAudioMuted(True)
    QFileDialog.getExistingDirectory=lambda *a,**k:str(saved if '结果' in str(a) else inputs)
    report=dict(success=False,checks=[],layouts=[],scope='Windows actual Qt hidden; production built frontend; native QTest pointer/keyboard; synthetic audio; muted WebAudio',schema_applied=[])
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda result:(box.append(result),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        if not box:raise RuntimeError('JS timeout')
        return box[0]
    def until(code,seconds=70):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' '+str(js('document.body.innerText.slice(-5000)')))
    def click(text):
        assert js('(()=>{const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text)+');if(!e||e.disabled)return false;e.click();return true})()'),text
        pause(100)
    def fill(label,value):
        js('(()=>{const e=document.querySelector('+json.dumps('[aria-label="'+label+'"]')+');e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}))})()')
    def native_click(selector):
        rect=js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.scrollIntoView({block:"nearest"});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
        target=w.view.focusProxy() or w.view;QTest.mouseClick(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(rect['x']),round(rect['y'])));pause(200)
    def idle():until('document.querySelector(".m07-page").getAttribute("aria-busy")==="false"')
    def curves(count):until('document.querySelectorAll(".f0-plot path[data-curve=synthesis]").length==='+str(count))
    try:
        until('!!document.querySelector("nav")');click('发声类型合成');until('!!document.querySelector(".m07-page")')
        click('打开音频目录');until('document.querySelector("select[aria-label=源音频]").options.length===3')
        for label,index in [('源音频',1),('目标音频',2)]:fill(label,js('document.querySelector('+json.dumps('select[aria-label="'+label+'"]')+').options['+str(index)+'].value'))
        until('[...document.querySelectorAll("button")].some(b=>b.textContent.trim()==="提取 F0"&&!b.disabled)');click('提取 F0');idle()
        until('document.body.innerText.includes("分析完成，控制点尚未修改")');fill('连续统步数',3);click('生成全部六组');idle();curves(3)
        assert js('document.querySelectorAll(".result-row").length')==1
        assert js('[...document.querySelectorAll(".history-select")].filter(e=>e.textContent.includes(" · 3 步")).length')==6
        js('window.audioStarts=[];window.nativeEvents=[];for(const type of ["pointerdown","keydown"]){document.addEventListener(type,e=>window.nativeEvents.push({type:e.type,key:e.key,trusted:e.isTrusted}),true);}window.sourceFactory=AudioContext.prototype.createBufferSource;AudioContext.prototype.createBufferSource=function(){const n=window.sourceFactory.call(this),s=n.start;n.start=function(w,o,d){window.audioStarts.push({offset:o,duration:d,frames:this.buffer.length});return s.call(this,w,o,d);};return n;}')
        native_click('.result-audios button:nth-child(3)');curves(1);until('window.audioStarts.length===1')
        assert 'step02.wav' in js('document.querySelector(".current-audio").textContent')
        single=js('window.audioStarts[0]');click('停止')
        native_click('.result-audios button:first-child');curves(3);until('window.audioStarts.length===2')
        whole=js('window.audioStarts[1]');assert whole['frames']==single['frames']*3
        assert js('[...document.querySelectorAll(".f0-plot path[data-curve=synthesis]")].map(e=>Number(e.dataset.axisEnd))')==[100,100,100]
        click('停止')
        # Exclusive selection ownership set by direct audition also supports Space.
        target=w.view.focusProxy() or w.view
        native_click('.synthesized-plot h2');QTest.keyClick(target,Qt.Key.Key_Space);pause(250);until('window.audioStarts.length===3');QTest.keyClick(target,Qt.Key.Key_Space);pause(150)
        assert all(e['trusted'] for e in js('window.nativeEvents'))
        report['checks'].append('QWebChannel real six groups; native single-step/whole click starts correct PCM; whole overlays three independent curves; one Space restart after direct audition')
        js('[...document.querySelectorAll(".history-select")].find(e=>e.textContent.includes("源到目标 · 仅发声类型变化")).click()');curves(3)
        assert '源到目标 · 仅发声类型变化' in js('document.querySelector(".result-row").textContent')
        js('[...document.querySelectorAll(".history-select")].find(e=>e.textContent.includes("提取 F0")).click()')
        assert js('document.querySelectorAll(".result-row").length')==0
        js('[...document.querySelectorAll(".history-select")].find(e=>e.textContent.includes("目标到源 · F0 与发声类型同时变化")).click()');curves(3)
        click('输出位置');click('保存完整组');until('document.body.innerText.includes("已保存本组完整文件和参数清单")')
        assert len(list(saved.glob('M07-*/*.wav')))==4
        click('刷新任务历史');until('!document.querySelector(".history-heading button").disabled');curves(3)
        report['checks'].append('history task grouping, analysis selection without result, real complete-group save and refresh selection preservation')
        for width,height in [(1920,1080),(1440,900),(1280,800)]:
            w.resize(width,height);pause(180)
            for theme in ('light','dark'):
                js('document.documentElement.dataset.theme='+json.dumps(theme));w.view.repaint();pause(1000)
                geo=js('[...document.querySelectorAll(".m07-plots>.module-section")].map(e=>{const r=e.getBoundingClientRect();return {label:e.getAttribute("aria-label"),x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom}})')
                assert abs(geo[0]['y']-geo[1]['y'])<2 and abs(geo[2]['y']-geo[3]['y'])<2
                if width==1920:assert all(g['bottom']<=js('innerHeight') for g in geo)
                w.view.grab().save(str(out/f'r1-qt-{width}-{theme}.png'));report['layouts'].append(dict(width=width,height=height,theme=theme,actual_theme=js('document.documentElement.dataset.theme'),background=js('getComputedStyle(document.querySelector(".module-section")).backgroundColor'),geometry=geo))
        report['audio_starts']=js('window.audioStarts');report['native_events']=js('window.nativeEvents')
        assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        report['checks'].append('six hidden native light/dark layouts; source bytes unchanged; no physical device or DPI claims');report['success']=True
    finally:
        (out/'r1-qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        js('window.onbeforeunload=null');w.close();pause(300);app.quit();print(out,flush=True)


if __name__=='__main__':main()
