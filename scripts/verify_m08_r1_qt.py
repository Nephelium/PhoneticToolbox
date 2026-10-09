"""M08-R1 actual Windows Qt, hidden window and muted audio, synthetic input only."""
import hashlib
import json
import os
import time
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --mute-audio --autoplay-policy=no-user-gesture-required')
from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication,QFileDialog
from scipy.io import wavfile
from ptb_desktop.host import Workbench,register_scheme
from verify_m08_wiring import setup,ROOT


def main():
    out,db,cache=setup();inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    source=inputs/'合成 ɑ̃˥.wav';source.write_bytes((ROOT/'output/validation/m08-wiring/input.wav').read_bytes())
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    register_scheme();app=QApplication(['M08-R1-owned-QA'])
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
    w.resize(1800,1100);w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.show()
    choices=[]
    def directory(*a,**k):
        target=saved if '结果' in str(a) else inputs;choices.append(str(target));return str(target)
    QFileDialog.getExistingDirectory=directory
    report=dict(success=False,checks=[],layouts=[],scope='Windows actual Qt hidden; built bundle; muted WebAudio; synthetic input; no hardware/DPI',schema_applied=[])
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda result:(box.append(result),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        if not box:raise RuntimeError('JS timeout')
        return box[0]
    def until(code,seconds=60):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' '+str(js('document.body.innerText.slice(-5000)')))
    def click(text):
        assert js('(()=>{const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text)+');if(!e||e.disabled)return false;e.click();return true})()'),text
    def fill(label,value):
        selector=json.dumps('[aria-label="'+label+'"]')
        js('(()=>{const e=document.querySelector('+selector+');e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}))})()')
    def idle():until('document.querySelector(".m08-page").getAttribute("aria-busy")==="false"')
    def drag(selector,a,b,shift=False):
        box=js('(()=>{const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})()')
        def point(pair):return QPoint(round(box['x']+pair[0]*box['w']),round(box['y']+pair[1]*box['h']))
        target=w.view.focusProxy() or w.view;modifier=Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
        QTest.mousePress(target,Qt.MouseButton.LeftButton,modifier,point(a))
        for i in range(1,21):QTest.mouseMove(target,point((a[0]+(b[0]-a[0])*i/20,a[1]+(b[1]-a[1])*i/20)),10)
        QTest.mouseRelease(target,Qt.MouseButton.LeftButton,modifier,point(b));pause(150)
    def play(text):
        previous=js('window.m08Starts.length');click(text);until(f'window.m08Starts.length>{previous}',10);return js('window.m08Starts.at(-1)')
    try:
        until('!!document.querySelector("nav")');click('变速变调');until('!!document.querySelector(".m08-page")');click('打开音频目录')
        until('document.querySelectorAll(".m08-page select option").length>1')
        js('(()=>{const e=document.querySelector(".m08-page select");e.value=e.options[1].value;e.dispatchEvent(new Event("change",{bubbles:true}))})()')
        until('!!document.querySelector("svg.m08-curve")');idle()
        js('window.m08Starts=[];(()=>{const create=AudioContext.prototype.createBufferSource;AudioContext.prototype.createBufferSource=function(){const node=create.call(this),start=node.start;node.start=function(when,offset,duration){const result=start.call(this,when,offset,duration);window.m08Starts.push({offset,duration,frames:this.buffer.length});return result;};return node;};})()')
        before=js('document.querySelector(".m08-curve .modified").getAttribute("d")')
        drag('.m08-curve',(.25,.35),(.8,.55),True)
        assert js('document.querySelector(".m08-curve .modified").getAttribute("d")')!=before
        fill('语速倍率',.8);click('合成当前视野')
        until('document.querySelectorAll(".history li").length===1 && document.querySelector(".m08-page").innerText.includes("输出 1.250 s")');idle()
        report['checks'].append('Qt pointer hand draw -> actual durable Praat synthesis -> F0 history without save')
        drag('.wave-viewport .wave-track>svg',(.2,.5),(.6,.5))
        a=play('播放当前视野原音');b=play('播放选区');assert abs(a['offset'])<.001 and abs(a['duration']-1)<.002
        assert abs(b['offset']-.2)<.007 and abs(b['duration']-.4)<.007;click('停止')
        a=play('播放合成音');b=play('播放选区');assert abs(a['duration']-1.25)<.002 and abs(b['offset']-.2)<.007;click('停止')
        a=play('试听此版本');assert abs(a['duration']-1.25)<.002;click('停止')
        report['checks'].append('Qt real WebAudio starts for direct view/result/version and exact pointer selection')
        click('编辑拐点表');fill('频率 0','120,180');fill('频率 1','140,200');fill('连接 0','full');fill('连接 1','full');click('保存更改');click('批量生成')
        until('document.querySelectorAll(".history li").length===5');idle();assert not list(saved.iterdir())
        # Preserve an unrelated colliding name. Its bytes must remain unchanged.
        names=js('[...document.querySelectorAll(".history li .result-name")].map(e=>e.title)')
        occupied=saved/names[0];occupied.write_bytes(b'KEEP original unrelated file')
        click('批量保存');click('保存所选音频');until('!document.querySelector("dialog[open]")');idle()
        assert choices.count(str(saved))==1 and len(list(saved.glob('*.wav')))==6
        assert occupied.read_bytes()==b'KEEP original unrelated file'
        audit=[]
        for file in saved.glob('*.wav'):
            if file==occupied:continue
            rate,data=wavfile.read(file);assert rate==16000 and data.dtype.name=='int16' and len(data) in (16000,20000)
            audit.append(dict(name=file.name,rate=rate,frames=len(data),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        assert len(audit)==5;report['saved']=audit
        click('批量保存');click('保存所选音频');until('!document.querySelector("dialog[open]")');idle();assert len(list(saved.glob('*.wav')))==6
        report['checks'].append('Qt one native directory selection, five actual PCM16 WAVs, collision preserved, repeat export reused')
        for width,height in [(1800,1100),(1280,800),(960,760)]:
            w.resize(width,height);pause(500)
            box=js('(()=>{const e=document.querySelector(".m08-page");return {client:e.clientWidth,scroll:e.scrollWidth}})()');assert box['scroll']<=box['client']+2
            report['layouts'].append(dict(width=width,height=height,**box));assert js('!document.querySelector("dialog[open]")');w.view.repaint();w.view.grab();pause(400);app.processEvents();w.view.grab().save(str(out/f'm08-r1-qt-{width}.png'))
        assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
        report['playback']=js('window.m08Starts');report['checks'].append('three Qt layouts, source hash unchanged');report['success']=True
    finally:
        (out/'m08-r1-qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        w.view.grab().save(str(out/'m08-r1-qt-final.png'))
        # This owns a disposable test profile. Use the host cleanup path directly,
        # without presenting a real user's unsaved-draft confirmation dialog.
        w.closing=True;w.close();pause(300);app.quit();print(out,flush=True)


if __name__=='__main__':main()
