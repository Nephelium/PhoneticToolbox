"""M08-R2: actual Qt native keys, hidden owned window, synthetic audio only."""
import hashlib
import json
import os
import time
os.environ.setdefault('QT_QPA_PLATFORM','windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --disable-gpu-compositing --mute-audio --autoplay-policy=no-user-gesture-required')
from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication,QFileDialog
from scipy.io import wavfile
from ptb_desktop.host import Workbench,register_scheme
from verify_m08_wiring import setup,ROOT


def main():
    out,db,cache=setup();inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    source=inputs/'合成 ɑ̃˥.wav';source.write_bytes((ROOT/'output/validation/m08-wiring/input.wav').read_bytes())
    original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    register_scheme();app=QApplication(['M08-R2-owned-QA'])
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
    w.resize(1800,1100);w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.show()
    choices=[]
    def directory(*a,**k):
        target=saved if '结果' in str(a) else inputs;choices.append(str(target));return str(target)
    QFileDialog.getExistingDirectory=directory
    report=dict(success=False,checks=[],layouts=[],flags=os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS'),scope='Windows native Qt keys; hidden built-bundle window; synthetic audio; no physical IME/hardware/DPI',schema_applied=[])
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
    def keys(label,value):
        selector=json.dumps('[aria-label="'+label+'"]')
        box=js('(()=>{const e=document.querySelector('+selector+');e.scrollIntoView({block:"center"});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
        target=w.view.focusProxy() or w.view
        QTest.mouseClick(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(box['x']),round(box['y'])))
        QTest.keyClick(target,Qt.Key.Key_A,Qt.KeyboardModifier.ControlModifier)
        for char in value:
            QTest.keyClicks(target,char,Qt.KeyboardModifier.NoModifier,30);pause(100)
            if char=='.':
                pause(1200)
                assert js('document.querySelector('+selector+').value').endswith('.'),label
        pause(1200);actual=js('document.querySelector('+selector+').value');assert actual==value,(label,value,actual)
    def idle():until('document.querySelector(".m08-page").getAttribute("aria-busy")==="false"')
    def screenshot(name):
        w.view.repaint();pause(1000);w.view.grab();pause(500);app.processEvents();w.view.grab().save(str(out/name))
    try:
        until('!!document.querySelector("nav")');click('变速变调');until('!!document.querySelector(".m08-page")');click('打开音频目录')
        until('document.querySelectorAll(".m08-page select option").length>1')
        js('(()=>{const e=document.querySelector(".m08-page select");e.value=e.options[1].value;e.dispatchEvent(new Event("change",{bubbles:true}))})()')
        until('!!document.querySelector("svg.m08-curve")');idle()
        assert not js('!!document.querySelector("svg.history-plot")')
        screenshot('m08-r2-qt-empty.png')
        keys('语速倍率','0.8');keys('F0 下限','50.5');keys('F0 上限','350.5');click('应用范围');keys('参考线 Hz','200.25');click('添加参考线')
        click('批量变速变调');keys('音高倍率','1.25');keys('音高偏移 Hz','-2.5');click('单文件与基频')
        click('编辑拐点表');click('添加行');keys('时间 1','0.3')
        fill('时间 0','0.1');fill('时间 2','0.9')
        for i in range(3):fill('频率 '+str(i),'120,180');fill('连接 '+str(i),'full')
        screenshot('m08-r2-qt-decimal.png');click('保存更改')
        assert js('[...document.querySelectorAll(".point-summary span")].some(e=>e.textContent.startsWith("0.3 s"))')
        report['checks'].append('native Qt pointer focus / real key events preserve partial decimal through polling for all seven numeric fields')
        click('合成当前视野');until('document.querySelectorAll(".history li").length===1 && document.querySelector(".m08-page").innerText.includes("输出 1.250 s")');idle()
        click('批量生成');until('document.querySelectorAll(".history li").length===9');idle();assert not list(saved.iterdir())
        click('删除本批次音频')
        boxes=js('[...document.querySelectorAll("dialog[open] .file-list input")].map(e=>{const r=e.getBoundingClientRect();return {width:r.width,height:r.height}})')
        assert len(boxes)==9 and all(b['width']==16 and b['height']==16 for b in boxes),boxes
        screenshot('m08-r2-qt-checkboxes.png');click('取消');report['checks'].append('nine short/long filename checkboxes stay 16 x 16 pixels, no real history deleted')
        click('批量保存');click('保存所选音频');until('!document.querySelector("dialog[open]")');idle()
        assert js('document.querySelector(".save-location").innerText.includes('+json.dumps(str(saved))+')')
        assert len(list(saved.glob('*.wav')))==9 and choices.count(str(saved))==1
        names=js('[...document.querySelectorAll(".save-location li")].map(e=>e.textContent)')
        audit=[]
        for file in saved.glob('*.wav'):
            rate,data=wavfile.read(file);assert rate==16000 and data.dtype.name=='int16' and file.name in names
            audit.append(dict(name=file.name,frames=len(data),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        report['saved']=audit;report['directory']=str(saved)
        report['checks'].append('native one-directory export: nine actual WAV names and resolved directory match persistent on-screen save location')
        for width,height in [(1800,1100),(1280,800),(960,760)]:
            for theme in ('light','dark'):
                w.resize(width,height);js('document.documentElement.dataset.theme='+json.dumps(theme));pause(400)
                box=js('(()=>{const e=document.querySelector(".m08-page");return {client:e.clientWidth,scroll:e.scrollWidth,panels:[...document.querySelector(".plots").children].map(n=>{const r=n.getBoundingClientRect();return {w:r.width,h:r.height}})}})()')
                assert box['scroll']<=box['client']+2,box
                if width==1800:
                    a=box['panels'];assert abs(a[0]['h']-a[1]['h'])<1 and abs(a[2]['h']-a[3]['h'])<1 and abs(a[0]['w']-a[3]['w'])<1
                report['layouts'].append(dict(width=width,height=height,theme=theme,**box));screenshot(f'm08-r2-qt-{width}-{theme}.png')
        assert hashlib.sha256(source.read_bytes()).hexdigest()==original_hash
        report['checks'].append('six actual Qt layouts, light/dark, equal desktop panel pairs and no horizontal overflow; source hash unchanged')
        report['success']=True
    finally:
        (out/'m08-r2-qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        w.closing=True;w.close();pause(300);app.quit();print(out,flush=True)


if __name__=='__main__':main()
