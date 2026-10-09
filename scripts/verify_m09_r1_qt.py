"""M09-R1 actual Qt/QWebChannel, hidden owned window and synthetic input."""
import json
import os
import time
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --disable-gpu-compositing --mute-audio')
from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint,QPointF
from PyQt6.QtGui import QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_desktop.m09_capture import crop_corners
from verify_m09_r1 import ROOT,setup


def main():
    out,db,cache=setup('qt');print(out,flush=True);register_scheme();app=QApplication(['M09-R1-owned-QA'])
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
    w.resize(1920,1080);w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.show();w.page.setAudioMuted(True)
    QFileDialog.getExistingDirectory=lambda *a,**k:str(out/('saved' if '结果' in str(a) else 'inputs'))
    report=dict(success=False,checks=[],layouts=[],scope='Windows actual hidden Qt, real QWebChannel jobs/save, synthetic images/audio',schema_applied=[])
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        if not box:raise RuntimeError('JS timeout')
        return box[0]
    def until(code,seconds=90):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' '+str(js('document.body.innerText.slice(-3000)')))
    def click(text):
        assert js('(()=>{const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")||b.textContent.trim())==='+json.dumps(text)+');if(!e||e.disabled)return false;e.click();return true})()'),text
        pause(120)
    def fill(label,value):
        js('(()=>{const e=[...document.querySelectorAll("label")].find(l=>l.textContent.trim().startsWith('+json.dumps(label)+'));const input=e.querySelector("input");input.value='+json.dumps(str(value))+';input.dispatchEvent(new Event("input",{bubbles:true}));input.dispatchEvent(new Event("change",{bubbles:true}));})()')
    target=w.view.focusProxy() or w.view
    def pointer(selector,x,y,down=False,up=False):
        r=js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.scrollIntoView({block:"nearest"});const r=e.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})()')
        p=QPoint(round(r['x']+x*r['w']),round(r['y']+y*r['h']))
        if down:QTest.mousePress(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,p)
        elif up:QTest.mouseRelease(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,p)
        else:QTest.mouseMove(target,p)
        pause(50)
    def draw():
        selector='canvas[aria-label="可缩放频谱画布"]'
        pointer(selector,.3,.4,down=True)
        for i in range(1,8):pointer(selector,.3+i*.035,.4+i*.015)
        pointer(selector,.58,.52,up=True)
        until('document.querySelector(".axis-note").textContent.includes("1 笔")')
    def result():until('!!document.querySelector(".comparison img")&&!document.querySelector(".m09-page[aria-busy=true]")')
    try:
        until('!!document.querySelector("nav")');click('语谱图转音频');until('!!document.querySelector(".original-card")')
        assert js('document.querySelector(".original-card").open')
        assert js('[...document.querySelectorAll("h2")].filter(e=>e.textContent==="原图与四点校正").length')==1
        # Real capture translation, using a synthetic screen pixmap, never user's desktop.
        import ptb_desktop.m09_capture as capture
        screen=QPixmap(str(out/'inputs/skewed.png'))
        capture.capture_spectrogram=lambda window:crop_corners(screen,[QPointF(x,y) for x,y in [(10,5),(110,25),(90,90),(20,80)]],121,101)
        click('截取屏幕');until('document.querySelectorAll(".image-frame circle").length===4')
        fill('时间终点',.2);fill('迭代次数',2);click('开始重建');result()
        assert not js('document.querySelector(".original-card").open')
        assert js('document.body.innerText.includes("已按四点透视校正")')
        click('保存重建结果');until('document.body.innerText.includes("已保存 4 个结果文件")')
        report['checks'].append('actual native capture result includes cropped four corners; real immutable warp task, collapsed title and four artifact save')
        click('图片涂鸦重建');click('应用校正并绘图');until('!!document.querySelector("canvas[aria-label=可缩放频谱画布]")')
        js('window.paintEvents=[];document.addEventListener("pointerdown",e=>window.paintEvents.push(e.isTrusted),true)')
        draw();click('撤销笔迹');until('document.querySelector(".axis-note").textContent.includes("0 笔")');click('重做笔迹');click('生成涂鸦音频');result();click('保存重建结果');until('document.body.innerText.includes("已保存 4 个结果文件")')
        report['checks'].append('native QTest brush, undo/redo, image drawing real Griffin-Lim task and save')
        click('音频藏信息');assert js('document.querySelectorAll(".m09-page .workbench-left").length')==0
        click('选择音频目录');until('!!document.querySelector("canvas[aria-label=可缩放频谱画布]")');draw();click('生成藏信息音频');result();click('保存重建结果');until('document.body.innerText.includes("已保存 4 个结果文件")')
        assert js('document.body.innerText.includes("原始相位")')
        report['checks'].append('no image calibration in audio mode; original-phase stereo preview, QTest painting, real audio task/save')
        for mode in ('音频藏信息','图片涂鸦重建'):
            click(mode)
            for width,height in [(1920,1080),(1366,768),(1100,800)]:
                w.resize(width,height);pause(200)
                for theme in ('light','dark'):
                    js('document.documentElement.dataset.theme='+json.dumps(theme));pause(500)
                    geo=js('(()=>{const e=document.querySelector(".m09-page"),t=document.querySelector(".paint-tools");return {scroll:e.scrollWidth,client:e.clientWidth,toolsHeight:t.getBoundingClientRect().height,brushes:[...t.querySelectorAll("button")].map(e=>{const r=e.getBoundingClientRect();return {w:r.width,h:r.height}})}})()')
                    assert geo['scroll']<=geo['client']+1 and all(b['h']<50 for b in geo['brushes']),geo
                    name=f'{mode}-{width}-{theme}';w.view.grab().save(str(out/(name+'.png')));report['layouts'].append(dict(name=name,geometry=geo))
        assert all(js('window.paintEvents'))
        # Verify actual saved metadata/audio, not only button state.
        import soundfile as sf
        import numpy as np
        metas=[json.loads(p.read_text('utf8')) for p in (out/'saved').glob('*.json')]
        assert len(metas)==3 and any(m.get('phase_method')=='original-stft-phase' for m in metas)
        rates=[sf.info(p).samplerate for p in (out/'saved').glob('*.wav')];assert 16000 in rates
        report['success']=True
    finally:
        (out/'qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        js('window.onbeforeunload=null');w.close();pause(300);app.quit();print(out,flush=True)


if __name__=='__main__':main()
