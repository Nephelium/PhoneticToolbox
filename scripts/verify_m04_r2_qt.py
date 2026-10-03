"""M04 R2 actual Qt + real task service, synthetic audio, owned offscreen window."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --mute-audio')
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
from uuid import uuid4
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/p) for p in ('desktop/src','backend/src','packages/phonetic_core/src')]
os.environ['PYTHONPATH'] = os.pathsep.join(sys.path[:3])
os.environ['PTB_EGG_PYTHON'] = str(ROOT/'.venv/m03-compatible/python.exe')
import numpy as np
from scipy.io import wavfile
from PyQt6.QtCore import QEventLoop,QTimer,Qt
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files


def main():
    out=ROOT/'output/validation/m04-r2'/('qt-'+uuid4().hex)
    out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
        source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    t=np.arange(96000)/48000
    a=.3*np.sin(2*np.pi*150*t)+.1*np.sin(2*np.pi*800*t)
    wavfile.write(inputs/'LPC-Qt.wav',48000,np.column_stack([a,a*.5]))
    wavfile.write(inputs/'second.wav',48000,a*.5)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    register_scheme();app=QApplication(['M04-R2-owned-offscreen-QA'])
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen,True);w.resize(1920,1080);w.show()
    def pause(ms=50):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda v:(box.append(v),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        return box[0] if box else None
    def wait(code,seconds=50):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if js(code):return
            pause()
        raise AssertionError(code+' / '+str(js('document.querySelector(".lpc-page")?.innerText')))
    def click(label):
        code='[...document.querySelectorAll("button")].find(e=>e.offsetParent&&!e.disabled&&e.textContent.trim()==='+json.dumps(label)+')'
        wait('!!'+code);js(code+'.click()');pause()
    def fill(selector,value):
        js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
    def select(name):
        selector='select[aria-label="LPC 音频文件"]'
        wait('[...document.querySelectorAll('+json.dumps(selector+' option')+')].some(e=>e.textContent==='+json.dumps(name)+')')
        js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.value=[...e.options].find(o=>o.textContent==='+json.dumps(name)+').value;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        wait('!!document.querySelector(".lpc-page .wave-track svg")&&!document.querySelector(".lpc-files").innerText.includes("正在读取音频")')
    QFileDialog.getExistingDirectory=lambda *a,**k:str(saved if '结果' in str(a) or '输出' in str(a) else inputs)
    dialogs=[];choices=['',str(saved/'fixed.png'),str(saved/'dynamic.png'),str(saved/'fixed-return.png')]
    def dialog(*a,**k):
        dialogs.append(str(a[2]));return (choices.pop(0),'')
    QFileDialog.getSaveFileName=dialog
    report={'success':False,'checks':[],'png':[],'layouts':[],'scope':'Windows source Qt offscreen; synthetic two-channel audio; no physical audio or DPI'}
    preview='!!document.querySelector(".spectrogram-canvas[aria-busy=false] canvas")&&document.querySelector(".spectrogram-canvas canvas").offsetParent!==null'
    def native_file(name):
        deadline=time.monotonic()+30;p=saved/name
        while time.monotonic()<deadline:
            if p.exists() and not QImage(str(p)).isNull():return p
            pause()
        raise AssertionError('Native PNG write did not complete: '+name)
    def axis():
        return js('[...document.querySelectorAll(".lpc-spectrum svg text")].map(e=>e.textContent)')
    try:
        wait('document.querySelector(".host-badge")?.textContent==="本地桌面"')
        click('LPC 谱图');click('打开 WAV 目录');select('LPC-Qt.wav')
        fill('.lpc-transport .selection-controls label:first-of-type input',.1)
        fill('.lpc-transport .selection-controls label:last-of-type input',.2)
        assert js('document.querySelectorAll(".lpc-transport .selection-controls input").length===2&&document.querySelectorAll(".lpc-transport .playback-seek input").length===1&&document.querySelectorAll(".lpc-transport .volume input").length===1')
        assert not js('[...document.querySelectorAll(".lpc-page button")].some(e=>e.textContent==="清除选区")')
        report['checks'].append('Qt shared selection, playback progress and volume; no clear button')
        click('开始分析');wait('!!document.querySelector(".lpc-spectrum svg")');fixed=axis()
        click('保存 PNG 图片');wait('document.querySelector(".lpc-results").innerText.includes("保存 PNG")');pause(300)
        assert len(dialogs)==1 and not list(saved.glob('*.png'))
        report['checks'].append('Qt native PNG dialog cancellation writes no file')
        click('保存 PNG 图片');native_file('fixed.png')
        click('动态纵轴');assert axis()!=fixed;w.view.grab().save(str(out/'qt-spectrum-dynamic.png'));click('保存 PNG 图片');native_file('dynamic.png')
        click('固定纵轴');assert axis()==fixed;click('保存 PNG 图片');native_file('fixed-return.png')
        for name in ('fixed.png','dynamic.png','fixed-return.png'):
            p=saved/name;image=QImage(str(p));assert (image.width(),image.height())==(2400,1350)
            assert image.pixelColor(0,0).name()=='#ffffff'
            assert abs(image.dotsPerMeterX()*.0254-300)<.01 and abs(image.dotsPerMeterY()*.0254-300)<.01
            report['png'].append({'name':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'dpi':image.dotsPerMeterX()*.0254,'size':[image.width(),image.height()]})
        assert report['png'][0]['sha256']==report['png'][2]['sha256']!=report['png'][1]['sha256']
        report['checks'].append('Qt immediate axis switch and 3 native 2400x1350/300 DPI white PNGs; fixed return bytes identical')
        click('波形');js('(()=>{const e=[...document.querySelectorAll(".lpc-page .view-tabs input")][0];e.checked=true;e.dispatchEvent(new Event("change",{bubbles:true}));})()');wait(preview)
        fill('.lpc-transport .selection-controls label:last-of-type input',.35);wait(preview)
        click('+');pause(100);wait(preview)
        select('second.wav');wait(preview)
        report['checks'].append('Qt actual Praat preview survives numeric edits, zoom and equal-duration file switch')
        for width,height in ((1920,1080),(1280,720)):
            w.resize(width,height);pause(150);wait(preview)
            report['layouts'].append(js('({width:innerWidth,height:innerHeight,overflow:document.documentElement.scrollWidth>innerWidth})'))
            assert not report['layouts'][-1]['overflow'];w.view.grab().save(str(out/('qt-wave-'+str(width)+'.png')))
        report['originals_unchanged']=all(hashlib.sha256((inputs/n).read_bytes()).hexdigest()==v for n,v in hashes.items())
        assert report['originals_unchanged'];report['dialogs']=dialogs;report['success']=True
    except Exception as exc:
        report['failure']=str(exc);w.view.grab().save(str(out/'qt-failed.png'));raise
    finally:
        (out/'qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps({'out':str(out),'success':report['success'],'checks':report['checks'],'failure':report.get('failure')},ensure_ascii=False),flush=True)
        w.close();pause(500)

if __name__=='__main__':main()
