"""Actual Qt/source host, real-only inputs, owned new workspace; no global UI."""
import json
import os
from pathlib import Path
import time
from p17_m04_r2_bridge import prepare,ROOT
from PyQt6.QtCore import QEventLoop,QTimer,Qt
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme

def main():
    out,inputs,saved,db,cache,manifest=prepare()
    register_scheme();app=QApplication(['P17 M04 R2 Qt'])
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,
        reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe',vocal_profile=out/'vocal')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen,True);w.showMaximized();w.showNormal();w.resize(1920,1000)
    def pause(ms=50):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda v:(box.append(v),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec();return box[0] if box else None
    def wait(code,seconds=90):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if js(code):return
            pause()
        raise AssertionError(code+' / '+str(js('document.querySelector("main")?.innerText')))
    def click(label):
        code='[...document.querySelectorAll("button")].find(e=>e.offsetParent&&!e.disabled&&e.textContent.trim()==='+json.dumps(label)+')'
        wait('!!'+code);js(code+'.click()');pause()
    def select(selector,label):
        js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.value=[...e.options].find(o=>o.textContent==='+json.dumps(label)+').value;e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
    def fill(label,value):
        js('(()=>{const e=document.querySelector('+json.dumps('input[aria-label="'+label+'"]')+');e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
    QFileDialog.getExistingDirectory=lambda *a,**k:str(saved if '结果' in str(a) or '输出' in str(a) else inputs)
    QFileDialog.getSaveFileName=lambda *a,**k:(str(saved/Path(a[2]).name),'')
    report=dict(success=False,checks=[],timings=[],geometry=[],hidden=True)
    def geometry(mid):
        report['geometry'].append(dict(module=mid,**js('({width:innerWidth,height:innerHeight,dpr:devicePixelRatio,screen:[screen.width,screen.height],modules:[...document.querySelectorAll(".module-frame")].filter(e=>e.offsetParent).map(e=>({name:e.className,height:e.clientHeight,scroll:e.scrollHeight}))})')))
        w.view.grab().save(str(out/(mid+'-qt.png')))
    def wait_native(predicate,seconds=90):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if predicate():return
            pause()
        raise AssertionError('native file condition timed out')
    dialogs=[];download_states=[];choices=['',str(saved/'R2-direct.png')]
    def save_dialog(*a,**k):
        dialogs.append(str(a[2]));return (choices.pop(0),'')
    QFileDialog.getSaveFileName=save_dialog
    def observed(download):
        download.stateChanged.connect(lambda _state:download_states.append(download.state().name))
    w.profile.downloadRequested.connect(observed)
    try:
        wait('document.querySelector(".host-badge")?.textContent==="本地桌面"')
        click('LPC 谱图');click('打开 WAV 目录');select('select[aria-label="LPC 音频文件"]',manifest[0]['copy']);wait('!!document.querySelector(".lpc-page .wave-track")&&document.querySelectorAll(".lpc-files select")[2].value!==""');fill('LPC 选区起点',.2);fill('LPC 选区终点',.4)
        assert js('document.querySelectorAll(".workbench-right .lpc-range-controls input").length===2')
        click('开始分析');wait('!!document.querySelector(".lpc-spectrum svg")');geometry('M04-result')
        click('保存 PNG 图片');wait_native(lambda:len(dialogs)==1);pause(150);assert not list(saved.rglob('*.png'));report['checks'].append('Qt native PNG save dialog cancellation writes no file')
        click('保存 PNG 图片');wait_native(lambda:len(dialogs)==2 and (saved/'R2-direct.png').exists() and (saved/'R2-direct.png').stat().st_size>1000);wait_native(lambda:'DownloadCompleted' in download_states)
        click('选择目录保存完整结果');wait('document.querySelector(".lpc-page").innerText.includes("已保存 3 个")')
        import hashlib
        direct=saved/'R2-direct.png';full=next(p for p in saved.rglob('*.png') if p!=direct)
        assert hashlib.sha256(direct.read_bytes()).hexdigest()==hashlib.sha256(full.read_bytes()).hexdigest()
        from PyQt6.QtGui import QImage
        image=QImage(str(direct));assert (image.width(),image.height())==(2400,1350);assert image.pixelColor(0,0).name()=='#ffffff'
        report['png']=dict(bytes=direct.stat().st_size,width=image.width(),height=image.height(),dpi_x=image.dotsPerMeterX()*.0254,sha256=hashlib.sha256(direct.read_bytes()).hexdigest(),same_as_full_result=True)
        report['checks'].append('Qt native direct PNG completed; 2400x1350 white background and bytes identical to full managed PNG')
        report['dialogs']=dialogs;report['download_states']=download_states;report['success']=True
    except Exception as exc:
        report['error']=str(exc);w.view.grab().save(str(out/'qt-failed.png'));raise
    finally:
        import hashlib
        corpus=Path(r'C:\Users\13680\Desktop\project\音频数据')
        unchanged=all(hashlib.sha256((corpus/m['source']).read_bytes()).hexdigest()==m['sha256'] for m in manifest)
        (out/'originals-unchanged.json').write_text(json.dumps(dict(unchanged=unchanged)),encoding='utf8')
        (out/'qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(out,flush=True);w.close();pause(300)

if __name__=='__main__':main()
