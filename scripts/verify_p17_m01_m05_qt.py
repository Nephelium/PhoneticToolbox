"""Actual Qt/source host, real-only inputs, owned new workspace; no global UI."""
import json
import os
from pathlib import Path
import time
from p17_m01_m05_bridge import prepare,ROOT
from PyQt6.QtCore import QEventLoop,QTimer,Qt
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme

def main():
    out,inputs,saved,db,cache,manifest=prepare()
    register_scheme();app=QApplication(['P17 M01-M05 Qt'])
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
    try:
        wait('document.querySelector(".host-badge")?.textContent==="本地桌面"')
        if os.environ.get('P17_QT_LIP_ONLY')=='1':
            click('唇形提取');wait('!!document.querySelector(".lip-page")');click('刷新本地历史任务')
            wait('![...document.querySelectorAll("button")].find(e=>e.textContent.trim()==="刷新本地历史任务")?.disabled')
            assert not js('document.querySelector("select[aria-label=\"历史唇形任务\"]")?.options.length>0')
            geometry('M05');report['checks'].append('M05 final Qt refresh native empty history; no capture requested');report['success']=True
            return
        short=manifest[0]['copy'];click('参数估计');click('选择音频目录')
        start=time.perf_counter();js('[...document.querySelectorAll(".m01-file-list .file-row")].find(e=>e.innerText.includes('+json.dumps(short)+')).click()')
        wait('!!document.querySelector(".m01-page .wave-track")&&document.querySelectorAll(".m01-tiers option").length>0')
        report['timings'].append(dict(name='M01 waveform+TextGrid',seconds=time.perf_counter()-start));geometry('M01')
        click('参数显示');click('选择音频目录');start=time.perf_counter();js('[...document.querySelectorAll(".m02-files button")].find(e=>e.innerText.includes('+json.dumps(short)+')).click()')
        wait('document.querySelectorAll(".m02-parameters input").length>0');report['timings'].append(dict(name='M02 waveform+table',seconds=time.perf_counter()-start));js('document.querySelector(".m02-parameters input").click()');click('将 1 项分配到图窗');wait('!!document.querySelector(".parameter-curve")');click('保存当前图');wait('![...document.querySelectorAll("button")].find(e=>e.textContent==="保存当前图")?.disabled');geometry('M02');report['checks'].append('M01/M02 real file, UTF16 TextGrid, parameter table and native PNG save')
        click('EGG 信号分析');click('打开 WAV 目录');start=time.perf_counter();select('.egg-source select','EGG-real.wav');wait('document.querySelector(".egg-live-status")?.textContent==="实时预览"');report['timings'].append(dict(name='M03 first four plots',seconds=time.perf_counter()-start));geometry('M03')
        fill('EGG 选区起点',40);fill('EGG 选区时长',.5);wait('document.querySelector(".egg-live-status")?.textContent==="实时预览"')
        fill('EGG 微观窗口',100);wait('document.querySelector(".egg-live-status")?.textContent==="实时预览"')
        for i in range(20):
            start=time.perf_counter();fill('EGG 微观窗口',100 if i%2 else 50);wait('document.querySelector(".egg-live-status")?.textContent==="实时预览"');report['timings'].append(dict(name='M03 update '+str(i),seconds=time.perf_counter()-start))
        report['checks'].append('M03 20 native Qt real-session updates')
        click('LPC 谱图');click('打开 WAV 目录');select('select[aria-label="LPC 音频文件"]',short);wait('!!document.querySelector(".lpc-page .wave-track")&&document.querySelectorAll(".lpc-files select")[2].value!==""');fill('LPC 选区起点',.2);fill('LPC 选区终点',.4);start=time.perf_counter();click('开始分析');wait('!!document.querySelector(".lpc-spectrum svg")');report['timings'].append(dict(name='M04 analysis',seconds=time.perf_counter()-start));click('选择目录保存完整结果');wait('document.querySelector(".lpc-page").innerText.includes("已保存 3 个")');geometry('M04');report['checks'].append('M04 native real LPC job and PNG/WAV/JSON save')
        click('唇形提取');wait('!!document.querySelector(".lip-page")');geometry('M05');report['checks'].append('M05 actual Qt empty layout; physical capture user-deferred')
        report['success']=True
    except Exception as exc:
        report['error']=str(exc);w.view.grab().save(str(out/'qt-failed.png'));raise
    finally:
        import hashlib
        corpus=Path(r'C:\Users\13680\Desktop\project\音频数据')
        unchanged=all(hashlib.sha256((corpus/m['source']).read_bytes()).hexdigest()==m['sha256'] for m in manifest)
        (out/'originals-unchanged.json').write_text(json.dumps(dict(unchanged=unchanged)),encoding='utf8')
        (out/'qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(out,flush=True);w.close();pause(300)

if __name__=='__main__':main()
