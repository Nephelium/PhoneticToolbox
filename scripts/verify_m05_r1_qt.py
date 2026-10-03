"""Actual offscreen Qt/QWebChannel, private authorized video and V2-format readback.

No physical camera/microphone use. Synthetic capture is injected explicitly.
M05_R1_VIDEO and M05_R1_SAVED are local-only environment inputs.
"""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/src'),str(ROOT/'backend/src'),str(ROOT/'packages/phonetic_core/src'),str(ROOT/'scripts')]
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/p) for p in ('backend/src','desktop/src','packages/phonetic_core/src'))
from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication,QFileDialog,QMessageBox
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files
from prepare_m05_r1_compat import prepare


def main():
    out=ROOT/'output/validation/m05-r1'/('qt-'+uuid4().hex[:10]);out.mkdir(parents=True)
    print(out,flush=True)
    dist=out/'dist';shutil.copytree(ROOT/'frontend/dist',dist)
    video=Path(os.environ['M05_R1_VIDEO']);shutil.copyfile(video,dist/'qa-recording.webm')
    compat=out/'compat';fixture=prepare(os.environ['M05_R1_SAVED'],compat,ROOT.parent/'PhoneticToolbox_v2')
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    saved=out/'saved';saved.mkdir();os.environ['PTB_M05_PYTHON']=str(ROOT/'.venv/m05/Scripts/python.exe')
    register_scheme();app=QApplication(['M05-R1-offscreen-QA']);window=Workbench(dist,test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal')
    window.show();window.resize(1920,1080)
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        box=[];loop=QEventLoop();window.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(5000,loop.quit);loop.exec();return box[0] if box else None
    def wait(code,seconds=80):
        start=time.monotonic()
        while time.monotonic()-start<seconds:
            if js(code):return
            pause()
        raise AssertionError('Qt timeout: '+code+'; '+str(js("document.body.innerText.slice(-1600)")))
    def click(text):
        code="(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==="+json.dumps(text)+");if(!b||b.disabled)return null;b.scrollIntoView({block:'nearest'});const r=b.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()"
        point=js(code);assert point,text
        pause(100);point=js(code)
        QTest.mouseClick(window.view.focusProxy(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(point[0]),round(point[1])))
        pause(100)
    report=dict(success=False,checks=[],fixture=fixture,physical_devices=False)
    original=QFileDialog.getExistingDirectory;original_question=QMessageBox.question
    QFileDialog.getExistingDirectory=lambda *args,**kwargs:str(saved if '结果' in str(args[1]) else compat)
    QMessageBox.question=lambda *args,**kwargs:QMessageBox.StandardButton.Yes
    try:
        wait("!!document.querySelector('nav')");click('唇形提取');wait("!!document.querySelector('.lip-page')")
        js("(async()=>{const raw=await(await fetch('/qa-recording.webm')).blob();const d=new DataTransfer();d.items.add(new File([raw],'录制.webm'));const input=document.querySelector('.lip-page input[type=file]');input.files=d.files;input.dispatchEvent(new Event('change',{bubbles:true}));})()")
        wait("document.querySelector('.lip-page').textContent.includes('分析所选 1 个视频')");click('分析所选 1 个视频')
        wait("document.querySelectorAll('.lip-curves svg').length===4")
        assert js("!!document.querySelector('.workbench-left .lip-state')")
        report['checks'].append('actual Qt full offline analysis and left capture/four curves')
        js("window.r1frames=new Set();window.r1start=performance.now();window.r1done=false;const record=()=>{window.r1frames.add(Number(document.querySelector('[aria-label=\"回放帧\"]').value));if(performance.now()-window.r1start<3000)requestAnimationFrame(record);else window.r1done=true;};requestAnimationFrame(record);")
        js("document.querySelector('[aria-label=\"原始视频音频回放\"]').muted=true")
        click('播放动画');wait('window.r1done',10)
        report['playback_frames_in_3s']=js('window.r1frames.size');assert report['playback_frames_in_3s']>45,report
        click('暂停动画');click('应用偏移并保存');wait("document.querySelector('.lip-page').textContent.includes('完整结果与偏移写入完成')")
        for theme in ('light','dark'):
            js("document.documentElement.dataset.theme="+json.dumps(theme)+";document.documentElement.style.colorScheme="+json.dumps(theme));pause(200);js("document.getAnimations().forEach(a=>a.finish())");pause(100);window.grab().save(str(out/(theme+'.png')))
        report['checks'].append('actual Qt video-clock playback, native save, light/dark screenshots')
        # Actual bundled page, synthetic canvas + AudioContext input only.
        js("navigator.mediaDevices.getUserMedia=async constraints=>{const c=document.createElement('canvas');c.width=320;c.height=240;const x=c.getContext('2d');let n=0;const paint=()=>{x.fillStyle=`rgb(${n++%200},70,100)`;x.fillRect(0,0,320,240)};paint();const timer=setInterval(paint,33),s=c.captureStream(30),v=s.getVideoTracks()[0],stop=v.stop.bind(v);v.stop=()=>{clearInterval(timer);stop()};if(constraints.audio){const a=new AudioContext(),o=a.createOscillator(),d=a.createMediaStreamDestination();o.connect(d);o.start();const t=d.stream.getAudioTracks()[0],stop=t.stop.bind(t);t.stop=()=>{o.stop();void a.close();stop()};s.addTrack(t)}return s};")
        js("const s=[...document.querySelectorAll('.lip-page select')].find(s=>[...s.options].some(o=>o.value==='raw'));s.value='raw';s.dispatchEvent(new Event('change',{bubbles:true}));")
        for attempt in range(2):
            click('开始录制');wait("document.querySelector('.lip-state').textContent.includes('阶段 recording')");pause(1300)
            click('停止并收尾');wait("document.querySelector('.lip-state').textContent.includes('阶段 ready')")
            click('保存录制（视频 + 音频 + 唇形）');wait("document.querySelector('.lip-page').textContent.includes('MP4、WAV 与采集记录已保存并核对')")
        report['checks'].append('actual Qt two synthetic recordings, native MP4/WAV save and same-page restart')
        click('参数显示');wait("!!document.querySelector('.m02-page')");click('选择音频目录')
        for stem in ('v3','v2'):
            wait("!![...document.querySelectorAll('.m02-files button')].find(e=>e.textContent.trim()==="+json.dumps(stem+'.wav')+")")
            click(stem+'.wav');wait("[...document.querySelectorAll('.m02-parameters label')].filter(e=>/Lip(Area|Width|Open|Circ)/.test(e.textContent)).length===4")
            click('全选可见参数');click('将 4 项分配到图窗')
            wait("document.querySelectorAll('.parameter-chart path').length>0")
            window.grab().save(str(out/('M02-'+stem+'.png')))
            report['checks'].append('M02 reads '+stem+' WAV and all four directly associated lip tracks')
        click('语音标注对齐');wait("!!document.querySelector('.annotation-page')");click('选择语料文件夹')
        for stem in ('v3','v2'):
            wait("!![...document.querySelectorAll('.annotation-file-list button')].find(e=>e.textContent.includes("+json.dumps(stem+'.wav')+"))")
            assert js("(()=>{[...document.querySelectorAll('.annotation-file-list button')].find(e=>e.textContent.includes("+json.dumps(stem+'.wav')+")).click();return true})()")
            wait("(()=>{const root=document.querySelector('.annotation-page'),select=document.querySelector('[aria-label=\"唇形记录\"]'),offset=document.querySelector('[aria-label=\"唇形共同偏移毫秒\"]');return !root.querySelector('[role=alert]')&&!root.textContent.includes('正在读取音频与标注')&&select&&!select.disabled&&select.selectedOptions[0]?.textContent==="+json.dumps(stem+('.lip.json' if stem=='v3' else '.pkl'))+"&&Number(offset?.value)===125;})()")
            pause(200);window.grab().save(str(out/('M12-'+stem+'.png')))
            report['checks'].append('M12 loads '+stem+' audio/lip association with the shared time contract')
        report['success']=True
    finally:
        report['page_state']=js('document.body.innerText');report['user_agent']=js('navigator.userAgent')
        window.grab().save(str(out/'final.png'));QFileDialog.getExistingDirectory=original;QMessageBox.question=original_question
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
        window.accept_close();pause()
    print(json.dumps({k:v for k,v in report.items() if k!='page_state'},ensure_ascii=False))


if __name__=='__main__':main()
