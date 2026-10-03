"""M05-R3: actual offscreen Qt, synthetic owned devices, direct native save/readback."""
import base64
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/p) for p in ('desktop/src','backend/src','packages/phonetic_core/src')]
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/p) for p in ('backend/src','desktop/src','packages/phonetic_core/src'))
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu-compositing --disable-gpu-rasterization --mute-audio')
from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication,QFileDialog,QMessageBox
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files

def main():
    out=ROOT/'output/validation/m05-r3'/('qt-'+uuid4().hex[:10]);out.mkdir(parents=True);print(out,flush=True)
    dist=out/'dist';shutil.copytree(ROOT/'frontend/dist',dist)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    saved=out/'saved';saved.mkdir();os.environ['PTB_M05_PYTHON']=str(ROOT/'.venv/m05/Scripts/python.exe')
    register_scheme();app=QApplication(['M05-R3-offscreen-QA']);window=Workbench(dist,test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal')
    window.show();window.resize(1920,1080)
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        box=[];loop=QEventLoop();window.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(5000,loop.quit);loop.exec();return box[0] if box else None
    def wait(code,seconds=90):
        start=time.monotonic()
        while time.monotonic()-start<seconds:
            if js(code):return
            pause()
        raise AssertionError('Qt timeout: '+code+'; '+str(js('document.body.innerText')))
    def click(text):
        code="(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==="+json.dumps(text)+");if(!b||b.disabled)return null;b.scrollIntoView({block:'nearest'});const r=b.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()"
        point=js(code);assert point,text;pause();point=js(code)
        QTest.mouseClick(window.view.focusProxy(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(point[0]),round(point[1])));pause()
    def phase(value):wait("document.querySelector('.lip-state')?.dataset.phase===("+json.dumps(value)+")")
    def set_mode(value):js("(()=>{const s=document.querySelector('[aria-label=\"采集模式\"]');s.value="+json.dumps(value)+";s.dispatchEvent(new Event('change',{bubbles:true}));})()")
    report=dict(success=False,checks=[],physical_devices=False)
    original=QFileDialog.getExistingDirectory;original_question=QMessageBox.question
    input_directory=saved;cancel_next=False
    def choose(*args,**kwargs):
        nonlocal cancel_next
        if cancel_next:cancel_next=False;return ''
        return str(saved if '结果' in str(args[1]) else input_directory)
    QFileDialog.getExistingDirectory=choose
    QMessageBox.question=lambda *args,**kwargs:QMessageBox.StandardButton.Yes
    try:
        wait("!!document.querySelector('nav')");click('唇形提取');wait("!!document.querySelector('.lip-page')")
        js((ROOT/'tests/support/m05_r2_capture.js').read_text('utf8'))
        portrait='data:image/png;base64,'+base64.b64encode((ROOT/'output/validation/m05/inputs/astronaut.png').read_bytes()).decode()
        js('window.installM05R2Capture('+json.dumps(portrait)+').then(()=>window.fixtureReady=true)');wait('window.fixtureReady')
        set_mode('realtime');click('开始录制');phase('recording');wait("Number(document.querySelector('meter')?.value)>-40")
        wait("Number(document.querySelector('.lip-state').textContent.match(/检出 (\\d+)/)?.[1])>8");pause(1300)
        click('结束录制');phase('ready');wait("document.querySelector('.lip-page').getAttribute('aria-busy')==='false'")
        click('检查偏移量');wait("!!document.querySelector('.offset-editor')");wait("!document.querySelector('.offset-editor [role=alert]')&&!document.querySelector('.offset-editor [role=status]')")
        js("(()=>{const s=document.querySelector('[aria-label=\"唇形偏移量 ms\"]');s.value='125';s.dispatchEvent(new Event('input',{bubbles:true}));s.focus()})()")
        QTest.keyClick(window.view.focusProxy(),Qt.Key.Key_Right);pause();assert js("document.querySelector('[aria-label=\"唇形偏移量 ms\"]').value==='126'")
        window.grab().save(str(out/'offset.png'));click('应用偏移量');pause()
        report['checks'].append('actual Qt native decoded waveform, aligned axes and keyboard 1 ms offset')
        def save_options(video=True,animation=True):
            click('另存录制');wait("!!document.querySelector('.lip-save-options')")
            js("(()=>{const x=document.querySelectorAll('.lip-save-options input');x[0].checked="+json.dumps(video)+";x[0].dispatchEvent(new Event('change',{bubbles:true}));x[1].checked="+json.dumps(animation)+";x[1].dispatchEvent(new Event('change',{bubbles:true}));})()")
            click('选择目录并保存');wait("!document.querySelector('.lip-save-options')",180)
        save_options();input_directory=next(saved.glob('M05-recording-*'));info=json.loads((input_directory/'recording-export.json').read_text('utf8'))
        assert (input_directory/'raw_recording.mp4').is_file() and (input_directory/'face_animation.mp4').is_file()
        exchange=json.loads((input_directory/'audio_recording.lip.json').read_text('utf8'));assert exchange['data']['metadata']['lip_manual_offset']==.126
        report['checks'].append('native video + animation + WAV + lip save, offset preserved')
        old=set(saved.iterdir());save_options(False,False);folder=next(iter(set(saved.iterdir())-old));assert not list(folder.glob('*.mp4'));assert (folder/'audio_recording.wav').is_file();report['checks'].append('native save with video and animation unchecked contains no MP4')
        js("(()=>{const v=document.querySelector('[aria-label=\"视频与动画同步回放\"]');v.currentTime=.7;v.play()})()")
        pause(350);js("document.querySelector('[aria-label=\"视频与动画同步回放\"]').pause()")
        assert js("Number(document.querySelector('[aria-label=\"回放帧\"]').value)>0")
        for width,height,theme in ((1920,1080,'light'),(1440,900,'dark'),(1280,800,'dark')):
            window.resize(width,height);js('document.documentElement.dataset.theme='+json.dumps(theme));pause(300)
            layout=js("(()=>{const q=s=>{const r=document.querySelector(s).getBoundingClientRect();return {x:r.x,y:r.y,right:r.right,bottom:r.bottom}};return {face:q('.lip-face-section'),parameters:q('.lip-metrics'),media:q('.lip-media-section'),width:innerWidth,height:innerHeight}})()")
            assert layout['media']['x']>layout['face']['x'],layout
            assert layout['parameters']['bottom']<layout['height'],layout
            report['checks'].append(dict(layout=layout));window.grab().save(str(out/(str(width)+'-'+theme+'.png')))
        window.resize(1920,1080);pause()
        set_mode('record_then_analyze');click('开始录制');phase('recording');pause(1300);click('结束录制');phase('ready');wait("document.querySelector('.lip-page').getAttribute('aria-busy')==='false'")
        old=set(saved.iterdir());click('另存录制');wait("document.querySelector('.lip-page').getAttribute('aria-busy')==='false'");folder=next(iter(set(saved.iterdir())-old));assert {p.name for p in folder.iterdir()}=={'raw_recording.mp4','audio_recording.wav','recording-export.json'}
        tracks=js('({created:window.m05R2.created,ended:window.m05R2.ended})');assert tracks['created']==tracks['ended'];report['checks'].append(dict(high_rate_media_only_and_cleanup=tracks))
        report['first_directory']=str(input_directory);report['success']=True
    finally:
        report['page_state']=js('document.body.innerText');report['user_agent']=js('navigator.userAgent')
        window.grab().save(str(out/'final.png'));QFileDialog.getExistingDirectory=original;QMessageBox.question=original_question
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8');window.accept_close();pause()
    print(json.dumps({k:v for k,v in report.items() if k!='page_state'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
