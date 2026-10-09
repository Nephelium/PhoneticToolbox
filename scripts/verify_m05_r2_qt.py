"""M05-R2: actual offscreen Qt, synthetic owned devices, direct native save/readback."""
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
    out=ROOT/'output/validation/m05-r2'/('qt-'+uuid4().hex[:10]);out.mkdir(parents=True);print(out,flush=True)
    dist=out/'dist';shutil.copytree(ROOT/'frontend/dist',dist)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    saved=out/'saved';saved.mkdir();os.environ['PTB_M05_PYTHON']=str(ROOT/'.venv/m05/Scripts/python.exe')
    register_scheme();app=QApplication(['M05-R2-offscreen-QA']);window=Workbench(dist,test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal')
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
    def phase(value):wait("document.querySelector('.lip-state')?.textContent.includes("+json.dumps('阶段 '+value)+")")
    def set_mode(value):js("(()=>{const s=[...document.querySelectorAll('.lip-page select')].find(s=>[...s.options].some(o=>o.value==='raw'));s.value="+json.dumps(value)+";s.dispatchEvent(new Event('change',{bubbles:true}));})()")
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
        wait("Number(document.querySelector('.lip-state').textContent.match(/检出 (\d+)/)?.[1])>8");pause(1600)
        assert js("window.m05R2.requests.length===2&&window.m05R2.requests[1].audio.deviceId.exact==='mic'")
        click('停止并收尾');phase('ready');wait("!!document.querySelector('[aria-label=\"刚录制的视频与声音\"]')")
        cancel_next=True;click('保存录制（视频 + 音频 + 唇形）');wait("document.body.textContent.includes('已取消保存，录制保留')");assert not list(saved.iterdir())
        click('保存录制（视频 + 音频 + 唇形）');wait("document.body.textContent.includes('MP4、WAV 与采集记录已保存并核对')")
        input_directory=next(saved.glob('M05-recording-*'));info=json.loads((input_directory/'recording-export.json').read_text('utf8'))
        assert info['audio']['signal']['peak_dbfs']>-30 and info['candidate_frames']>8
        exchange=json.loads((input_directory/'audio_recording.lip.json').read_text('utf8'));assert exchange['data']['metadata']['source_status']=='candidate'
        report['checks'].append('actual Qt automatic microphone, level meter, cancelled save/retry, direct MP4/WAV/associated lip without analysis')
        click('另存录制（视频 + 音频 + 唇形）');wait("document.querySelector('.lip-page').getAttribute('aria-busy')==='false'");assert len(list(saved.glob('M05-recording-*')))==2
        report['checks'].append('saved recording remains available for another native directory save')
        # Downstream readers receive the directly recorded bundle, before legacy analysis.
        click('参数显示');wait("!!document.querySelector('.m02-page')");click('选择音频目录');wait("!!document.querySelector('.m02-files button')");click('audio_recording.wav')
        wait("[...document.querySelectorAll('.m02-parameters label')].filter(e=>/Lip(Area|Width|Open|Circ)/.test(e.textContent)).length===4")
        click('全选可见参数');click('将 4 项分配到图窗');wait("document.querySelectorAll('.parameter-chart path').length>0");window.grab().save(str(out/'M02.png'))
        click('TextGrid标注');wait("!!document.querySelector('.annotation-page')");click('选择语料文件夹')
        wait("!![...document.querySelectorAll('.annotation-file-list button')].find(e=>e.textContent.includes('audio_recording.wav'))")
        js("[...document.querySelectorAll('.annotation-file-list button')].find(e=>e.textContent.includes('audio_recording.wav')).click()")
        wait("(()=>{const s=document.querySelector('[aria-label=\"唇形记录\"]');return s&&!s.disabled&&s.selectedOptions[0]?.textContent==='audio_recording.lip.json'&&!document.querySelector('.annotation-page').textContent.includes('正在读取音频与标注')})()")
        window.grab().save(str(out/'M12.png'));report['checks'].append('M02 four lip tracks and M12 auto-association from direct realtime save')
        click('唇形提取');wait("!!document.querySelector('.lip-page')");click('可选：按 V2 模型重新分析');wait("document.body.textContent.includes('正式分析完成，结果等待保存')")
        js("(()=>{const s=document.querySelector('[aria-label=\"回放帧\"]');s.value=Math.floor(Number(s.max)/2);s.dispatchEvent(new Event('input',{bubbles:true}))})()")
        wait("(()=>{const c=document.querySelector('.lip-video canvas');return c&&c.getContext('2d').getImageData(0,0,c.width,c.height).data.some((v,i)=>i%4===3&&v>0)})()")
        pixels=js("(()=>{const c=document.querySelector('.lip-video canvas'),d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let top=c.height,bottom=0;for(let y=0;y<c.height;y++)for(let x=0;x<c.width;x++){const i=(y*c.width+x)*4;if(d[i+1]>180&&d[i]<80&&d[i+2]<80&&d[i+3]>0){top=Math.min(top,y);bottom=Math.max(bottom,y)}}return {top,bottom,height:c.height}})()")
        assert .75<(pixels['bottom']-pixels['top'])/pixels['height']<.9,pixels
        report['checks'].append(dict(fitted_animation=pixels))
        for theme in ('light','dark'):
            js('document.documentElement.dataset.theme='+json.dumps(theme));pause(300);window.grab().save(str(out/(theme+'.png')))
        click('保存但不应用偏移');wait("document.body.textContent.includes('完整结果与偏移写入完成')")
        set_mode('raw');click('开始录制');phase('recording');pause(1500);click('停止并收尾');phase('ready');click('保存录制（视频 + 音频 + 唇形）');wait("document.body.textContent.includes('MP4、WAV 与采集记录已保存并核对')")
        tracks=js('({created:window.m05R2.created,ended:window.m05R2.ended})');assert tracks['created']==tracks['ended'];report['checks'].append(dict(same_page_raw_restart_and_owned_track_cleanup=tracks))
        report['first_directory']=str(input_directory);report['success']=True
    finally:
        report['page_state']=js('document.body.innerText');report['user_agent']=js('navigator.userAgent')
        window.grab().save(str(out/'final.png'));QFileDialog.getExistingDirectory=original;QMessageBox.question=original_question
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8');window.accept_close();pause()
    print(json.dumps({k:v for k,v in report.items() if k!='page_state'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
