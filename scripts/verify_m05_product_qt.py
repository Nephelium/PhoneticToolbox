"""Actual Qt Workbench page + QWebChannel + persistent M05 task, no devices."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/src'),str(ROOT/'backend/src'),str(ROOT/'packages/phonetic_core/src')]
from PyQt6.QtCore import QEventLoop,QTimer,Qt
from PyQt6.QtWidgets import QApplication,QFileDialog,QMessageBox
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files

def main():
    out=ROOT/'output/validation/m05'/('qt-product-'+uuid4().hex);out.mkdir(parents=True)
    dist=out/'dist';shutil.copytree(ROOT/'frontend/dist',dist)
    shutil.copyfile(ROOT/'output/validation/m05/inputs/motion-occlusion-vfr/input.mkv',dist/'qa-public.mkv')
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    saved=out/'saved';saved.mkdir();os.environ['PTB_M05_PYTHON']=str(ROOT/'.venv/m05/Scripts/python.exe')
    register_scheme();app=QApplication(['M05-owned-product-QA']);window=Workbench(dist,test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal')
    window.setWindowTitle('M05 自动验收 · 短时测试摄像头 · 请勿操作此窗口')
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    if '--hidden' not in sys.argv:window.show()
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        box=[];loop=QEventLoop();window.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(5000,loop.quit);loop.exec();return box[0] if box else None
    def wait(code,seconds=60):
        start=time.monotonic()
        while time.monotonic()-start<seconds:
            if js(code):return
            pause()
        raise AssertionError('Qt condition timeout: '+code+'; '+str(js("document.body.innerText.slice(-1800)")))
    def click(text):
        assert js("(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==="+json.dumps(text)+");if(!b||b.disabled)return false;b.click();return true})()"),text
    report=dict(success=False,checks=[],physical_devices='--devices' in sys.argv)
    permission_timer=QTimer()
    permission_answer=QMessageBox.StandardButton.Yes
    def accept_owned_permission():
        for widget in app.topLevelWidgets():
            if isinstance(widget,QMessageBox) and widget.windowTitle()=='唇形采集权限':widget.button(permission_answer).click()
    permission_timer.timeout.connect(accept_owned_permission)
    if '--devices' in sys.argv:permission_timer.start(100)
    original=QFileDialog.getExistingDirectory
    original_question=QMessageBox.question
    if '--hidden' in sys.argv:
        QMessageBox.question=lambda *args,**kwargs:permission_answer
    try:
        wait("!!document.querySelector('nav')");click('唇形提取');wait("!!document.querySelector('.lip-page')")
        wait("document.querySelector('.lip-page').textContent.includes('正式离线分析')")
        js("(async()=>{const raw=await(await fetch('/qa-public.mkv')).blob();const d=new DataTransfer();d.items.add(new File([raw],'公开测试.mkv'));const input=document.querySelector('.lip-page input[type=file]');input.files=d.files;input.dispatchEvent(new Event('change',{bubbles:true}));})()")
        wait("document.querySelector('.lip-page').textContent.includes('分析所选 1 个视频')");click('分析所选 1 个视频')
        wait("!!document.querySelector('select[aria-label=\"选择唇形结果\"]')",90)
        report['checks'].append('actual Qt page/QWebChannel/HTTP/Job Object/legacy results')
        QFileDialog.getExistingDirectory=lambda *args,**kwargs:str(saved)
        click('应用偏移并保存');wait("document.querySelector('.lip-page').textContent.includes('完整结果与偏移写入完成')")
        folders=list(saved.iterdir());assert len(folders)==1 and (folders[0]/'frames.jsonl').is_file()
        meta=json.loads((folders[0]/'manifest.json').read_text('utf8'));assert meta['timing']['decoded_frames']==30
        report['checks'].append('actual Qt native directory grant and full streamed disk export')
        if '--devices' in sys.argv:
            if '--permission-retry' in sys.argv:
                report['permission_attempts']=[]
                for attempt in range(2):
                    js("(()=>{const s=document.querySelector('.module-toolbar select');s.value='preview';s.dispatchEvent(new Event('change',{bubbles:true}));})()")
                    permission_answer=QMessageBox.StandardButton.No
                    click('打开预览')
                    wait("document.querySelector('.lip-state').textContent.includes('阶段 failed')")
                    wait("document.querySelector('.lip-page').textContent.includes('未允许本次摄像头和麦克风采集')")
                    report['permission_attempts'].append(dict(attempt=attempt,events=list(window.m05_media.events)))
                    assert window.m05_media.status()['decision']=='denied_user'
                permission_answer=QMessageBox.StandardButton.Yes
                for attempt in range(2):
                    js("(()=>{const s=document.querySelector('.module-toolbar select');s.value='preview';s.dispatchEvent(new Event('change',{bubbles:true}));})()")
                    click('打开预览')
                    wait("document.querySelector('.lip-state').textContent.includes('阶段 previewing')",70)
                    pause(1500)
                    click('停止并收尾')
                    wait("document.querySelector('.lip-state').textContent.includes('阶段 ready')")
                    report['permission_attempts'].append(dict(attempt=attempt+2,events=list(window.m05_media.events)))
                report['checks'].append('deny twice then allow; preview stop/restart and transition to raw recording')
            for mode in ('raw','realtime','record_then_analyze'):
                js("(()=>{const s=document.querySelector('.module-toolbar select');s.value="+json.dumps(mode)+";s.dispatchEvent(new Event('change',{bubbles:true}));const input=[...document.querySelectorAll('label')].find(e=>e.textContent.includes('请求帧率')).querySelector('input');input.value="+json.dumps('60' if mode=='record_then_analyze' else '30')+";input.dispatchEvent(new Event('input',{bubbles:true}));})()")
                click('开始录制');wait("document.querySelector('.lip-state').textContent.includes('阶段 recording')",70)
                pause(8000)
                if mode=='realtime':window.grab().save(str(out/'physical-mesh.png'))
                click('停止并收尾');wait("document.querySelector('.lip-state').textContent.includes('阶段 ready')")
                click('保存原始录制');wait("document.querySelector('.lip-page').textContent.includes('原始录制写入完成并核对大小')")
                click('保存候选参数与时间记录');wait("document.querySelector('.lip-page').textContent.includes('候选参数与时间元数据已保存')")
                report['checks'].append('physical Qt '+mode+' explicit media permission / encoding / native disk save')
        window.grab().save(str(out/'host.png'))
        report['user_agent']=js('navigator.userAgent');report['success']=True
    finally:
        report['permission_events']=window.m05_media.events
        window.grab().save(str(out/'final-state.png'))
        report['page_state']=js("document.querySelector('.lip-page')?.innerText")
        permission_timer.stop()
        QFileDialog.getExistingDirectory=original
        QMessageBox.question=original_question
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        window.accept_close();pause()
    print(json.dumps(dict(output=str(out),**report),ensure_ascii=False))

if __name__=='__main__':main()
