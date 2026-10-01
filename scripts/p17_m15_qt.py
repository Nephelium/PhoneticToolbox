"""P17 actual Qt Workbench, ptbapp bundle, authorized real recording, no task database.

The test supplies native save-dialog destinations; production still asks the user.
No system volume/device changes. Only the normal product preflight beep may be generated; stimulus is a real recording.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
for folder in ('packages/phonetic_core/src', 'backend/src', 'desktop/src'):
    sys.path.insert(0, str(ROOT / folder))
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog
from ptb_desktop.host import Workbench, register_scheme


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', type=Path, default=ROOT / 'frontend/dist')
    args = parser.parse_args()
    out = ROOT / 'output/validation/p17/M15-qt' / uuid4().hex
    out.mkdir(parents=True)
    register_scheme()
    app = QApplication(['M15 Qt verification'])
    window = Workbench(args.dist, test=True, jobs_path=None, vocal_resources=ROOT / 'resources/vocal_tract/native')
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.showMaximized()
    downloads = []
    def destination(parent, title, suggested, filter, **kwargs):
        target = out / Path(suggested).name
        downloads.append(str(target))
        return str(target), filter
    QFileDialog.getSaveFileName = destination
    button = lambda name: "[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===" + json.dumps(name) + ")"
    click = lambda name: button(name) + '?.click()'
    phase = lambda name: "document.querySelector('.m15-run')?.dataset.phase===" + json.dumps(name)
    source=json.loads((ROOT/'output/validation/p17/natural-inventory.json').read_text('utf8'))['selected']['short']
    raw=Path(source['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==source['sha256']
    encoded=base64.b64encode(raw).decode()
    import_file = "(()=>{const data=Uint8Array.from(atob("+json.dumps(encoded)+"),c=>c.charCodeAt(0));const dt=new DataTransfer();dt.items.add(new File([data],'P17-real.wav',{type:'audio/wav'}));const input=document.querySelector('[aria-label=\"导入刺激 X\"]');input.files=dt.files;input.dispatchEvent(new Event('change',{bubbles:true}));})()"
    fill_params = r"""(()=>{for(const label of ['ISI 毫秒','试次间隔毫秒']){const e=document.querySelector('[aria-label="'+label+'"]');e.value='0';e.dispatchEvent(new Event('input',{bubbles:true}));}const labels=[...document.querySelectorAll('label')];labels.find(l=>l.textContent.trim()==='提示音').querySelector('input').click();})()"""
    confirm_ready = r"""(()=>{for(const label of ['已听见试音，设备正确','已暂停其他录制、播放及重型任务']){const e=[...document.querySelectorAll('label')].find(l=>l.textContent.trim()===label).querySelector('input');e.click();}})()"""
    fill_questions = r"""(()=>{const e=document.querySelector('[aria-label="姓名/编号"]');e.value='P17 automated session';e.dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('input[type=radio][value="女"]').click();})()"""
    stages = [
        ("document.querySelector('.host-badge')?.textContent==='本地桌面'", "[...document.querySelectorAll('.nav-item')].find(b=>b.textContent.includes('感知实验')).click()"),
        ("document.querySelector('.perception-page')?.getAttribute('aria-busy')==='false'", import_file),
        ("!!document.querySelector('.m15-asset')&&document.querySelector('.perception-page').getAttribute('aria-busy')==='false'", click('参数')),
        ("!!document.querySelector('[aria-label=\"ISI 毫秒\"]')", fill_params),
        ("document.querySelector('[aria-label=\"ISI 毫秒\"]')?.value==='0'", '__click:预检与试音'),
        ("document.body.textContent.includes('预检通过')", confirm_ready),
        (button('填写被试信息') + '?.disabled===false', click('填写被试信息')),
        ("!!document.querySelector('[aria-label=\"姓名/编号\"]')", fill_questions),
        ("document.querySelector('[aria-label=\"姓名/编号\"]')?.value==='P17 automated session'", click('建立独立会话')),
        (phase('intro'), '__click:开始实验'),
        (phase('responding'), '__key'),
        (phase('completed'), click('完整溯源 JSON')),
        ("document.body.textContent.includes('下载请求已发出')", '__capture'),
    ]
    report = {'success':False, 'host':'actual Windows Qt Workbench / ptbapp', 'database_operations':'none', 'stages':[], 'physical_timing_measured':False, 'save_dialog':'destination supplied by test','source':source,'screen':{'size':[app.primaryScreen().size().width(),app.primaryScreen().size().height()],'available':[app.primaryScreen().availableGeometry().width(),app.primaryScreen().availableGeometry().height()],'dpr':app.primaryScreen().devicePixelRatio()},'window_maximized':window.isMaximized(),'visibility':'WA_DontShowOnScreen actual Qt maximized host','heard_checkbox':'automated only; human hearing NOT verified'}
    index, pending, started = 0, False, time.monotonic()
    def finish(success, message=None):
        timer.stop()
        report.update(success=success,error=message,downloads=downloads)
        def done(value):
            report['diagnostics']=value
            for file in out.glob('*.json'):
                if file.name!='report.json':
                    try:
                        data=json.loads(file.read_text(encoding='utf8'))
                        report['export_readback']={'attempts':len(data['attempts']), 'status':data['attempts'][0]['status'], 'roles':data['attempts'][0]['order'], 'original_sample_rate':data['attempts'][0]['roles']['X']['originalSampleRate']}
                    except Exception as exc: report['readback_error']=str(exc)
            if success and report.get('export_readback',{}).get('status')!='completed':report['success']=False;report['error']='download_readback_failed'
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
            window.closing=True
            window.close()
        window.page.runJavaScript("({body:document.body.innerText,secure:isSecureContext,locks:!!navigator.locks,indexedDB:!!window.indexedDB,audio:!!window.AudioContext,resources:performance.getEntriesByType('resource').map(e=>e.name)})",done)
    def tick():
        nonlocal index,pending
        if pending:return
        if time.monotonic()-started>70:finish(False,f'timeout_stage_{index+1}');return
        pending=True
        def ready(value):
            nonlocal index,pending
            pending=False
            if not value:return
            action=stages[index][1];report['stages'].append(index+1);index+=1
            if action.startswith('__click:'):
                target=button(action.split(':',1)[1])
                def mouse(rect):
                    if rect:QTest.mouseClick(window.view.focusProxy(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(rect[0]),round(rect[1])))
                window.page.runJavaScript('(()=>{const r='+target+'.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()',mouse)
            elif action=='__key':QTest.keyClick(window.view.focusProxy(),Qt.Key.Key_J)
            elif action=='__capture':window.view.grab().save(str(out/'qt-completed.png'))
            else:window.page.runJavaScript(action)
            if index==len(stages):timer.stop();QTimer.singleShot(1500,lambda:finish(True))
        window.page.runJavaScript(stages[index][0],ready)
    timer=QTimer();timer.timeout.connect(tick);timer.start(80)
    app.exec()
    print(out/'report.json')
    if not report['success']:raise SystemExit(1)

if __name__=='__main__':main()
