"""M05 device selection in actual isolated Qt, without opening physical devices."""
import json
import os
from pathlib import Path
import sqlite3
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --mute-audio')
from PyQt6.QtCore import QEvent, QEventLoop, QTimer
from PyQt6.QtWebEngineCore import QWebEngineScript
from PyQt6.QtWidgets import QApplication
from verification_artifacts import verification_run
from workbench_source import configure

ROOT = Path(__file__).resolve().parents[1]


def main():
    configure()
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files
    register_scheme()
    app = QApplication(['M05 device selection QA'])
    with verification_run('m05-devices', 'qt', 'Actual Qt shared page; synthetic microphone, USB sound card/line input, loopback and speaker enumeration; getUserMedia intercepted before opening any device') as (out, scratch):
        print(out, flush=True)
        db = scratch / 'jobs.sqlite3'
        with sqlite3.connect(db) as connection:
            for name in ('002_jobs_sqlite.sql', '005_acoustic_batches_sqlite.sql'):
                connection.executescript((ROOT / 'backend/migrations' / name).read_text('utf-8'))
        cache = scratch / 'cache'
        cache.mkdir()
        initialize_local_files(cache)
        window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db, local_files_root=cache, vocal_profile=scratch / 'vocal', start_module='M05')
        script = QWebEngineScript()
        script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        script.setSourceCode('''
window.qaDeviceCalls=[];
window.qaDevicesUnlocked=false;window.qaDeviceStops=0;
const devices=[
 {kind:'videoinput',deviceId:'camera',label:'Test camera'},
 {kind:'audioinput',deviceId:'mic',label:'麦克风阵列 (Realtek Audio)'},
 {kind:'audioinput',deviceId:'voice',label:'Voice (Steinberg UR22C)'},
 {kind:'audioinput',deviceId:'streaming',label:'Streaming (Steinberg UR22C)'},
 {kind:'audioinput',deviceId:'loopback',label:'立体声混音'},
 {kind:'audioinput',deviceId:'line',label:'Input 1/2 (Steinberg UR22C)'},
 {kind:'audiooutput',deviceId:'speaker',label:'Test speaker'}];
Object.defineProperty(navigator.mediaDevices,'enumerateDevices',{value:async()=>qaDevicesUnlocked?devices:[devices[0],{kind:'audioinput',deviceId:'',label:''}]});
Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:constraints=>{
 qaDeviceCalls.push(constraints);
 if(constraints.video===false)return new Promise((resolve,reject)=>{
  window.qaAllowDevices=()=>{window.qaDevicesUnlocked=true;resolve({getTracks:()=>[{stop:()=>window.qaDeviceStops++}]});};
  window.qaRejectDevices=()=>reject(new DOMException('Synthetic permission denied','NotAllowedError'));
 });
 return new Promise((resolve,reject)=>{window.qaRejectCapture=()=>reject(new DOMException('Synthetic device unavailable','NotReadableError'));});
}});
''')
        window.page.scripts().insert(script)
        window.resize(1440, 900)
        window.show()
        report = dict(success=False, scope='Windows source / actual Qt / synthetic enumeration and acquisition interception; no physical devices opened', checks=[])

        def pause(ms=60):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
            return values[0] if values else None

        def until(code, seconds=25):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText')))

        def select(label, value):
            js('(()=>{const e=document.querySelector(' + json.dumps('[aria-label="' + label + '"]') + ');if(!e||e.disabled)throw Error("Disabled select");e.value=' + json.dumps(value) + ';e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            pause()

        mic = 'document.querySelector("[aria-label=麦克风]")'
        try:
            until('!!' + mic + '&&' + mic + '.options.length===1')
            assert js('qaDeviceCalls.length') == 0, 'Mount must not request audio'
            assert '未授权麦克风' not in js(mic + '.textContent')
            def discover():
                js('[...document.querySelectorAll(".lip-page button")].find(b=>b.textContent.trim()==="授权并刷新设备").click()')
                until('!!window.qaRejectDevices')
                assert js(mic + '.disabled'), 'Discovery locks inputs until temporary audio is released'
            discover()
            assert js('qaDeviceCalls.at(-1)') == dict(audio=True, video=False)
            js('qaRejectDevices();window.qaRejectDevices=null')
            until('!' + mic + '.disabled')
            assert '未获得麦克风权限' in js('document.querySelector(".lip-page").innerText')
            discover()
            js('qaAllowDevices();window.qaRejectDevices=null')
            until(mic + '.options.length===6&&!'+mic+'.disabled')
            assert js('qaDeviceStops') == 1
            assert window.m05_media.purpose == 'devices', 'Shared desktop port must arm audio discovery purpose'
            report['checks'].append('Hidden device list on mount; explicit audio-only authorisation, refusal/retry, five named inputs and immediate temporary track release')
            assert not js(mic + '.disabled'), 'Preview must allow microphone preselection'
            assert js('[...document.querySelector("[aria-label=麦克风]").options].map(o=>o.value)') == ['', 'mic', 'voice', 'streaming', 'loopback', 'line']
            select('麦克风', 'line')
            js('[...document.querySelectorAll(".lip-page button")].find(b=>b.textContent.trim()==="刷新设备").click()')
            pause()
            assert js(mic + '.value') == 'line'
            assert js('qaDeviceCalls.length') == 2, 'Selection and already-authorised refresh must never reopen devices'
            report['checks'].append('Preview selects USB sound card/line input; all audio inputs included, speakers excluded; refresh retains choice without capture')
            for mode in ('preview', 'realtime', 'record_then_analyze'):
                select('采集模式', mode)
                assert not js(mic + '.disabled') and js(mic + '.value') == 'line'
                js('[...document.querySelectorAll(".lip-page button")].find(b=>["打开预览","开始录制"].includes(b.textContent.trim())).click()')
                until('!!window.qaRejectCapture')
                assert js(mic + '.disabled'), 'Opening capture locks device selection'
                constraints = js('qaDeviceCalls.at(-1)')
                if mode == 'preview':
                    assert constraints['audio'] is False
                else:
                    assert constraints['audio']['deviceId'] == {'exact': 'line'}
                    assert all(constraints['audio'][key] is False for key in ('echoCancellation', 'noiseSuppression', 'autoGainControl'))
                js('qaRejectCapture();window.qaRejectCapture=null')
                until('!' + mic + '.disabled')
                assert js(mic + '.value') == 'line'
                report['checks'].append(dict(mode=mode, constraints=constraints, locked_while_opening=True, selected_after_failure='line'))
            select('采集模式', 'offline')
            assert js(mic) is None
            select('采集模式', 'preview')
            assert js(mic + '.value') == 'line'
            for theme in ('light', 'dark'):
                js('document.documentElement.dataset.theme=' + json.dumps(theme))
                pause(250)
                assert js('document.documentElement.scrollWidth<=innerWidth+1')
                assert js('document.querySelector(".lip-page").innerText.includes("预览仅使用摄像头，所选麦克风用于录制。")')
                window.view.grab().save(str(out / (theme + '.png')))
            report['checks'].append('Offline hides device controls; returning retains sound card; light/dark layouts and preview hint pass')
            report['success'] = True
        finally:
            report['page'] = js('document.body.innerText')
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
            window.accept_close()
            pause(150)
            window.page.deleteLater()
            app.processEvents()
            window.profile.deleteLater()
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
        print(json.dumps(dict(success=report['success'], checks=len(report['checks']), out=str(out))), flush=True)


if __name__ == '__main__':
    main()
