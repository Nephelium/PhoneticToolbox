"""Actual isolated Qt: MP4/MKV import, automatic compatible replay and MP4 save."""
import base64
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --mute-audio')
from PyQt6.QtCore import QEventLoop, QTimer, Qt, QPoint, QEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog
from verification_artifacts import verification_run
from workbench_source import configure

ROOT = Path(__file__).resolve().parents[1]


def main():
    configure()
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files
    register_scheme()
    app = QApplication(['M05 compatible replay QA'])
    with verification_run('m05-playback', 'qt', 'NASA astronaut public fixture; two 6-frame FFV1 MKV inputs, H264/AAC MP4 derivative; 1s 440Hz sine; actual Qt/worker/encoder; no physical devices') as (out, scratch):
        print(out, flush=True)
        inputs = scratch / 'inputs'
        inputs.mkdir()
        saved = scratch / 'saved'
        saved.mkdir()
        runtime = os.environ['PTB_M05_PYTHON']
        subprocess.run([runtime, '-B', '-X', 'utf8', str(ROOT / 'tests/support/m05_offline_media.py'), 'generate', str(inputs)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        mp4 = scratch / 'mp4'
        mp4.mkdir()
        subprocess.run([runtime, '-B', '-X', 'utf8', '-c', 'import sys;sys.path[:0]=[sys.argv[1],sys.argv[2]];from ptb_worker.m05_media_export import recording_bundle;recording_bundle(sys.argv[3],sys.argv[4])', str(ROOT / 'backend/src'), str(ROOT / 'packages/phonetic_core/src'), str(inputs / 'first.mkv'), str(mp4)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        subprocess.run([runtime, '-B', '-c', 'import av,sys\nwith av.open(sys.argv[1]) as source,av.open(sys.argv[2],"w") as target:\n streams={s.index:target.add_stream_from_template(s) for s in source.streams}\n for packet in source.demux():\n  if packet.dts is not None:\n   packet.stream=streams[packet.stream.index];target.mux(packet)', str(inputs / 'first.mkv'), str(inputs / 'common.avi')], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        db = scratch / 'jobs.sqlite3'
        with sqlite3.connect((ROOT / 'output/validation/p06/local-state.sqlite3').as_uri() + '?mode=ro', uri=True) as source, sqlite3.connect(db) as target:
            source.backup(target)
            target.execute('DELETE FROM events')
            target.execute('DELETE FROM jobs')
            target.commit()
        cache = scratch / 'cache'
        cache.mkdir()
        initialize_local_files(cache)
        window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db, local_files_root=cache, vocal_profile=scratch / 'vocal', start_module='M05')
        calls = []
        invoke = window.bridge.tasks.m05.invoke

        def track(body):
            calls.append(body.copy())
            return invoke(body)

        window.bridge.tasks.m05.invoke = track
        window.resize(1440, 900)
        window.show()
        report = dict(success=False, scope='Windows source / actual offscreen Qt / synthetic public media; decoded audio signal, no physical speaker or capture test', checks=[])
        original_picker = QFileDialog.getExistingDirectory
        QFileDialog.getExistingDirectory = lambda *args, **kwargs: str(saved)

        def pause(ms=80):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
            return values[0] if values else None

        def until(code, seconds=90):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText')))

        def click(text):
            code = '(()=>{const b=[...document.querySelectorAll(".lip-page button")].find(b=>b.textContent.trim()===' + json.dumps(text) + ');if(!b||b.disabled)return null;b.scrollIntoView({block:"nearest"});const r=b.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()'
            point = js(code)
            assert point, text
            QTest.mouseClick(window.view.focusProxy(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(round(point[0]), round(point[1])))
            pause()

        def select(selector, value):
            js('(()=>{const e=document.querySelector(' + json.dumps(selector) + ');e.value=' + json.dumps(value) + ';e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            pause()

        media_selector = '.lip-recording-preview'
        try:
            until('!!document.querySelector(".lip-page")')
            assert js('[...document.querySelector("[aria-label=采集模式]").options].map(o=>o.textContent)') == ['预览', '实时计算', '视频录制', '离线分析']
            assert js('document.querySelector("input[type=file]").accept') == '.mp4,.mov,.avi,.mkv,.wmv,.m4v,.webm'
            select('[aria-label=采集模式]', 'offline')
            for source in (mp4 / 'raw_recording.mp4', inputs / 'first.mkv', inputs / 'common.avi'):
                encoded = base64.b64encode(source.read_bytes()).decode('ascii')
                js('(()=>{const d=new DataTransfer();d.items.add(new File([Uint8Array.from(atob(' + json.dumps(encoded) + '),c=>c.charCodeAt(0))],' + json.dumps(source.name) + '));const e=document.querySelector("input[type=file]");e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
                before = len([b for b in calls if b['op'] == 'm05_playback_info'])
                previous_url = js('document.querySelector(".lip-recording-preview")?.getAttribute("src")')
                click('分析所选文件')
                until('document.querySelector(".lip-page").getAttribute("aria-busy")==="false"')
                until('document.querySelector(".lip-metrics strong")?.textContent!=="—"')
                until('(()=>{const v=document.querySelector(".lip-recording-preview");return v&&v.getAttribute("src")!==' + json.dumps(previous_url) + '&&v.currentSrc===v.getAttribute("src")&&v.readyState>=2})()')
                conversions = len([b for b in calls if b['op'] == 'm05_playback_info']) - before
                assert conversions == 1
                assert not window.bridge.tasks.m05.playbacks
                assert js('document.querySelector(".workbench-left").innerText.includes("解码 6")&&document.querySelector(".workbench-left").innerText.includes("检出 6")')
                js('(()=>{const v=document.querySelector(".lip-recording-preview");if(!window.qaAudio){window.qaAudio=new AudioContext();window.qaAnalyser=qaAudio.createAnalyser();qaAudio.createMediaElementSource(v).connect(qaAnalyser);qaAnalyser.connect(qaAudio.destination);}qaAudio.resume();})()')
                click('同步回放')
                until('document.querySelector(".lip-recording-preview").currentTime>.25')
                signal = js('(()=>{const a=new Float32Array(qaAnalyser.fftSize);qaAnalyser.getFloatTimeDomainData(a);return Math.max(...a.map(Math.abs));})()')
                assert signal > .01, signal
                assert js('Number(document.querySelector("[aria-label=回放帧]").value)>0')
                js('document.querySelector(".lip-recording-preview").pause()')
                js('(()=>{const v=document.querySelector(".lip-recording-preview");v.currentTime=.4;})()')
                until('Number(document.querySelector("[aria-label=回放帧]").value)>=3')
                state = js('(()=>{const v=document.querySelector(".lip-recording-preview");return {width:v.videoWidth,height:v.videoHeight,decoded:v.webkitDecodedFrameCount,time:v.currentTime,lip:document.querySelector("[aria-label=回放帧]").value};})()')
                assert state['width'] > 0 and state['decoded'] > 1
                click('检查偏移量')
                until('!!document.querySelector(".offset-editor")&&!document.querySelector(".offset-editor [role=status]")')
                js('(()=>{const e=document.querySelector(' + json.dumps('[aria-label="唇形偏移量 ms"]') + ');e.value="125";e.dispatchEvent(new Event("input",{bubbles:true}));})()')
                click('应用偏移量')
                js('(()=>{const e=document.querySelector("[aria-label=回放帧]");e.value="3";e.dispatchEvent(new Event("input",{bubbles:true}));})()')
                until('Math.abs(document.querySelector(".lip-recording-preview").currentTime-.425)<.01')
                report['checks'].append(dict(input=source.suffix, replay=state, decoded_audio_peak=signal, native_scratch_released=True))
                print(json.dumps(report['checks'][-1]), flush=True)
            # The original selected source is exported as MP4, regardless of the
            # temporary WebM derivative used by Qt for playback.
            js('(()=>{const e=document.querySelector(".lip-save-options label:last-child input");e.checked=false;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            click('保存（不应用偏移）')
            until('document.querySelector(".lip-page").getAttribute("aria-busy")==="false"')
            bundles = list(saved.glob('M05-*'))
            assert len(bundles) == 1 and (bundles[0] / 'raw_recording.mp4').is_file()
            assert not list(bundles[0].glob('*.webm'))
            report['checks'].append('Source AVI saves as raw_recording.mp4; temporary playback WebM is not exported')
            for theme in ('light', 'dark'):
                for width in (1440, 1100, 900):
                    window.resize(width, 900)
                    js('document.documentElement.dataset.theme=' + json.dumps(theme))
                    pause(350)
                    assert js('document.documentElement.scrollWidth<=innerWidth+1')
                    if width == 1440:
                        window.view.grab().save(str(out / (theme + '.png')))
            report['checks'].append('Light/dark: 1440/1100/900 widths, left analysis counts visible, no horizontal overflow')
            # Public synthetic camera fixture only; no physical device request.
            js('window.qaAudio?.close();window.qaAudio=null')
            click('放弃未保存内容')
            click('放弃')
            js((ROOT / 'tests/support/m05_r2_capture.js').read_text('utf8'))
            portrait = 'data:image/png;base64,' + base64.b64encode((ROOT / 'output/validation/m05/inputs/astronaut.png').read_bytes()).decode('ascii')
            js('window.installM05R2Capture(' + json.dumps(portrait) + ').then(()=>window.qaFixtureReady=true)')
            until('window.qaFixtureReady')
            select('[aria-label=采集模式]', 'record_then_analyze')
            click('开始录制')
            until('Number(document.querySelector(".lip-state").textContent.match(/呈现 (\\d+)/)?.[1])>8')
            pause(1200)
            click('结束录制')
            until('document.querySelector(".lip-state").dataset.phase==="ready"')
            until('document.querySelector(".lip-page").getAttribute("aria-busy")==="false"')
            counts = js('document.querySelector(".lip-state").textContent')
            select('[aria-label=采集模式]', 'offline')
            assert js('document.querySelector(".workbench-left .lip-state").textContent') == counts
            for label in ('呈现', '已推理', '检出', '跳过推理', '呈现缺口'):
                assert label in counts
            js('document.querySelector(".lip-stat-help").open=true')
            window.resize(1440, 900)
            pause(300)
            window.view.grab().save(str(out / 'retained-capture-statistics.png'))
            report['checks'].append(dict(retained_capture_stats=counts, capture_mode='record_then_analyze', mode='offline', definitions_visible=True, physical_devices=False))
            report['success'] = True
        finally:
            js('window.qaAudio?.close()')
            report['operations'] = [b['op'] for b in calls]
            report['page'] = js('document.body.innerText')
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf8')
            window.view.grab().save(str(out / 'final.png'))
            QFileDialog.getExistingDirectory = original_picker
            window.accept_close()
            pause(300)
            window.page.deleteLater()
            app.processEvents()
            window.profile.deleteLater()
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
        print(json.dumps(dict(success=report['success'], out=str(out), checks=len(report['checks']))), flush=True)


if __name__ == '__main__':
    main()
