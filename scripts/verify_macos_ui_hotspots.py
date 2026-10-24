"""Actual frozen/source Mac IPA, reader and native presentation acceptance."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main(output, fixtures=None):
    if sys.platform != 'darwin':
        raise RuntimeError('This acceptance requires macOS')
    from PyQt6.QtCore import QEventLoop, QPoint, QPointF, QTimer, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench
    from ptb_desktop.presentation import FirstMaximizePresentation
    from ptb_desktop.web_assets import register_scheme
    from ptb_worker.local_workspace import prepare_workspace
    from PyQt6.QtWidgets import QFileDialog

    root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    scratch = out / 'scratch'
    fixture_root = Path(fixtures).resolve() if fixtures else root / 'tests/fixtures'
    database, files = prepare_workspace(scratch / 'workspace', root / 'backend/migrations')
    register_scheme()
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(['PTB macOS final UI acceptance'])
    app.setQuitOnLastWindowClosed(False)
    window = Workbench(root / 'frontend/dist', test=True, jobs_path=database,
        local_files_root=files, vocal_profile=scratch / 'vocal', start_module='M17',
        reaper_binary=root / 'resources/research/reaper', vocal_resources=root / 'resources/vocal_tract/native')
    report = dict(success=False, frozen=bool(getattr(sys, 'frozen', False)), checks=[], timings=[],
        scope='Actual macOS Qt default GPU, isolated workspace/profile; real application controls, native Qt wheel and window states; no physical devices',
        chromiumFlags=os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS'), rendererTerminations=[], presentation=dict(attachMs=[], scans=[]))
    report['inputIdentity'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ('frontend/dist/index.html', 'resources/macos-validation.json',
            'desktop/src/ptb_desktop/host.py', 'desktop/src/ptb_desktop/presentation.py',
            'desktop/src/ptb_desktop/m05_bridge.py', 'scripts/verify_macos_ui_hotspots.py',
            'scripts/verify_release_lifecycle_ui.py', 'scripts/m16_test_backend.py') if (root / name).is_file()}
    window.page.renderProcessTerminated.connect(lambda status, code: report['rendererTerminations'].append([str(status), code]))
    original_attach, original_find = window._first_maximize.attach, window.view.findChildren

    def attach():
        start = time.perf_counter()
        try:
            return original_attach()
        finally:
            report['presentation']['attachMs'].append(round((time.perf_counter() - start) * 1000, 3))

    def find(*values, **kwargs):
        started = time.perf_counter()
        try:
            return original_find(*values, **kwargs)
        finally:
            report['presentation']['scans'].append(dict(ms=round((time.perf_counter()-started)*1000, 3),
                directOnly=kwargs.get('options') == Qt.FindChildOption.FindDirectChildrenOnly))

    window._first_maximize.attach = attach
    window.view.findChildren = find

    def pause(ms=100):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def js(code):
        loop, values = QEventLoop(), []
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        started = time.perf_counter()
        window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
        if not values:
            timer.start(5000)
            loop.exec()
        timer.stop()
        elapsed = (time.perf_counter()-started)*1000
        assert values and elapsed < 5000, ('Qt JavaScript responsiveness', elapsed, code)
        return values[0]

    def until(expression, seconds=30):
        end = time.monotonic() + seconds
        while not js(expression):
            assert time.monotonic() < end, expression
            pause(50)

    def click(selector):
        assert js('(()=>{const e=document.querySelector('+json.dumps(selector)+');if(!e)return false;e.click();return true})()'), selector

    def label(value):
        assert js('(()=>{const e=[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(value)+');if(!e||e.disabled)return false;e.click();return true})()'), value

    def phase(name):
        with (out / 'phases.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(stage=name, monotonic=time.monotonic()))+'\n')
        print(json.dumps(dict(stage=name)), flush=True)

    def measured(name, callback):
        start = time.perf_counter()
        callback()
        report['timings'].append(dict(name=name, ms=round((time.perf_counter()-start)*1000, 3)))

    def wheel(selector):
        position = js('(()=>{const e=document.querySelector('+json.dumps(selector)+'),r=e.getBoundingClientRect();e.scrollTop=0;return {x:r.x+r.width*.6,y:r.y+r.height*.5,max:e.scrollHeight-e.clientHeight}})()')
        assert position['max'] > 0, position
        target = window.view.focusProxy() or window.view
        point = QPoint(round(position['x']), round(position['y']))
        global_point = window.view.mapToGlobal(point)
        local = target.mapFromGlobal(global_point)
        QTest.mouseMove(target, local)
        pause(80)
        for index, delta in enumerate((-120, -120, -120)):
            event = QWheelEvent(QPointF(local), QPointF(global_point), QPoint(0, 0), QPoint(0, delta),
                Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.ScrollBegin if index == 0 else Qt.ScrollPhase.ScrollUpdate, False)
            QApplication.sendEvent(target, event)
            pause(100)
        QApplication.sendEvent(target, QWheelEvent(QPointF(local), QPointF(global_point), QPoint(), QPoint(),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollEnd, False))
        pause(200)
        after = js('document.querySelector('+json.dumps(selector)+').scrollTop')
        assert after > 0, (selector, after)
        return dict(selector=selector, after=after, max=position['max'])

    def lifecycle():
        """Real ports and decoded media; only devices and response timing are synthetic."""
        import base64
        from fractions import Fraction
        import threading
        import av
        import numpy as np
        import soundfile as sf
        from m16_test_backend import Backend

        inputs, project = scratch / 'lifecycle-inputs', scratch / 'recording-project'
        inputs.mkdir()
        project.mkdir()
        samples = np.load(fixture_root / 'm07/v2.npz')
        spec = json.loads((fixture_root / 'm07/v2.json').read_text())
        for index in (0, 1):
            sf.write(inputs / f'input{index}.wav', samples[f'input{index}'].astype(float) / 32768,
                     spec['cases'][index]['fs'], subtype='FLOAT')
        video = inputs / 'neutral-no-face.mkv'
        with av.open(str(video), 'w') as container:
            stream = container.add_stream('ffv1', rate=30)
            stream.width, stream.height, stream.pix_fmt = 320, 240, 'yuv420p'
            audio = container.add_stream('pcm_s16le', rate=48000)
            audio.layout = 'mono'
            for index in range(60):
                frame = av.VideoFrame.from_ndarray(np.full((240, 320, 3), 24 + index % 32, dtype='uint8'), format='rgb24')
                frame.pts = index
                for packet in stream.encode(frame):
                    container.mux(packet)
            wave = np.int16(np.sin(np.arange(96000) * (2 * np.pi * 220 / 48000)) * 4096)[None, :]
            frame = av.AudioFrame.from_ndarray(wave, format='s16', layout='mono')
            frame.sample_rate, frame.pts, frame.time_base = 48000, 0, Fraction(1, 48000)
            for packet in audio.encode(frame):
                container.mux(packet)
            for encoder in (stream, audio):
                for packet in encoder.encode():
                    container.mux(packet)

        gate, reached = threading.Event(), threading.Event()
        control = dict(kind=None)
        calls = []
        original_task = window.bridge.tasks.invoke
        recording = window.bridge.recording_bridge()
        original_recording, original_choose = recording.dispatch, recording.choose
        original_picker = QFileDialog.getExistingDirectory
        recording.service.backend = Backend()
        recording.choose = lambda purpose: recording.service.grant(project, purpose)
        QFileDialog.getExistingDirectory = lambda *args, **kwargs: str(inputs)
        window.page.setAudioMuted(True)

        def hold(kind):
            if control['kind'] == kind:
                control['kind'] = None
                reached.set()
                assert gate.wait(30), 'Controlled response latch timed out'

        def task(body):
            value = original_task(body)
            calls.append(body['op'])
            if body['op'] == 'm05_playback_info':
                hold('M05')
            elif body['op'] == 'result' and control['kind'] == 'M07':
                raw = base64.b64decode(value.get('base64', ''))
                if raw.startswith(b'RIFF'):
                    hold('M07')
            return value

        def dispatch(body):
            value = original_recording(body)
            calls.append('recording:' + body['op'])
            if body['op'] == 'play':
                hold('M16')
            return value

        def arm(kind):
            gate.clear()
            reached.clear()
            control['kind'] = kind

        def wait_reached():
            end = time.monotonic() + 120
            while not reached.is_set():
                assert time.monotonic() < end, ('Response latch was not reached', calls[-20:])
                pause(50)

        def change(selector, value):
            assert js('(()=>{const e=document.querySelector('+json.dumps(selector)+');if(!e)return false;e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'), selector
            pause(50)

        def choose_label(text, value):
            assert js('(()=>{const l=[...document.querySelectorAll(".recording-page label")].find(e=>e.textContent.trim().startsWith('+json.dumps(text)+'));const e=l?.querySelector("select");if(!e)return false;e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'), text
            pause(50)

        def button(selector):
            until('!!document.querySelector('+json.dumps(selector)+')&&!document.querySelector('+json.dumps(selector)+').disabled')
            point = js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.scrollIntoView({block:"nearest"});const r=e.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
            target = window.view.focusProxy() or window.view
            QTest.mouseClick(target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                             QPoint(round(point[0]), round(point[1])))
            pause(50)

        def away_and_back(module):
            click('#tab-M17')
            until('!!document.querySelector(".m17-body")?.offsetParent')
            pause(100)
            click('#tab-' + module)
            pause(100)

        window.bridge.tasks.invoke = task
        recording.dispatch = dispatch
        result = dict(scope='Actual macOS Qt/QWebChannel and unchanged port responses with controlled latches; synthetic PortAudio, neutral no-face video and fixed M07 fixtures; muted output; no physical capture or speaker claim', checks=[])
        try:
            phase('lifecycle-m16-start')
            label('录音')
            until('!!document.querySelector(".recording-page")?.offsetParent')
            label('新建工程')
            until('document.body.innerText.includes("本地录音工程已建立")')
            label('设备与录前检测')
            device = recording.service.dispatch(dict(op='devices'))[0]['id']
            choose_label('输入设备', device)
            choose_label('扬声器 / 耳机', device)
            label('收起')
            button('.recording-controls .record-button')
            pause(1800)
            label('■ 停止录音')
            until('document.body.innerText.includes("已存入本地工程，尚未导出")')
            arm('M16')
            button('.recording-page .play-button')
            wait_reached()
            away_and_back('M16')
            gate.set()
            until('document.querySelector(".recording-page .play-button")?.textContent.trim()==="▶ 播放"&&!document.querySelector(".recording-page .play-button")?.disabled')
            pause(200)
            assert recording.service.player is None
            button('.recording-page .play-button')
            until('document.querySelector(".recording-page .play-button")?.textContent.includes("停止播放")')
            assert recording.service.player is not None
            button('.recording-page .play-button')
            until('document.querySelector(".recording-page .play-button")?.textContent.trim()==="▶ 播放"')
            result['checks'].append('M16 late real play reply after hide/return is stopped; a fresh explicit audition plays and stops')
            button('.recording-controls .record-button')
            pause(250)
            assert recording.service.recording
            away_and_back('M16')
            assert recording.service.recording
            pause(250)
            label('■ 停止录音')
            until('document.body.innerText.includes("已存入本地工程，尚未导出")')
            assert len(recording.service.project.data['takes']) == 2
            result['checks'].append('M16 synthetic capture continues across actual module hide/return and saves the second take')
            window.view.grab().save(str(out / 'lifecycle-m16.png'))

            phase('lifecycle-m05-start')
            label('唇形提取')
            until('!!document.querySelector(".lip-page")?.offsetParent')
            change('[aria-label="采集模式"]', 'offline')
            encoded = base64.b64encode(video.read_bytes()).decode('ascii')
            js('(()=>{const d=new DataTransfer();d.items.add(new File([Uint8Array.from(atob('+json.dumps(encoded)+'),c=>c.charCodeAt(0))],"neutral-no-face.mkv"));const e=document.querySelector(".lip-page input[type=file]");e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            label('分析所选文件')
            until('document.querySelector(".lip-page")?.getAttribute("aria-busy")==="false"&&document.querySelector(".lip-recording-preview")?.readyState>=2', 180)
            phase('m05-analysis-complete')
            released = calls.count('m05_playback_release')
            js('document.querySelector(".lip-recording-preview").dispatchEvent(new Event("error"))')
            if js('[...document.querySelectorAll(".lip-page button")].some(e=>e.textContent.trim()==="重试音视频回放"&&!e.disabled)'):
                label('重试音视频回放')
            end = time.monotonic() + 90
            while calls.count('m05_playback_release') <= released:
                assert time.monotonic() < end, ('Initial compatible replay did not complete', calls[-15:])
                pause(50)
            until('document.querySelector(".lip-recording-preview")?.readyState>=2&&!document.querySelector(".lip-replay-controls button")?.disabled')
            phase('m05-compatible-replay-established')
            # A controlled native media error asks the actual retry control to
            # prepare the already validated compatible video again.
            js('document.querySelector(".lip-recording-preview").dispatchEvent(new Event("error"))')
            arm('M05')
            button('.lip-replay-controls button')
            wait_reached()
            away_and_back('M05')
            gate.set()
            until('document.querySelector(".lip-recording-preview")?.readyState>=2&&!document.querySelector(".lip-replay-controls button")?.disabled')
            pause(500)
            assert js('document.querySelector(".lip-recording-preview").paused')
            button('.lip-replay-controls button')
            until('!document.querySelector(".lip-recording-preview").paused&&document.querySelector(".lip-recording-preview").currentTime>.35')
            replay = js('(()=>{const v=document.querySelector(".lip-recording-preview");return {time:v.currentTime,width:v.videoWidth,height:v.videoHeight,decodedFrames:v.webkitDecodedFrameCount,audioDecodedBytes:v.webkitAudioDecodedByteCount}})()')
            assert replay['width'] == 320 and replay['height'] == 240 and replay['decodedFrames'] > 1 and replay['audioDecodedBytes'] > 0, replay
            result['m05Replay'] = replay
            button('.lip-replay-controls button')
            assert js('document.querySelector(".lip-recording-preview").paused')
            result['checks'].append('M05 actual no-face analysis and compatible replay; retry preparation after a controlled media error stays paused after hide/return, and a new explicit replay works')
            window.view.grab().save(str(out / 'lifecycle-m05.png'))

            phase('lifecycle-m07-start')
            label('发声类型合成')
            until('!!document.querySelector(".m07-page")?.offsetParent')
            label('打开音频目录')
            until('document.querySelector("select[aria-label=源音频]")?.options.length===3')
            for title, index in (('源音频', 1), ('目标音频', 2)):
                selector = 'select[aria-label="' + title + '"]'
                change(selector, js('document.querySelector('+json.dumps(selector)+').options['+str(index)+'].value'))
            label('提取 F0')
            until('document.querySelector(".m07-page")?.getAttribute("aria-busy")==="false"&&document.body.innerText.includes("分析完成，控制点尚未修改")', 180)
            label('生成当前')
            until('document.querySelector(".m07-page")?.getAttribute("aria-busy")==="false"&&document.querySelectorAll(".result-audios button").length>1&&!document.querySelector(".synthesized-plot .audio-transport button")?.disabled', 180)
            arm('M07')
            button('.result-audios button:first-child')
            wait_reached()
            away_and_back('M07')
            gate.set()
            until('!document.querySelector(".synthesized-plot .audio-transport button")?.disabled')
            pause(300)
            assert js('document.querySelector(".synthesized-plot .audio-transport button").textContent.trim()') == '播放选区'
            selection = js('(()=>{const p=document.querySelector(".synthesized-plot"),s=p.querySelector(".wave-track>svg"),r=s.querySelector(".wave-selection");return {start:Number(s.dataset.start),end:Number(s.dataset.end),x:Number(r.getAttribute("x")),width:Number(r.getAttribute("width")),selected:p.querySelector(".wave-viewport").dataset.selectionActive,duration:Number(p.querySelector(".audio-transport .mono").textContent.split("/")[1].replace("s","").trim())}})()')
            assert selection['start'] == 0 and selection['end'] > 0 and selection['x'] == 0 and selection['width'] == 1000 and selection['selected'] == 'true' and abs(selection['end'] - selection['duration']) <= .000501, selection
            button('.result-audios button:first-child')
            until('document.querySelector(".synthesized-plot .audio-transport button").textContent.trim()==="暂停"')
            button('.synthesized-plot .audio-transport button:nth-child(2)')
            result['checks'].append('M07 actual fixed-fixture analysis/generation/read/decode selects the full late result without playing after hide/return; a new explicit result audition works')
            result['m07Selection'] = selection
            window.view.grab().save(str(out / 'lifecycle-m07.png'))
            phase('playback-lifecycle-complete')
        finally:
            gate.set()
            window.bridge.tasks.invoke = original_task
            recording.dispatch, recording.choose = original_recording, original_choose
            QFileDialog.getExistingDirectory = original_picker
            result['operations'] = calls
            report['playbackLifecycle'] = result
        report['checks'].extend(result['checks'])

    try:
        window.resize(1260, 820)
        window.show()
        window.raise_()
        until('document.querySelector(".m17-body")?.dataset.loaded==="true"&&document.querySelector(".m17-body")?.dataset.fontReady==="true"')
        report['renderer'] = js('(()=>{const c=document.createElement("canvas"),g=c.getContext("webgl"),e=g?.getExtension("WEBGL_debug_renderer_info");return {userAgent:navigator.userAgent,dpr:devicePixelRatio,renderer:e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):null}})()')
        assert not any(flag in (report['chromiumFlags'] or '') for flag in ('--disable-gpu', '--use-gl=swiftshader', '--use-angle=swiftshader'))
        js('window.__uiLong=[];window.__uiObserver=new PerformanceObserver(l=>window.__uiLong.push(...l.getEntries().map(e=>e.duration)));window.__uiObserver.observe({entryTypes:["longtask"]})')
        for theme, button in (('light', '浅色'), ('dark', '深色')):
            phase('ipa-reader-' + theme)
            label('设置')
            label(button)
            click('#tab-M17')
            until('document.documentElement.dataset.theme==='+json.dumps(theme))
            for title, system in (('extIPA', 'extipa'), ('VoQS', 'voqs'), ('IPA', 'ipa')):
                measured(theme+' '+title, lambda title=title, system=system: (label(title), until('!!document.querySelector("[data-chart='+system+']")')))
            label('extIPA')
            report.setdefault('nativeWheels', []).append(dict(theme=theme, **wheel('.m17-chart-viewport')))
            click('.m17-symbol')
            until('document.querySelector(".m17-editor").value.length>0')
            draft = js('document.querySelector(".m17-editor").value')
            window.view.grab().save(str(out / ('extipa-'+theme+'.png')))
            label('使用说明')
            until('!!document.querySelector(".manual-document")')
            click('[data-manual-chapter="m19"]')
            until('document.querySelector(".manual-document")?.dataset.manualChapter==="m19"')
            assert js('document.querySelectorAll(".manual-document .manual-media-image").length') == 8
            js('window.__readerDocument=document.querySelector(".manual-document")')
            click('[data-manual-section="m19-maps"]')
            pause(100)
            assert js('window.__readerDocument===document.querySelector(".manual-document")')
            report['nativeWheels'].append(dict(theme=theme, **wheel('.manual-reading-scroll')))
            for _ in range(6):
                measured(theme+' return IPA', lambda: (click('#tab-M17'), until('!!document.querySelector(".m17-body")?.offsetParent')))
                assert js('document.querySelector(".m17-editor").value') == draft
                measured(theme+' return reader', lambda: (click('#tab-help'), until('document.querySelector(".manual-document")?.dataset.manualChapter==="m19"')))
            pause(350)
            position = js('document.querySelector(".manual-reading-scroll").scrollTop')
            normal = [window.width(), window.height()]
            window.showMaximized()
            deadline = time.monotonic()+12
            while not window.isMaximized() or window._first_maximize.active:
                assert time.monotonic()<deadline, 'Native maximize cover failed to settle'
                pause(80)
            assert FirstMaximizePresentation.valid_picture(window.view.grab().toImage())
            window.view.grab().save(str(out / ('manual-maximized-'+theme+'.png')))
            window.showNormal()
            pause(400)
            assert not window.isMaximized() and all(abs(a-b)<=4 for a,b in zip(normal,[window.width(),window.height()])), (normal, window.size())
            window.showMinimized()
            pause(250)
            assert window.isMinimized()
            window.showNormal()
            window.raise_()
            pause(400)
            assert not window.isMinimized() and window.isVisible()
            assert js('document.querySelector(".manual-document")?.dataset.manualChapter') == 'm19'
            assert js('document.querySelector(".manual-reading-scroll").scrollTop') == position
            assert FirstMaximizePresentation.valid_picture(window.view.grab().toImage())
            window.view.grab().save(str(out / ('manual-restored-'+theme+'.png')))
            click('#tab-M17')
            assert js('document.querySelector(".m17-editor").value') == draft
        assert report['presentation']['scans'] and all(row['directOnly'] for row in report['presentation']['scans'])
        assert max(report['presentation']['attachMs']) < 1000, report['presentation']
        assert not report['rendererTerminations']
        report['rendererLongTasksMs'] = js('window.__uiLong')
        report['manualImageCount'] = 8
        report['checks'] = ['light/dark IPA/extIPA/VoQS controls and neutral symbol input',
            'native Qt wheel scrolls extIPA and actual M19 reader', '24 actual IPA/reader tab returns preserve editor and reader',
            'same-chapter section navigation preserves document', 'native maximize/normal/minimize/restore preserves chapter and scroll',
            'nonblack rendered surfaces and direct-child-only presentation scans under 1000ms', 'default GPU and no renderer termination']
        lifecycle()
        from verify_release_lifecycle_ui import fixtures as release_fixtures, verify_m18
        phase('m18-library-start')
        release_fixtures(scratch, synthesis=False)
        m18_report = dict(checks=[])
        verify_m18(window, out, scratch, m18_report)
        report['m18LibraryLifecycle'] = m18_report
        report['checks'].extend(m18_report['checks'])
        phase('m18-library-complete')
        report['success'] = True
    except Exception:
        report['error'] = traceback.format_exc()
        report['pageAtFailure'] = js('document.body.innerText.slice(-6000)')
        window.view.grab().save(str(out / 'failure.png'))
        raise
    finally:
        interaction_success = report['success']
        report['success'] = False
        phase('owned-window-cleanup-start')
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        (out / 'cleanup.json').write_text(json.dumps(dict(state='retained', scratch=str(scratch),
            reason='Mac owned scratch retained until final verification reports are recovered; no user data deletion'))+'\n', encoding='utf-8')
        window.closing = True
        window.close()
        # Deliberately exercise the real desktop entry's staged close recipe:
        # callbacks must remain safe while page and profile are released.
        window.page.deleteLater()
        pause(100)
        window.profile.deleteLater()
        pause(200)
        window.deleteLater()
        pause(250)
        app.quit()
        report['cleanupComplete'] = True
        report['success'] = interaction_success
        phase('owned-window-cleanup-complete')
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(success=report['success'], cases=len(report['timings']), output=str(out))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
