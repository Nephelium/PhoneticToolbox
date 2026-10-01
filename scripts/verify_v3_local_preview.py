"""Owned package checks; explicit natural-input mode, no device recording."""
import hashlib
import io
import json
import os
import sys
from pathlib import Path
import time
import traceback
from uuid import uuid4


def verify(bundle, out, natural_manifest=None):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu')
    out.mkdir(parents=True, exist_ok=False)
    mode = 'frozen' if getattr(sys, 'frozen', False) else 'source'
    report = dict(success=False, pages=[], tasks=[], scope=f'{mode} Windows local trial, '+('authorized natural audio' if natural_manifest else 'synthetic inputs'))
    window = app = None
    originals = {}
    natural = None
    if natural_manifest:
        manifest=json.loads(Path(natural_manifest).read_text(encoding='utf-8'))
        source_root=Path(manifest['root']).resolve(strict=True)
        natural=manifest['selected']
        for item in natural.values():
            path=Path(item['path']).resolve(strict=True)
            if not path.is_relative_to(source_root):raise ValueError('Natural input outside authorized corpus')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=item['sha256']:raise ValueError('Natural input changed')
            originals[path]=digest
        report['inputs']=[dict(relative=item['relative'],sha256=item['sha256'],duration=item['duration']) for item in natural.values()]
    try:
        from ptb_worker.local_workspace import prepare_workspace
        from ptb_worker.store import LOCAL_PROJECT
        from ptb_desktop.local_service import LocalService
        database, cache = prepare_workspace(out / 'state', bundle / 'backend/migrations')
        reaper = bundle / 'resources/research/reaper.exe'
        if not reaper.exists():
            reaper = bundle / 'phonetic_toolbox/core/acoustic/reaper.exe'
        fixture_root = bundle / 'preview-fixtures'
        # Fail with a precise local traceback if the document dependency or
        # its default template is absent from the frozen package.
        from docx import Document
        Document().save(io.BytesIO())
        with LocalService(database, local_files_root=cache, reaper_binary=reaper) as service:
            report['capabilities'] = service.get('/api/v1/capabilities')
            def run(module, **payload):
                body = dict(project_id=LOCAL_PROJECT, idempotency_key=uuid4().hex, **payload)
                job = service.request('/api/v1/jobs/' + module + '/create', 'POST', body)
                end = time.monotonic() + 180
                if module == 'batches':
                    batch = job
                    while not batch['summary']['closed']:
                        if time.monotonic() > end:
                            raise TimeoutError('acoustic batch')
                        time.sleep(.15)
                        batch = service.get('/api/v1/jobs/batches/' + batch['id'])
                    job = service.get('/api/v1/jobs/' + batch['summary']['items'][0]['job_id'])
                    module = 'm01'
                while job['state'] in ('queued', 'running', 'cancel_requested'):
                    if time.monotonic() > end:
                        raise TimeoutError(module)
                    time.sleep(.15)
                    job = service.get('/api/v1/jobs/' + job['id'])
                if job['state'] != 'succeeded':
                    raise AssertionError((module, job))
                result = {}
                for f in job['result_manifest']['files']:
                    data = b''.join(service.binary(
                        f'/api/v1/jobs/local-results/{f["id"]}?offset={offset}&size={min(1048576, f["size_bytes"]-offset)}')
                        for offset in range(0, f['size_bytes'], 1048576))
                    assert hashlib.sha256(data).hexdigest() == f['sha256']
                    result[f['name']] = data
                report['tasks'].append(dict(module=module, action=payload.get('action', payload.get('config', {}).get('action')), files=list(result)))
                print(f'Verified {mode} task:', module, flush=True)
                return job, result

            import numpy as np
            from scipy.io import wavfile
            if natural:
                audio_path=Path(natural['short']['path'])
                audio_bytes=audio_path.read_bytes()
                audio_name=audio_path.name
            else:
                t = np.arange(16000) / 16000
                audio = (.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
                stream = io.BytesIO(); wavfile.write(stream, 16000, audio)
                audio_bytes,audio_name=stream.getvalue(),'public-tone.wav'
            ref = service.import_input(audio_bytes, audio_name, 'audio')
            _, data = run('batches', operation='acoustic_analysis', inputs=[dict(audio=ref)],
                config=dict(selection=dict(keys=['pF0','rF0','pF1']), backend_policy=dict(reaper='native_required')))
            acoustic = json.loads(data['result.ptb.json'])
            assert acoustic['metadata']['computation_revision'] == 'acoustic/2'
            assert any(item['actual'] == 'native_reaper' for item in acoustic['metadata']['backends'])
            report['acoustic_revision'] = acoustic['metadata']['computation_revision']
            run('m08', audio=ref, config=dict(action='transform', speed=.8, pitch_ratio=1.1))
            job, data = run('m07', action='analyze', source=ref, target=ref)
            run('m07', action='generate', source=ref, target=ref, analysis_job_id=job['id'], generation=dict(step_count=3))

            from phonetic_core.synthesis.klatt.api import defaults, export_parameters
            config = defaults(); config.update(sequence='a i', duration=.6)
            for curve in config['curves'].values(): curve['points'][-1][0] = .6
            _, data = run('m06', action='generate', parameters=service.import_input(export_parameters(config).encode(), 'parameters.csv', 'table'))
            config = json.loads(data['m06.ptb.json'])['config']
            run('m06', action='synthesize', parameters=service.import_input(export_parameters(config).encode(), 'parameters.csv', 'table'))

            lpc_job, lpc_data = run('lpc', audio=ref, config=dict(roi_start=.1, roi_end=.15, order=20))
            if natural:
                egg_path=Path(natural['egg']['path'])
                egg_bytes,egg_name=egg_path.read_bytes(),egg_path.name
            else:
                frozen = np.load(fixture_root / 'EGG-SYN-PCM16.npz' if fixture_root.exists() else bundle / 'tests/fixtures/m03/EGG-SYN-PCM16.npz')
                stereo = np.column_stack((frozen['load.audio_signal'], frozen['load.egg_signal_raw']))
                stream = io.BytesIO(); wavfile.write(stream, 44100, stereo)
                egg_bytes,egg_name=stream.getvalue(),'public-egg.wav'
            egg = service.import_input(egg_bytes, egg_name, 'audio')
            run('egg', audio=egg, config=dict(mode='single', roi_start=4 if natural else 0, roi_end=4.5 if natural else .5))
            if natural:
                preview_timings=[]
                for index in range(5):
                    t=time.monotonic()
                    value=service.preview(audio_bytes,dict(channel=0,start=.05,end=.2,width=800))
                    preview_timings.append(time.monotonic()-t)
                    assert value['backend']=='praat' and value['sha256']==hashlib.sha256(audio_bytes).hexdigest()
                report['frozen_spectrogram_seconds']=preview_timings

            table_path = fixture_root / 'public.xlsx' if fixture_root.exists() else bundle / 'tests/fixtures/m14/public.xlsx'
            table = service.import_input(table_path.read_bytes(), 'public.xlsx', 'table')
            _, data = run('m14', table=table, config=dict(action='preview', skip_first_row=True, consonant_only_as_zero_initial=True))
            preview = json.loads(next(iter(data.values())))
            run('m14', table=table, config=dict(action='export', skip_first_row=True, consonant_only_as_zero_initial=True,
                settings=preview['config'], font=dict(schema_version='font/1', zh='Microsoft YaHei', latin='Segoe UI', ipa='Doulos SIL', size_px=14)))
            from ptb_worker.mfa.runtime import load_registry
            registry = load_registry()
            report['mfa_registered'] = bool(registry.get('runtimes') and registry.get('models'))
            assert report['mfa_registered'], 'Expected existing local optional MFA registry'

        from PyQt6.QtCore import QEventLoop, QTimer
        from PyQt6.QtWidgets import QApplication
        from ptb_desktop.host import register_scheme, Workbench
        register_scheme(); app = QApplication(['v3-local-preview-check'])
        window = Workbench(bundle / 'frontend/dist', test=True, jobs_path=database, local_files_root=cache,
                           reaper_binary=reaper, vocal_resources=bundle / 'resources/vocal_tract/native', vocal_profile=out / 'vocal-profile')
        window.resize(1920 if natural else 1440, 1000); window.show()
        def pause():
            loop = QEventLoop(); QTimer.singleShot(100, loop.quit); loop.exec()
        def js(code):
            loop = QEventLoop(); values = []
            window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
            QTimer.singleShot(5000, loop.quit); loop.exec()
            return values[0] if values else None
        def wait(code):
            end = time.monotonic() + 35
            while time.monotonic() < end:
                if js(code): return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-2500)')))
        wait('!!document.querySelector("nav")')
        pages = [('参数估计', '.m01-page'), ('参数显示', '.m02-page'), ('EGG 信号分析', '.egg-page'),
                 ('LPC 谱图', '.lpc-page'), ('唇形提取', '.lip-page'), ('语音合成', '[aria-label="语音合成工作区"]'),
                 ('发声类型合成', '[aria-label="发声类型连续统工作区"]'), ('变速变调', '.m08-page'),
                 ('语谱图转音频', '.m09-page'), ('声道工作台', 'iframe[title="声道工作台"]'),
                 ('MFA 自动标注', '[aria-label="MFA 自动标注工作区"]'), ('语音标注对齐', '.annotation-page'),
                 ('普通话转 IPA', '.mandarin-ipa-page'), ('音系归纳', '[aria-label="音系归纳工作区"]'),
                 ('感知实验', '.perception-page')]
        for i, (title, selector) in enumerate(pages, 1):
            if natural and i==10:continue
            js('[...document.querySelectorAll("nav button")].find(b=>b.textContent.includes(' + json.dumps(title) + '))?.click()')
            wait('!!document.querySelector(' + json.dumps(selector) + ')')
            if i == 12:
                # Exercise the class update that displaced the blue resize line.
                report['annotation_boundaries'] = []
                for toggle in range(3):
                    if toggle:
                        js('[...document.querySelectorAll(".annotation-plots button")].find(b=>b.textContent.trim()==="文件列表")?.click()')
                    wait('getComputedStyle(document.querySelector(".annotation-layout")).position==="relative"')
                    pause()
                    geometry = js('''(()=>{const r=document.querySelector('.annotation-layout'),b=r.getBoundingClientRect();
                        return [...r.querySelectorAll('.panel-resize-handle:not([hidden])')].map(h=>{
                            const q=h.getBoundingClientRect();return {label:h.getAttribute('aria-label'),
                                inside:q.top>=b.top-1&&q.left>=b.left-1&&q.right<=b.right+1};});})()''')
                    assert geometry and all(item['inside'] for item in geometry), geometry
                    report['annotation_boundaries'].append(geometry)
            # DOM readiness precedes Qt's compositor. Wait for two renderer frames
            # and color transitions before reading actual pixels.
            js('window.__p17paint=false;requestAnimationFrame(()=>requestAnimationFrame(()=>window.__p17paint=true))')
            wait('window.__p17paint===true')
            for _ in range(4):pause()
            if natural:
                assert not js('document.querySelector("main").innerText.includes("目录授权已失效")'), (title, 'unselected_directory_reported_expired')
            window.view.grab().save(str(out / f'M{i:02d}.png'))
            geometry=js('''(()=>{const m=document.querySelector('main'),s=document.querySelector('main>.module-frame:not([style*="display: none"])');return {innerWidth,innerHeight,dpr:devicePixelRatio,mainHeight:m.clientHeight,mainScroll:m.scrollHeight,moduleHeight:s?.clientHeight,moduleScroll:s?.scrollHeight};})()''')
            report['pages'].append(dict(title=title,geometry=geometry) if natural else title)
            if natural:assert geometry['mainScroll']<=geometry['mainHeight']+2, (title,geometry)
            print(f'Verified {mode} page:', i, flush=True)
        if natural:
            # Exercise the delivered Qt PNG action, not just the worker's PNG bytes.
            from PyQt6.QtWidgets import QFileDialog
            from PyQt6.QtGui import QImage
            png_path = out / 'frozen-lpc-direct.png'
            dialogs, download_states = [], []
            choices = ['', str(png_path)]
            original_dialog = QFileDialog.getSaveFileName
            def png_dialog(*args, **kwargs):
                dialogs.append(str(args[2]))
                return (choices.pop(0), '')
            def observe_download(download):
                download.stateChanged.connect(lambda state: download_states.append(download.state().name))
            def native_wait(condition):
                end = time.monotonic() + 35
                while time.monotonic() < end:
                    if condition(): return
                    pause()
                raise AssertionError('Frozen LPC PNG condition timed out')
            def click_button(text):
                target = '[...document.querySelectorAll(".lpc-page button")].find(b=>b.offsetParent&&!b.disabled&&b.textContent.trim()===' + json.dumps(text) + ')'
                wait('!!' + target)
                js(target + '.click()')
            QFileDialog.getSaveFileName = png_dialog
            window.profile.downloadRequested.connect(observe_download)
            try:
                js('[...document.querySelectorAll("nav button")].find(b=>b.textContent.includes("LPC 谱图"))?.click()')
                wait('!![...document.querySelectorAll(".lpc-page .history-links button")].find(b=>b.textContent.includes(' + json.dumps(lpc_job['id'][:8]) + '))')
                js('[...document.querySelectorAll(".lpc-page .history-links button")].find(b=>b.textContent.includes(' + json.dumps(lpc_job['id'][:8]) + ')).click()')
                wait('!!document.querySelector(".lpc-spectrum svg")')
                click_button('保存 PNG 图片')
                native_wait(lambda: len(dialogs) == 1)
                for _ in range(2): pause()
                assert not png_path.exists(), 'Cancelled PNG dialog must not write a file'
                click_button('保存 PNG 图片')
                native_wait(lambda: png_path.exists() and 'DownloadCompleted' in download_states)
                expected = next(value for name, value in lpc_data.items() if name.endswith('.png'))
                assert png_path.read_bytes() == expected, 'Frozen direct PNG differs from managed result'
                image = QImage(str(png_path))
                assert (image.width(), image.height()) == (2400, 1350)
                report['lpc_png'] = dict(dialogs=len(dialogs), cancelled_without_write=True,
                    download_states=download_states, sha256=hashlib.sha256(expected).hexdigest(),
                    bytes=len(expected), width=image.width(), height=image.height(),
                    dpi=image.dotsPerMeterX()*.0254, exact_managed_bytes=True)
                window.view.grab().save(str(out / 'M04-result.png'))
                print(f'Verified {mode} LPC PNG cancel and actual file save', flush=True)
            finally:
                QFileDialog.getSaveFileName = original_dialog
            report['originals_unchanged']=all(hashlib.sha256(path.read_bytes()).hexdigest()==digest for path,digest in originals.items())
            assert report['originals_unchanged']
        report['success'] = True
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
    finally:
        if window is not None:
            window.closing = True; window.close()
            window.page.deleteLater(); app.processEvents(); app.quit()
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if report['success'] else 1
