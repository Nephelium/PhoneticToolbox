"""Owned frozen-package checks; synthetic inputs, no camera/microphone capture."""
import hashlib
import io
import json
import os
import sys
from pathlib import Path
import time
import traceback
from uuid import uuid4


def verify(bundle, out):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu')
    out.mkdir(parents=True, exist_ok=False)
    mode = 'frozen' if getattr(sys, 'frozen', False) else 'source'
    report = dict(success=False, pages=[], tasks=[], scope=f'{mode} Windows local trial, synthetic inputs')
    window = app = None
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
            t = np.arange(16000) / 16000
            audio = (.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
            stream = io.BytesIO(); wavfile.write(stream, 16000, audio)
            ref = service.import_input(stream.getvalue(), 'public-tone.wav', 'audio')
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

            run('lpc', audio=ref, config=dict(roi_start=.1, roi_end=.15, order=20))
            frozen = np.load(fixture_root / 'EGG-SYN-PCM16.npz' if fixture_root.exists() else bundle / 'tests/fixtures/m03/EGG-SYN-PCM16.npz')
            stereo = np.column_stack((frozen['load.audio_signal'], frozen['load.egg_signal_raw']))
            stream = io.BytesIO(); wavfile.write(stream, 44100, stereo)
            egg = service.import_input(stream.getvalue(), 'public-egg.wav', 'audio')
            run('egg', audio=egg, config=dict(mode='single', roi_start=0, roi_end=.5))

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
        window.resize(1440, 1000); window.show()
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
            pause(); window.view.grab().save(str(out / f'M{i:02d}.png'))
            report['pages'].append(title)
            print(f'Verified {mode} page:', i, flush=True)
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
