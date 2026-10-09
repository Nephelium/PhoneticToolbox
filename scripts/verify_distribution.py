"""Frozen distribution checks in a caller-owned new directory; no real devices."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4


def verify(bundle, output):
    bundle, output = Path(bundle), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(success=False, scope='Frozen Windows distribution, isolated state, synthetic science and maximized hidden Qt; no hearing or physical DPI claim', tasks=[], pages=[], checks=[])
    os.environ['PTB_M11_COMPONENT_ROOT'] = str(output / 'components')
    window = app = None
    try:
        from ptb_desktop.bundle_manifest import runtime_bindings
        bindings = runtime_bindings(bundle)
        assert all(Path(value).is_relative_to(bundle) for key, value in bindings.items() if key != 'PTB_M11_COMPONENT_ROOT')
        os.environ.update(bindings)
        report['checks'].append('All pinned scientific runtimes resolve inside the relocated package')
        # A complete model initialization exercises the packaged graph payload.
        environment = {k: v for k, v in os.environ.items() if k not in ('PYTHONHOME', 'PYTHONPATH')}
        environment['PATH'] = os.pathsep.join([os.environ['SystemRoot'] + '/System32', os.environ['SystemRoot']])
        descriptor = json.loads((bundle / 'desktop-bundle.json').read_text('utf8'))
        host_manifest = json.loads((bundle/'host-files.json').read_text('utf8'))
        if host_manifest.get('schema') == 'ptb-host-files/2':
            shared = host_manifest['sharedFiles']
            for name,row in host_manifest['files'].items():
                path=bundle/name
                assert path.stat().st_size==row['size']
                with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256'],name
            linked=sum(os.path.samefile(bundle/name,bundle/source) for name,source in shared.items())
            report['shared_dependencies']=dict(files=len(shared),linked=linked,copied=len(shared)-linked,
                bytes=sum(host_manifest['files'][name]['size'] for name in shared),
                restoration=json.loads((bundle/'.ptb-host-ready.json').read_text('utf8'))['restoration'])
            report['checks'].append('Every restored host file matches original bytes; shared paths checked by file identity')
        report['runtime_paths'] = bindings
        report['bundle'] = str(bundle)
        if descriptor.get('runtimeArchive'):
            report['scientific_probes'] = []
            for name, key in [('egg', 'PTB_EGG_PYTHON'), ('m05', 'PTB_M05_PYTHON')]:
                probe_output = output / ('scientific-' + name)
                probe = subprocess.run([bindings[key], '-I', '-B', str(bundle / 'preview-fixtures/runtime-probe.py'),
                                        name, str(bundle), str(probe_output)], cwd=output, env=environment,
                                       capture_output=True, text=True, timeout=240,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                (output / ('probe-' + name + '.log')).write_text(probe.stdout + '\n' + probe.stderr, 'utf8')
                assert probe.returncode == 0, (name, probe.stderr[-2500:])
                result = json.loads((probe_output / 'result.json').read_text('utf8'))
                report['scientific_probes'].append(dict(runtime=name, result=result))
        model = subprocess.run([bindings['PTB_M05_PYTHON'], '-I', '-B', '-c',
            'import mediapipe as mp; m=mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1); m.close(); print("model initialized")'],
            cwd=output, env=environment, capture_output=True, text=True, timeout=90,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert model.returncode == 0, model.stderr[-2500:]
        report['checks'].append('Relocated MediaPipe FaceMesh graph initialization, no camera or video processing')
        from ptb_worker.local_workspace import prepare_workspace
        from ptb_worker.store import LOCAL_PROJECT
        from ptb_desktop.local_service import LocalService
        import numpy as np
        from scipy.io import wavfile
        database, cache = prepare_workspace(output / 'state', bundle / 'backend/migrations')
        reaper = bundle / 'resources/research/reaper.exe'
        with LocalService(database, local_files_root=cache, reaper_binary=reaper) as service:
            t = np.arange(16000) / 16000
            stream = io.BytesIO(); wavfile.write(stream, 16000, (.25 * np.sin(2 * np.pi * 150 * t)).astype(np.float32))
            ref = service.import_input(stream.getvalue(), 'public-tone.wav', 'audio')
            def task(module, **payload):
                job = service.request('/api/v1/jobs/' + module + '/create', 'POST', dict(project_id=LOCAL_PROJECT, idempotency_key=uuid4().hex, **payload))
                deadline = time.monotonic() + 180
                while True:
                    if module == 'batches':
                        batch = service.get('/api/v1/jobs/batches/' + job['id'])
                        if batch['summary']['closed']:
                            job = service.get('/api/v1/jobs/' + batch['summary']['items'][0]['job_id']); break
                    else:
                        job = service.get('/api/v1/jobs/' + job['id'])
                        if job['state'] not in ('queued', 'running', 'cancel_requested'): break
                    if time.monotonic() > deadline: raise TimeoutError(module)
                    time.sleep(.15)
                assert job['state'] == 'succeeded', (module, job)
                files = {}
                for f in job['result_manifest']['files']:
                    raw = b''.join(service.binary(f'/api/v1/jobs/local-results/{f["id"]}?offset={offset}&size={min(1048576,f["size_bytes"]-offset)}') for offset in range(0, f['size_bytes'], 1048576))
                    assert hashlib.sha256(raw).hexdigest() == f['sha256']
                    files[f['name']] = raw
                report['tasks'].append(dict(module=module, files=list(files), job=job['id']))
                print('Frozen task passed: ' + module, flush=True)
                return job, files
            task('batches', operation='acoustic_analysis', inputs=[dict(audio=ref)], config=dict(selection=dict(keys=['pF0','rF0','pF1']), backend_policy=dict(reaper='native_required')))
            task('m08', audio=ref, config=dict(action='transform',speed=.8,pitch_ratio=1.1))
            job, _ = task('m07', action='analyze',source=ref,target=ref)
            task('m07', action='generate',source=ref,target=ref,analysis_job_id=job['id'],generation=dict(step_count=3))
            from phonetic_core.synthesis.klatt.api import defaults, export_parameters
            parameters = service.import_input(export_parameters(defaults()).encode(), 'klatt.csv', 'table')
            task('m06', action='synthesize', parameters=parameters)
            task('lpc', audio=ref, config=dict(roi_start=.1,roi_end=.15,order=20))
            fixture = np.load(bundle / 'preview-fixtures/EGG-SYN-PCM16.npz')
            stream = io.BytesIO(); wavfile.write(stream,44100,np.column_stack((fixture['load.audio_signal'],fixture['load.egg_signal_raw'])))
            egg = service.import_input(stream.getvalue(), 'public-egg.wav', 'audio')
            task('egg',audio=egg,config=dict(mode='single',roi_start=0,roi_end=.5))
            # Both image reconstruction and original-phase audio drawing exercise
            # the host's SciPy/OpenCV chain, including real managed WAV output.
            import cv2
            image = np.full((64, 96), 255, dtype=np.uint8); image[:, 40:43] = 0
            image_ref = service.import_input(cv2.imencode('.png', image)[1].tobytes(), 'public-spectrogram.png', 'image')
            task('spec2wav', image=image_ref, config=dict(time_end=.1, n_iter=2))
            task('spec2wav', image=ref, config=dict(mode='audio_draw'))
            table = service.import_input((bundle / 'preview-fixtures/public.xlsx').read_bytes(), 'public.xlsx', 'table')
            _, data = task('m14', table=table, config=dict(action='preview', skip_first_row=True, consonant_only_as_zero_initial=True))
            preview = json.loads(next(iter(data.values())))
            task('m14', table=table, config=dict(action='export', skip_first_row=True, consonant_only_as_zero_initial=True,
                 settings=preview['config'], font=dict(schema_version='font/1', zh='Microsoft YaHei', latin='Segoe UI', ipa='Doulos SIL', size_px=14)))
        if descriptor.get('mfa'):
            from ptb_worker.mfa.runtime import load_registry, select
            from ptb_worker.mfa.probe import register
            registry = load_registry()
            chosen = next(row for row in registry['models'] if row.get('dictionary_name') == 'mandarin_pinyin_tab.dict')
            runtime, chosen = select(chosen['validated_runtime'], chosen['id'])
            receipt = register(runtime['path'],chosen['model'],chosen['dictionary'],root=output/'mfa-probe',publish=False)['receipt']
            assert receipt['runtime_fingerprint'] == runtime['fingerprint']
            report['checks'].append('Frozen host uses its relocated real MFA model/dictionary for the public a1 alignment probe')
        else:
            assert not (bundle / 'runtimes/mfa').exists()
            report['checks'].append('MFA environment and models are excluded as requested')
        from PyQt6.QtCore import QEventLoop,QTimer,Qt
        from PyQt6.QtWidgets import QApplication
        from ptb_desktop.host import Workbench,register_scheme
        register_scheme();app=QApplication(['distribution-owned-QA'])
        window=Workbench(bundle/'frontend/dist',test=True,jobs_path=database,local_files_root=cache,reaper_binary=reaper,vocal_resources=bundle/'resources/vocal_tract/native',vocal_profile=output/'vocal')
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);window.showMaximized()
        def pause(ms=100):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            loop=QEventLoop();box=[];window.page.runJavaScript(code,lambda v:(box.append(v),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec();assert box;return box[0]
        def until(code):
            deadline=time.monotonic()+45
            while time.monotonic()<deadline:
                if js(code):return
                pause()
            raise AssertionError(code)
        until('!!document.querySelector(".home-page")')
        project=json.loads((bundle/'frontend/dist/manual/project.json').read_text('utf8'))
        def count_headings(node):
            return int(node.get('type')=='heading' and node.get('attrs',{}).get('level')==2)+sum(count_headings(child) for child in node.get('content',[]))
        for descriptor in project['chapters']:
            mid=descriptor.get('moduleId')
            if not mid:continue
            title=descriptor['title']
            assert js('(()=>{const e=[...document.querySelectorAll(".nav-item")].find(e=>e.title==='+json.dumps(title)+');if(!e)return false;e.click();return true})()'),title
            until('document.querySelector("#tab-'+mid+'")?.getAttribute("aria-selected")==="true"');pause(350)
            assert js('document.querySelectorAll(".module-manual-entry").length') == 0
            if mid == 'M10':
                until('typeof document.querySelector("iframe")?.contentDocument?.getElementById("sharedHelp")?.onclick==="function"')
                assert js('(()=>{const e=document.querySelector("iframe").contentDocument.getElementById("sharedHelp");if(!e||e.disabled)return false;e.click();return true})()')
            elif mid == 'M18':
                # M18 currently uses the shell's contextual manual entry.
                # Keep the same chapter identity and return checks below.
                until('(()=>{const entries=[...document.querySelectorAll("button[title=使用说明]")].filter(e=>e.offsetParent);if(entries.length!==1||entries[0].disabled)return false;entries[0].click();return true})()')
            else:
                until('(()=>{const entries=[...document.querySelectorAll("main button")].filter(e=>e.offsetParent&&e.textContent.trim()==="帮助");if(entries.length!==1||entries[0].disabled)return false;entries[0].click();return true})()')
            until('document.querySelector(".manual-document")?.dataset.manualChapter==='+json.dumps(descriptor['id']))
            assert window.isMaximized()
            assert not js('document.querySelector(".manual-document")?.innerText.includes("素材暂不可用")')
            count=js('document.querySelectorAll(".manual-document h2").length')
            chapter=json.loads((bundle/'frontend/dist/manual'/descriptor['path']).read_text('utf8'))
            assert count==count_headings(chapter['body']),(mid,count)
            # Current P19-R17 exposes exactly one help entry inside each page.
            assert js('(()=>{const e=[...document.querySelectorAll(".manual-reading-toolbar button")].find(e=>e.textContent.trim()==='+json.dumps('返回 ' + title)+');if(!e||e.disabled)return false;e.click();return true})()')
            until('document.querySelector("#tab-'+mid+'")?.getAttribute("aria-selected")==="true"')
            report['pages'].append(dict(module=mid,chapter=descriptor['id'],maximized=True,pageHelp=mid!='M18',helpEntry='sidebar' if mid=='M18' else 'page',returned=True,h2=count))
        report['checks'].append('All current module manual entries resolve to their own in-app chapter and return to the originating module')
        state = window.vocal.invoke('status')
        assert isinstance(state, dict) and not state.get('active'), state
        report['checks'].append('Native vocal tract engine initialized from the bundled model')
        from verify_m16_m17_frozen import verify as recording_and_ipa
        recording_and_ipa(window, output, report)
        from verify_compact_pages import verify as client_workflows
        client_workflows(window, output, report)
        window.view.grab().save(str(output/'manual-frozen-maximized.png'))
        report['success']=True
    except Exception:
        import traceback
        report['error'] = traceback.format_exc()
        raise
    finally:
        if window is not None:
            window.closing=True;window.close()
        if app is not None:app.quit()
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(success=report['success'],report=str(output/'report.json'))),flush=True)
    return 0 if report['success'] else 1
