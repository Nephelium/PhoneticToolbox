"""P11-PERF synthetic, loopback-only benchmark against existing Linux runtime.

Copies an explicitly supplied quiescent SQLite template. Never runs DDL, installs
packages, changes system settings, contacts public services or deletes evidence.
Two owned API processes share the same files, DB and production admission lane.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from uuid import uuid4


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def percentile(values, percent):
    """Nearest rank, including small samples without invented interpolation."""
    return sorted(values)[max(0, math.ceil(len(values)*percent/100)-1)] if values else None


def distribution(values):
    return dict(n=len(values), p50=percentile(values, 50), p95=percentile(values, 95),
                p99=percentile(values, 99), maximum=max(values) if values else None)


def system_sample(root, pids):
    memory = {k:int(v.split()[0])*1024 for k,v in
              (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
              if k in ('MemTotal','MemAvailable','SwapTotal','SwapFree')}
    psi = {}
    for resource in ('memory','cpu','io'):
        psi[resource] = Path('/proc/pressure', resource).read_text().strip()
    processes = {}
    for pid in pids:
        try:
            values = {}
            for line in Path('/proc', str(pid), 'smaps_rollup').read_text().splitlines():
                if line.startswith(('Pss:', 'Private_Clean:', 'Private_Dirty:', 'Rss:')):
                    key, value = line.split(':', 1); values[key] = int(value.split()[0])*1024
            processes[str(pid)] = values
        except FileNotFoundError:
            processes[str(pid)] = {'gone': True}
    # Sample groups from the exact active lane journal, never enumerate/kill
    # unrelated services. Completed groups also produce collector evidence.
    from ptb_worker.native.admission import default_root
    groups = []
    unknown_owner = False
    state_path = default_root()/'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('active'):
            unknown_owner = state['active']['pid'] not in pids+[os.getpid()]
            from ptb_worker.native.posix import _sample
            observed = {}; _sample(state['active']['unit'], observed)
            group = observed.get('cgroup_path')
            if group:
                for name in ('memory.current','memory.peak','memory.events','cpu.stat','memory.pressure','cgroup.procs'):
                    try: observed[name] = (Path(group)/name).read_text().strip()
                    except FileNotFoundError: pass
            groups.append(observed)
    used = 0
    for path in root.rglob('*'):
        try:
            if path.is_file(): used += path.stat().st_size
        except FileNotFoundError:
            pass  # Owned temporary files can be retired during this sample.
    return dict(time=time.time(), memory=memory, pressure=psi, processes=processes,
                groups=groups, unknown_owner=unknown_owner,
                disk_free_bytes=shutil.disk_usage(root).free, evidence_bytes=used)


class StopPolicy:
    def __init__(self):
        self.pressure_since = None

    def reason(self, sample, now):
        if sample.get('unknown_owner'):
            return 'resource_ownership_unknown'
        if sample['disk_free_bytes'] < 10*1024**3:
            return 'disk_free_below_10_GiB'
        full = next(line for line in sample['pressure']['memory'].splitlines() if line.startswith('full '))
        average = float(dict(item.split('=') for item in full.split()[1:])['avg10'])
        pressure = sample['memory']['MemAvailable'] < 700*1024**2 or average > 1
        self.pressure_since = (self.pressure_since if self.pressure_since is not None else now) if pressure else None
        if self.pressure_since is not None and now-self.pressure_since >= 10:
            return 'sustained_memory_pressure'
        if any(p.get('gone') for p in sample['processes'].values()):
            return 'owned_api_exited'
        for group in sample['groups']:
            events = dict(line.split() for line in group.get('memory.events','').splitlines())
            if any(int(events.get(key,0)) for key in ('oom','oom_kill')):
                return 'owned_group_oom'
        return None


def serve(root, index):
    import uvicorn
    from ptb_api.main import create_app
    from ptb_worker.store import SQLiteJobStore
    from ptb_worker.local_acoustic_files import LocalAcousticFiles
    from ptb_worker.acoustic_batches import AcousticBatches
    from ptb_worker.acoustic_executor import execute_acoustic_claim
    from ptb_worker.native.linux_runtime import load_profile
    from ptb_worker.native import posix
    _, profile = load_profile()
    original_collector=posix.run_bounded
    def observed_collector(*args,**kwargs):
        evidence=kwargs.setdefault('evidence',{})
        if evidence is None: evidence={}; kwargs['evidence']=evidence
        started=time.monotonic()
        try: return original_collector(*args,**kwargs)
        finally:
            write_json(root/'collectors'/(uuid4().hex+'.json'),dict(api=index,
                wall_seconds=time.monotonic()-started,process=evidence))
    posix.run_bounded=observed_collector  # This benchmark's own API process only.
    store = SQLiteJobStore(root/'jobs.sqlite3', max_running=1)
    files = LocalAcousticFiles(store, root/'assets', reaper_binary=profile['reaper_binary'])
    AcousticBatches(store, files)
    token = os.environ['PTB_PERF_LOCAL_TOKEN']
    stop = threading.Event()
    app = create_app('local', job_store=store, local_token=token, local_origin='http://127.0.0.1')

    @app.middleware('http')
    async def elapsed(request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers['X-PTB-Server-Seconds'] = str(time.perf_counter()-start)
        return response

    def worker():
        try:
            while not stop.is_set():
                claim = store.claim('p11-perf-'+str(index))
                if claim is None: stop.wait(.1); continue
                evidence = {}; started = time.monotonic()
                execute_acoustic_claim(store, claim, 'p11-perf-'+str(index), stop, process_evidence=evidence)
                write_json(root/'processes'/(claim['id']+'.json'), dict(
                    job_id=claim['id'], elapsed_seconds=time.monotonic()-started, process=evidence))
        except Exception as exc:
            write_json(root/f'worker-error-{index}.json', dict(type=type(exc).__name__))
            stop.set()

    sock = socket.socket(); sock.bind(('127.0.0.1', 0))
    write_json(root/f'api-{index}.json', dict(port=sock.getsockname()[1], pid=os.getpid()))
    thread = threading.Thread(target=worker, daemon=True); thread.start()
    server = uvicorn.Server(uvicorn.Config(app, log_level='error', timeout_graceful_shutdown=10))
    try:
        server.run(sockets=[sock])
    finally:
        stop.set(); thread.join(timeout=20)


class Client:
    def __init__(self, url, token, samples, lock):
        import httpx
        self.http = httpx.Client(base_url=url, headers={'Authorization':'Bearer '+token, 'Origin':'http://127.0.0.1'},
                                 trust_env=False, timeout=330)
        self.samples, self.lock = samples, lock
        self.stage = 'short'

    def request(self, method, route, *, category, **kwargs):
        start = time.perf_counter()
        response = self.http.request(method, route, **kwargs)
        elapsed = time.perf_counter()-start
        raw = response.headers.get('X-PTB-Server-Seconds')
        record = dict(time=time.time(), stage=self.stage, category=category, status=response.status_code,
                      loopback_seconds=elapsed, server_seconds=float(raw) if raw is not None else None,
                      response_bytes=len(response.content))
        with self.lock:
            self.samples.append(record)
        return response


def benchmark(args):
    if sys.platform != 'linux' or os.environ.get('PTB_RESOURCE_PROFILE','server-small') != 'server-small':
        raise ValueError('This benchmark requires Linux server-small')
    if not args.authorized_test_directory:
        raise ValueError('Explicit --authorized-test-directory is required')
    from ptb_worker.native.linux_runtime import load_profile
    from ptb_worker.native.posix import _show
    from ptb_worker.store import LOCAL_PROJECT
    from ptb_worker.local_acoustic_files import initialize_local_files
    profile_path, profile = load_profile()
    if _show('ptb-p11-capability-probe.service').get('LoadState') != 'not-found':
        raise ValueError('No usable user systemd boundary; benchmark not started')
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    processes = []; clients = []; handles = []; samples = []; records = []; lock = threading.Lock()
    stop = threading.Event(); monitor = None; stop_reasons = []; failures = []
    token = uuid4().hex  # Inherited environment only, never written to evidence.
    report = dict(success=False, transport='loopback HTTP; local-token; existing SQLite copy; 10 synthetic clients',
                  schema_applied=[], profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                  versions=profile['versions'], threads=dict(OMP=1, OPENBLAS=1, MKL=1),
                  public_rtt_seconds=None, pure_compute_seconds=None,
                  limitations=['No PostgreSQL/accounts/browser/public-network validation',
                               'Repeated tasks start fresh child interpreters; warm refers to OS/font caches',
                               'M08/M09/trusted-worker capabilities are not bypassed'])
    report['benchmark_sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (Path(__file__),Path(__file__).with_name('p11_compute_probe.py'))}
    report['runtime_font_hashes']={Path(p).name:sha for p,sha in profile['hashes'].items()
                                  if Path(p).suffix.lower() in ('.ttf','.otf','.ttc')}
    write_json(root/'run-start.json', report)
    try:
        with sqlite3.connect(args.template.resolve().as_uri()+'?mode=ro', uri=True) as src:
            assert not src.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
            with sqlite3.connect(root/'jobs.sqlite3') as dst: src.backup(dst)
        (root/'assets').mkdir(); initialize_local_files(root/'assets'); (root/'processes').mkdir(); (root/'collectors').mkdir()
        import numpy as np
        from scipy.io import wavfile
        fixture = Path(__file__).resolve().parents[1]/'tests/fixtures/m03/EGG-SYN-PCM16.npz'
        with np.load(fixture) as data:
            values = np.column_stack([data['load.egg_signal_raw'],data['load.audio_signal']])
        source = {}
        for name, data in [('stereo',values),('mono',values[:,1])]:
            stream = io.BytesIO(); wavfile.write(stream,44100,data); source[name] = stream.getvalue()
        report['inputs'] = {k:dict(sha256=hashlib.sha256(v).hexdigest(),bytes=len(v),sample_rate=44100,
                                  samples=len(values)) for k,v in source.items()}
        report['fonts'] = font = dict(zh='Noto Sans SC',latin='DejaVu Sans')
        report['parameters']=dict(lpc=dict(roi_end=.05,font=font),
                                  egg=dict(mode='single',roi_end=.5,font=font),acoustic='AcousticConfigSnapshot defaults')
        report['api_fresh_process_startup_seconds']=[]
        for index in range(2):
            startup_started=time.monotonic()
            log = (root/f'api-{index}.log').open('xb'); handles.append(log)
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--serve', str(index),
                                        '--output', str(root)], stdout=log, stderr=log,
                                       env={**os.environ,'PTB_PERF_LOCAL_TOKEN':token})
            processes.append(process)
            deadline = time.monotonic()+30
            while not (root/f'api-{index}.json').exists():
                assert process.poll() is None and time.monotonic()<deadline, 'API startup failed'
                time.sleep(.05)
            port = json.loads((root/f'api-{index}.json').read_text())['port']
            client = Client(f'http://127.0.0.1:{port}', token, samples, lock); clients.append(client)
            while True:
                try:
                    assert client.request('GET','/api/v1/health',category='startup').status_code == 200
                    break
                except Exception:
                    assert process.poll() is None and time.monotonic()<deadline
                    time.sleep(.1)
            report['api_fresh_process_startup_seconds'].append(time.monotonic()-startup_started)
        def watch():
            policy = StopPolicy()
            try:
                with (root/'system.jsonl').open('x', encoding='utf-8') as log:
                    while not stop.is_set():
                        sample = system_sample(root, [p.pid for p in processes])
                        log.write(json.dumps(sample)+'\n'); log.flush()
                        reason = policy.reason(sample, time.monotonic())
                        if reason: stop_reasons.append(reason); stop.set(); return
                        if list(root.glob('worker-error-*.json')):
                            stop_reasons.append('worker_exception'); stop.set(); return
                        stop.wait(1)
            except Exception as exc:
                stop_reasons.append('monitor_'+type(exc).__name__); stop.set()
        monitor = threading.Thread(target=watch, daemon=True); monitor.start()
        refs = {}
        preview_tables = {}
        for name, raw in source.items():
            r = clients[0].request('POST','/api/v1/jobs/local-inputs',category='upload',
                                   params=dict(role='audio',name='public-'+name+'.wav'),content=raw)
            assert r.status_code == 200, r.text
            refs[name] = r.json()

        def submit(phase, client):
            body = dict(project_id=LOCAL_PROJECT, idempotency_key=uuid4().hex)
            if phase == 'acoustic':
                body.update(operation='acoustic_analysis',inputs=[{'audio':refs['mono']}],config={})
                r = client.request('POST','/api/v1/jobs/batches/create',category='submit_acoustic',json=body)
                assert r.status_code == 201, r.text
                batch = r.json()
                deadline = time.monotonic()+30
                while True:
                    # Resolve the real durable child, without bypassing submit/capability.
                    with sqlite3.connect((root/'jobs.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
                        row = db.execute('SELECT child_job_id FROM acoustic_batch_items WHERE batch_id=? AND child_job_id IS NOT NULL', (batch['id'],)).fetchone()
                    if row: return row[0]
                    assert time.monotonic()<deadline and not stop.is_set()
                    time.sleep(.1)
            body.update(audio=refs['stereo'],config=dict(roi_end=.05,font=font) if phase=='lpc' else dict(mode='single',roi_end=.5,font=font))
            r = client.request('POST',f'/api/v1/jobs/{phase}/create',category='submit_'+phase,json=body)
            assert r.status_code == 201, r.text
            return r.json()['id']

        def await_job(job_id, client, *, cancelled=False):
            deadline = time.monotonic()+600
            while True:
                r = client.request('GET','/api/v1/jobs/'+job_id,category='job_poll'); assert r.status_code==200
                done = r.json()
                if done['state'] in ('succeeded','failed','cancelled','interrupted'): break
                assert not stop.is_set() and time.monotonic()<deadline, 'job wait stopped'
                time.sleep(.2)
            if cancelled:
                assert done['state']=='cancelled', done
            else:
                assert done['state']=='succeeded', done
                for item in done['result_manifest']['files']:
                    checksum = hashlib.sha256()
                    table_bytes=bytearray() if item['name']=='result.ptb.sqlite' and 'raw' not in preview_tables else None
                    for offset in range(0,item['size_bytes'],1048576):
                        r = client.request('GET','/api/v1/jobs/local-results/'+item['id'],category='download',
                                           params=dict(offset=offset,size=min(1048576,item['size_bytes']-offset)))
                        assert r.status_code==200; checksum.update(r.content)
                        if table_bytes is not None: table_bytes.extend(r.content)
                    assert checksum.hexdigest()==item['sha256']
                    if table_bytes is not None: preview_tables['raw']=bytes(table_bytes)
            return done

        def task(phase, client, label):
            start = time.monotonic(); job_id = submit(phase,client); done=await_job(job_id,client)
            records.append(dict(phase=phase, label=label, id=job_id,state=done['state'],
                                end_to_end_seconds=time.monotonic()-start))

        def preview(client):
            r = client.request('POST','/api/v1/preview/spectrogram',category='spectrogram',
                               params=dict(channel=0,start=0.,end=.5,width=200),content=source['stereo'],
                               headers={'Content-Type':'application/octet-stream'})
            assert r.status_code in (200,429), r.text
            if r.status_code==200: assert r.json()['sha256']==report['inputs']['stereo']['sha256']

        def parameter_preview(client):
            raw=preview_tables['raw']
            r=client.request('POST','/api/v1/preview/parameters',category='parameter_preview',
                             params=dict(name='public.ptb.sqlite'),content=raw)
            assert r.status_code in (200,429),r.text
            if r.status_code==200: assert r.json()['sha256']==hashlib.sha256(raw).hexdigest()

        # Graded short gate. No mixed run starts after a failed short gate.
        for repetition in range(6):
            assert not stop.is_set()
            for phase in ('lpc','egg','acoustic'):
                task(phase, clients[0], 'first-observed' if repetition==0 else 'cache-warm-'+str(repetition))
            preview(clients[1])
            parameter_preview(clients[1])
            r=clients[1].request('POST','/api/v1/jobs/lpc/fonts',category='font_preflight',json=font)
            assert r.status_code==200 and r.json()['available'], r.text
            report['font_preflight_evidence']=r.json()
        report['short_gate_passed'] = True
        write_json(root/'short-gate.json', dict(records=records, profile_sha256=report['profile_sha256']))
        # Separate diagnostic samples: exact core calls, actual export calls,
        # startup/end-to-end overhead excluded from pure-compute measurements.
        from ptb_worker.native.posix import run_bounded
        from ptb_worker.io.limits import Limits
        compute_samples=[]
        for repetition in range(6):
            for phase in ('lpc','egg','acoustic'):
                directory=root/f'compute-{phase}-{repetition}'; directory.mkdir()
                evidence={}
                try:
                    raw=run_bounded([profile['python'],'-I','-B',str(Path(__file__).with_name('p11_compute_probe.py')),
                                     str(profile_path),phase,str(fixture),str(directory)],b'',directory,
                                    Limits(input_bytes=1,output_bytes=131072,process_bytes=1_073_741_824,timeout_seconds=240),
                                    stop=stop.is_set,evidence=evidence)
                    value=json.loads(raw)
                    assert value['input_sha256']==report['inputs']['mono' if phase=='acoustic' else 'stereo']['sha256']
                    compute_samples.append(dict(repetition=repetition,**value))
                finally: write_json(directory/'process.json',evidence)
        write_json(root/'compute-samples.json',compute_samples)
        report['pure_compute_seconds']={phase:distribution([x['pure_compute_seconds'] for x in compute_samples if x['phase']==phase])
                                        for phase in ('lpc','egg','acoustic')}
        report['export_seconds']={phase:distribution([x['export_seconds'] for x in compute_samples if x['phase']==phase])
                                 for phase in ('lpc','egg','acoustic')}

        if args.mixed_seconds:
            for client in clients: client.stage='mixed'
            mixed_start=time.monotonic(); deadline=mixed_start+args.mixed_seconds
            def light(index):
                c=Client(str(clients[index%2].http.base_url),token,samples,lock); c.stage='mixed'
                try:
                    iteration = 0
                    while time.monotonic()<deadline and not stop.is_set():
                        c.stage='mixed-light' if time.monotonic()-mixed_start<60 else 'mixed-compute'
                        r=c.request('GET','/api/v1/jobs',category='light_jobs',params=dict(project_id=LOCAL_PROJECT))
                        assert r.status_code==200
                        r=c.request('GET','/api/v1/health',category='light_health'); assert r.status_code==200
                        if index==0 and iteration%20==19:
                            preview(c)  # Competes with the other API's scientific worker.
                        if index==1 and iteration%40==39:
                            parameter_preview(c)
                        iteration += 1
                        stop.wait(.5)
                except Exception as exc: failures.append('light_'+type(exc).__name__); stop.set()
                finally: c.http.close()
            def compute():
                index=0
                try:
                    while time.monotonic()<deadline and not stop.is_set():
                        for client in clients:
                            client.stage='mixed-light' if time.monotonic()-mixed_start<60 else 'mixed-compute'
                        # First minute: 10 light interactions. Then one lane + 9
                        # light clients; client ten submits scientific/preview work.
                        if time.monotonic()-mixed_start<min(60,args.mixed_seconds/3):
                            r=clients[0].request('GET','/api/v1/health',category='light_health')
                            assert r.status_code==200
                            stop.wait(.5); continue
                        if index%5==4:
                            jobs=[submit('lpc',clients[0]) for _ in range(4)]
                            for job_id in jobs[1:]:
                                r=clients[0].request('POST','/api/v1/jobs/'+job_id+'/cancel',category='cancel')
                                assert r.status_code==200
                            await_job(jobs[0],clients[0])
                            for job_id in jobs[1:]: await_job(job_id,clients[0],cancelled=True)
                            running=submit('egg',clients[0])
                            started_deadline=time.monotonic()+60
                            while True:
                                r=clients[0].request('GET','/api/v1/jobs/'+running,category='job_poll')
                                if r.json()['state']=='running': break
                                assert time.monotonic()<started_deadline and not stop.is_set()
                                time.sleep(.05)
                            r=clients[0].request('POST','/api/v1/jobs/'+running+'/cancel',category='cancel_running')
                            assert r.status_code==200
                            await_job(running,clients[0],cancelled=True)
                            task('lpc',clients[0],'cancel-recovery')
                        else: task(('lpc','egg','acoustic')[index%3],clients[0],'mixed')
                        preview(clients[1]); index+=1
                except Exception as exc: failures.append('compute_'+type(exc).__name__); stop.set()
            with ThreadPoolExecutor(max_workers=10) as pool:
                futures=[pool.submit(light,i) for i in range(9)]+[pool.submit(compute)]
                for future in futures: future.result()
            report['mixed_elapsed_seconds']=time.monotonic()-mixed_start
        report['success']=not failures and not stop_reasons
    except Exception as exc:
        failures.append(type(exc).__name__+': '+str(exc)[:1000])
    finally:
        stop.set()
        if monitor: monitor.join(timeout=10)
        for client in clients: client.http.close()
        for process in processes:
            if process.poll() is None: process.send_signal(signal.SIGTERM)
        for process in processes:
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Exact owned PIDs only. Admission retains ownership of any
                # lingering unit and blocks future grants until clean recovery.
                process.kill(); process.wait(timeout=5)
        for handle in handles: handle.close()
        # If forced shutdown left an owned transient group, stop precisely that
        # journal record. Never stop another chat's unit or all ptb-* processes.
        from ptb_worker.native.admission import default_root
        from ptb_worker.native.posix import recover_abandoned_unit
        state_path=default_root()/'state.json'
        if state_path.exists():
            active=json.loads(state_path.read_text()).get('active')
            if active and active['pid'] in [p.pid for p in processes]:
                try: recover_abandoned_unit(active['unit'])
                except Exception as exc: failures.append('cleanup_'+type(exc).__name__)
        write_json(root/'http-samples.json',samples); write_json(root/'jobs.json',records)
        report.update(failures=failures,stop_reasons=stop_reasons,measurements={})
        for category in sorted({s['category'] for s in samples}):
            rows=[s for s in samples if s['category']==category]
            errors=sum(r['status']>=400 and r['status']!=429 for r in rows)
            report['measurements'][category]=dict(samples=len(rows),unexpected_http_failures=errors,
                failure_rate=errors/len(rows),expected_busy=sum(r['status']==429 for r in rows),
                server_seconds=distribution([r['server_seconds'] for r in rows if r['server_seconds'] is not None]),
                loopback_seconds=distribution([r['loopback_seconds'] for r in rows]))
        report['api_processes_stopped']=all(p.poll() is not None for p in processes)
        report['success']=report['success'] and not failures
        report['thirty_minute_gate_passed']=bool(report['success'] and args.mixed_seconds>=1800 and
                                                report.get('mixed_elapsed_seconds',0)>=1800)
        report['job_end_to_end_seconds']={phase:distribution([r['end_to_end_seconds'] for r in records if r['phase']==phase])
                                         for phase in ('lpc','egg','acoustic')}
        report['stage_measurements']={}
        for stage in sorted({s['stage'] for s in samples}):
            rows=[s for s in samples if s['stage']==stage and s['category'].startswith('light_')]
            report['stage_measurements'][stage]=dict(light_requests=len(rows),
                server_seconds=distribution([r['server_seconds'] for r in rows if r['server_seconds'] is not None]))
        system_path=root/'system.jsonl'
        if system_path.exists():
            observations=[json.loads(line) for line in system_path.read_text().splitlines()]
            if observations:
                peaks=[int(group['memory_peak_bytes']) for row in observations for group in row['groups'] if 'memory_peak_bytes' in group]
                collector_records=[json.loads(p.read_text()) for p in (root/'collectors').glob('*.json')]
                exact_peaks=[r['process']['memory_peak_bytes'] for r in collector_records if 'memory_peak_bytes' in r['process']]
                report['resources']=dict(samples=len(observations),
                    min_mem_available_bytes=min(r['memory']['MemAvailable'] for r in observations),
                    max_observed_group_memory_bytes=max(peaks) if peaks else None,
                    collector_group_peak_bytes=max(exact_peaks) if exact_peaks else None,
                    collector_samples=len(collector_records),
                    collector_cleaned=sum(r['process'].get('cleaned') is True for r in collector_records),
                    max_api_pss_bytes={str(p.pid):max((r['processes'].get(str(p.pid),{}).get('Pss',0) for r in observations),default=None) for p in processes},
                    disk_free_min_bytes=min(r['disk_free_bytes'] for r in observations),
                    evidence_bytes_max=max(r['evidence_bytes'] for r in observations))
        write_json(root/'report.json',report)
    return 0 if report['success'] else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mixed-seconds',type=int,default=0)
    parser.add_argument('--authorized-test-directory',action='store_true')
    parser.add_argument('--serve',type=int,choices=(0,1),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.serve is not None: serve(args.output,args.serve); return 0
    if args.template is None: parser.error('--template is required')
    if args.mixed_seconds<0: parser.error('--mixed-seconds cannot be negative')
    return benchmark(args)


if __name__=='__main__':
    raise SystemExit(main())
