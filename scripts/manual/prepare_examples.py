"""Prepare software-only manual audio in owned output folders, preserving source files."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ('desktop/src', 'backend/src', 'packages/phonetic_core/src', 'scripts')
for subdirectory in SOURCE_DIRS:
    sys.path.insert(0, str(ROOT / subdirectory))
# The owned local-service child must read the same checkout as the parent.
os.environ['PYTHONPATH'] = os.pathsep.join(str(ROOT / subdirectory) for subdirectory in SOURCE_DIRS)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def install_asset(project: dict, project_root: Path, asset_id: str, payload: bytes, name: str,
                  *, source_type='recording', caption='', mime='audio/wav', kind='audio'):
    relative = f'assets/software-only/{name}'
    destination = project_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha(destination.read_bytes()) != sha(payload):
        raise RuntimeError(f'Existing asset differs; preserve it before changing {asset_id}')
    destination.write_bytes(payload)
    asset = dict(id=asset_id, path=relative, kind=kind, mime=mime, sha256=sha(payload),
                 sourceType={'recording':'自然录音','processed':'v3 实际处理'}.get(source_type,source_type), source='作者指定录音及 v3 本地实际处理',
                 caption=caption, distribution='software-only', git=False)
    if kind=='audio':
        from io import BytesIO
        from scipy.io import wavfile
        rate,samples=wavfile.read(BytesIO(payload))
        asset.update(sampleRate=int(rate),channels=1 if samples.ndim==1 else int(samples.shape[1]),duration=len(samples)/rate)
    project['assets'] = [a for a in project.get('assets', []) if a['id'] != asset_id] + [asset]
    return asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, default=ROOT / 'local-data/manual-authoring/sources.json')
    parser.add_argument('--project', type=Path, default=ROOT / 'manual')
    parser.add_argument('--m08', action='store_true', help='Actually process specified single-syllable material in v3')
    parser.add_argument('--m06',action='store_true')
    parser.add_argument('--m07',action='store_true')
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text(encoding='utf-8'))
    project_path = args.project / 'project.json'
    project = json.loads(project_path.read_text(encoding='utf-8'))
    evidence = ROOT / 'output/manual-work/examples' / uuid4().hex
    evidence.mkdir(parents=True)
    audit = {'run_id': evidence.name, 'inputs': [], 'outputs': [],
             'distribution': 'software-only', 'public_github': False, 'success': False}
    for key, asset_id, name in [
        ('egg', 'egg-stereo-original', 'egg-stereo-original.wav'),
        ('single', 'single-original', 'single-original.wav'),
        ('target', 'single-target', 'single-target.wav'),
        ('sentence', 'sentence-original', 'sentence-original.wav'),
    ]:
        path = Path(sources[key])
        payload = path.read_bytes()
        audit['inputs'].append({'id': asset_id, 'path': str(path), 'sha256': sha(payload)})
        audit['outputs'].append(install_asset(project, args.project, asset_id, payload, name,
                                              caption={'egg':'同步音频/EGG 原始双声道，试听请使用单独音频声道',
                                                       'single':'自然单音节原声', 'target':'自然单音节目标原声',
                                                       'sentence':'自然语音与标注示例原声'}[key]))
    from scipy.io import wavfile
    from io import BytesIO
    rate, values = wavfile.read(sources['egg'])
    audio_channel = int(sources['egg_audio_channel'])
    if values.ndim != 2 or audio_channel >= values.shape[1]:
        raise ValueError('Confirmed EGG/audio channel mapping is required')
    audio = values[:, audio_channel].copy()
    output = BytesIO()
    wavfile.write(output, rate, audio)
    audit['outputs'].append(install_asset(project, args.project, 'egg-audio-original', output.getvalue(),
                                          'egg-audio-original.wav', source_type='processed',
                                          caption='同步记录的音频声道，未进行归一化或重采样'))
    audit['egg_roles'] = {'audio_channel_zero_based': audio_channel,
                           'egg_channel_zero_based': int(sources['egg_signal_channel'])}
    if args.m08:
        from ptb_desktop.file_provider import FileProvider
        from ptb_desktop.task_bridge import TaskBridge
        from ptb_desktop.local_service import LocalService
        from ptb_worker.store import LOCAL_PROJECT
        from verify_m08_wiring import setup, wait
        run, db, cache = setup()
        audit['task_run'] = str(run)
        with LocalService(db, local_files_root=cache) as service:
            bridge = TaskBridge(FileProvider(), service)
            ref = service.import_input(Path(sources['single']).read_bytes(), '自然单音节示例.wav', 'audio')
            variants = [
                ('m08-slow', 'm08-slow.wav', {'action':'transform','speed':0.8,'pitch_ratio':1.0,'pitch_hz':0.0},
                 'v3 实际变速结果：语速倍率 0.8，音高不变'),
                ('m08-raised', 'm08-raised.wav', {'action':'transform','speed':1.0,'pitch_ratio':1.2,'pitch_hz':0.0},
                 'v3 实际变调结果：语速倍率 1.0，基频倍率 1.2'),
            ]
            for asset_id, filename, config, caption in variants:
                job = service.request('/api/v1/jobs/m08/create', 'POST', {
                    'project_id': LOCAL_PROJECT, 'idempotency_key': uuid4().hex, 'audio': ref, 'config': config})
                done = wait(service, job)
                if done['state'] != 'succeeded':
                    raise RuntimeError(f'M08 actual processing failed: {done}')
                listing = service.get('/api/v1/jobs/m08/list/' + LOCAL_PROJECT)
                result = next(item for item in listing if item['id'] == job['id'])['results'][0]
                read = bridge.invoke({'op':'result','job':job['id'],'id':result['id']})
                payload = base64.b64decode(read['base64'])
                if sha(payload) != result['sha256']:
                    raise AssertionError('Persisted result hash mismatch')
                entry = install_asset(project, args.project, asset_id, payload, filename,
                                      source_type='processed', caption=caption)
                entry.update(task_id=job['id'], config=config)
                audit['outputs'].append(entry)
    if args.m06 or args.m07:
        from ptb_desktop.local_service import LocalService
        from ptb_worker.store import LOCAL_PROJECT
        from verify_m06_wiring import setup,wait,read
        from phonetic_core.synthesis.klatt.api import defaults,export_parameters
        run,db,cache=setup();audit['synthesis_task_run']=str(run)
        with LocalService(db,local_files_root=cache) as service:
            refs={key:service.import_input(Path(sources[key]).read_bytes(),key+'-syllable.wav','audio') for key in ('single','target')}
            def task(module,body):
                job=wait(service,service.request('/api/v1/jobs/'+module+'/create','POST',dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,**body)))
                if job['state']!='succeeded':raise RuntimeError('Actual '+module+' processing failed: '+json.dumps(job))
                raw={f['name']:read(service,f) for f in job['result_manifest']['files']}
                for f in job['result_manifest']['files']:
                    if sha(raw[f['name']])!=f['sha256']:raise AssertionError('Result identity changed')
                return job,raw
            if args.m06:
                parameters=service.import_input(export_parameters(defaults()).encode(),'m06-defaults.csv','table')
                extracted,raw=task('m06',dict(action='extract',parameters=parameters,audio=refs['single']))
                config=json.loads(raw['m06.ptb.json'])['config']
                parameters=service.import_input(export_parameters(config).encode(),'m06-extracted.csv','table')
                synthesized,raw=task('m06',dict(action='synthesize',parameters=parameters))
                entry=install_asset(project,args.project,'m06-extracted-klatt',raw['synthesis.wav'],'m06-extracted-klatt.wav',source_type='processed',
                    caption='v3 实际 Klatt 合成：由同一自然单音节提取参数后合成。供操作对照，不表示逐样本或音色复原。')
                audit['outputs'].append(dict(**entry,task_id=synthesized['id'],analysis_task_id=extracted['id'],config=config))
            if args.m07:
                common=dict(source=refs['single'],target=refs['target'],analysis=dict(max_f0_hz=600))
                analyzed,raw=task('m07',dict(action='analyze',**common))
                meta=json.loads(raw['m07.ptb.json'])
                applied,_=task('m07',dict(action='apply',analysis_job_id=analyzed['id'],controls=meta['controls'],**common))
                generated,raw=task('m07',dict(action='generate',analysis_job_id=applied['id'],continuum_type=2,generation=dict(step_count=9),**common))
                for name,asset_id,label in [('step01.wav','m07-step-01','第 1 步'),('step05.wav','m07-step-05','第 5 步'),
                                            ('step09.wav','m07-step-09','第 9 步'),('combined_steps.wav','m07-combined','九步整组')]:
                    entry=install_asset(project,args.project,asset_id,raw[name],asset_id+'.wav',source_type='processed',
                        caption='v3 实际发声类型合成：类型 2，源→目标，九步连续统的'+label+'。F0 提取上限在本例设为 600 Hz。')
                    audit['outputs'].append(dict(**entry,task_id=generated['id'],analysis_task_id=analyzed['id'],config=common['analysis'],continuum_type=2,step_count=9))
    # Source records remain in ignored local evidence, never public content.
    for record in audit['inputs']:
        if sha(Path(record['path']).read_bytes()) != record['sha256']:
            raise AssertionError('Source file changed during example preparation')
    save_json(project_path, project)
    audit['success'] = True
    save_json(evidence / 'report.json', audit)
    print(json.dumps({'success':True,'evidence':str(evidence),'assets':len(audit['outputs']),
                      'private_sources_preserved':True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
