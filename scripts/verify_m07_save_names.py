"""Owned M07 export host: real service, file grants and native writes."""
import base64
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import zipfile

from verification_artifacts import verification_run
from workbench_source import configure

ROOT = Path(__file__).resolve().parents[1]


def main():
    configure()
    import numpy as np
    from scipy.io import wavfile
    from ptb_desktop.file_provider import FileProvider
    from ptb_desktop.task_bridge import TaskBridge
    from ptb_desktop.local_service import LocalService
    from ptb_worker.local_acoustic_files import initialize_local_files

    with verification_run('m07-save-names', 'chrome',
                          'Recreate public tests/fixtures/m07/v2.npz inputs 0/1 at fixture rates; '
                          'real six nine-step groups; native complete-group exports and web ZIPs') as (out, scratch):
        inputs = scratch / 'inputs'; inputs.mkdir()
        saved = scratch / 'saved'; saved.mkdir()
        cases = json.loads((ROOT / 'tests/fixtures/m07/v2.json').read_text('utf-8'))['cases']
        with np.load(ROOT / 'tests/fixtures/m07/v2.npz', allow_pickle=False) as data:
            for index in (0, 1):
                wavfile.write(inputs / f'input{index}.wav', cases[index]['fs'], data[f'input{index}'])
        original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        db = scratch / 'jobs.sqlite3'
        with sqlite3.connect(db) as dest:
            for migration in ('002_jobs_sqlite.sql', '005_acoustic_batches_sqlite.sql'):
                dest.executescript((ROOT / 'backend/migrations' / migration).read_text('utf-8'))
        cache = scratch / 'cache'; cache.mkdir(); initialize_local_files(cache)
        provider = FileProvider()
        source = provider.choose('input', lambda: str(inputs))
        target = provider.choose('output', lambda: str(saved))
        sentinel = None
        try:
            with LocalService(db, local_files_root=cache) as service:
                bridge = TaskBridge(provider, service)
                print(json.dumps(dict(ready=True, out=str(out), scratch=str(scratch))), flush=True)
                for line in sys.stdin:
                    request = json.loads(line); body = request['body']
                    try:
                        if request['channel'] == 'task':
                            value = bridge.invoke(body)
                        elif request['channel'] == 'test':
                            if body['op'] == 'sentinel':
                                sentinel = saved / body['folder'] / 'step01.wav'
                                sentinel.write_bytes(b'preserve-existing-export')
                                value = True
                            elif body['op'] == 'audit':
                                summaries = []
                                for folder in sorted(saved.iterdir()):
                                    meta = json.loads((folder / 'm07.ptb.json').read_text('utf-8'))
                                    for entry in meta['outputs']:
                                        path = folder / entry['name']
                                        if path != sentinel:
                                            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
                                    summaries.append(dict(name=folder.name, kind=meta['continuum_type'],
                                                          reverse=meta['reverse_direction'], steps=meta['generation']['step_count']))
                                assert sentinel is not None and sentinel.read_bytes() == b'preserve-existing-export'
                                archives = []
                                for path in sorted((scratch / 'downloads').glob('*.zip')):
                                    with zipfile.ZipFile(path) as archive:
                                        assert archive.testzip() is None
                                        meta = json.loads(archive.read('m07.ptb.json'))
                                        assert set(archive.namelist()) == {e['name'] for e in meta['outputs']} | {'m07.ptb.json'}
                                        for entry in meta['outputs']:
                                            assert hashlib.sha256(archive.read(entry['name'])).hexdigest() == entry['sha256']
                                        archives.append(dict(name=path.name, files=len(archive.namelist())))
                                assert len(summaries) == 7 and len(archives) == 6
                                assert original == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
                                value = dict(folders=summaries, archives=archives, inputs_unchanged=True, conflict_preserved=True)
                                (out / 'exports.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
                            else:
                                raise ValueError('Unknown test operation')
                        else:
                            op = body['op']
                            if op == 'hello':value = dict(kind='desktop', session=provider.session, api_version='1.1.0', tasks=True)
                            elif op == 'fonts':value = ['Arial', 'Microsoft YaHei', 'Doulos SIL']
                            elif op == 'choose':value = target if body['purpose'] == 'output' else source
                            elif op == 'list':value = provider.list(body.get('id') or source['id'])
                            elif op == 'read':
                                raw, sha = provider.read(body['id']); value = dict(base64=base64.b64encode(raw).decode(), sha256=sha)
                            else:raise ValueError('Unknown test transport')
                        response = dict(id=request['id'], ok=True, value=value)
                    except Exception as exc:
                        response = dict(id=request['id'], ok=False, error=str(exc))
                    print(json.dumps(response, ensure_ascii=False), flush=True)
        finally:
            provider.close()


if __name__ == '__main__':
    main()
