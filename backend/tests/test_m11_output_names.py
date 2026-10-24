"""Result naming through the real local publisher and native save bridge.

Only MFA resource selection/execution is synthetic; no user corpus is opened.
"""
import base64
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from test_local_retention import storage
from test_mfa_retention import registry, job
from ptb_worker.store import public, LOCAL_PROJECT
from ptb_worker.m11_executor import textgrid_names


@pytest.mark.parametrize('names,expected', [
    (['女童_平静.wav', '女童_愤怒1.wav', '女老_愉快1.wav', '女老_愤怒1.wav', '男童_愤怒1.wav', '男童_愤怒2.wav'],
     ['女童_平静.TextGrid', '女童_愤怒1.TextGrid', '女老_愉快1.TextGrid', '女老_愤怒1.TextGrid', '男童_愤怒1.TextGrid', '男童_愤怒2.TextGrid']),
    (['001_原始 录音.v1.WAV'], ['001_原始 录音.v1.TextGrid']),
    (['甲/相同.wav', '乙/相同.wav', '丙/相同_2.wav', '丁/相同.wav'],
     ['相同.TextGrid', '相同_3.TextGrid', '相同_2.TextGrid', '相同_4.TextGrid']),
    (['甲/A.wav', '乙/a.wav', '丙/a_2.wav'], ['A.TextGrid', 'a_3.TextGrid', 'a_2.TextGrid']),
])
def test_textgrid_names_preserve_originals_and_avoid_collisions(names, expected):
    assert textgrid_names([dict(name=name) for name in names]) == expected


def test_published_names_survive_native_save_without_prefixes(storage, registry, monkeypatch, tmp_path):
    import ptb_worker.m11_executor as executor
    from phonetic_core.transcription.mfa_name_codec import encode_fs_name
    from ptb_desktop.file_provider import FileProvider
    from ptb_desktop.m11_bridge import M11Bridge

    store, files, _ = storage
    corpus = []
    for name in ('女童_平静.wav', '女老_愉快1.wav', '001_原始 录音.v1.WAV'):
        corpus.append(dict(name=name, audio=files.import_input(b'fixture audio', name, 'audio'),
                           transcript=files.import_input(b'a', Path(name).with_suffix('.lab').name, 'transcript'),
                           transcript_format='.lab'))
    with files.batch_transaction() as tx:
        inputs = [asset for item in corpus for asset in files.validate_batch_input(tx, 'local', LOCAL_PROJECT,
                  dict(audio=item['audio'], transcript=item['transcript']), tx.now())]
    model = registry/'registered-model.zip'; dictionary = registry/'registered-dictionary.dict'
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    snapshot = dict(operation='mfa_alignment', execution_route='desktop-local',
                    request=dict(runtime_id='runtime', model_id='model', corpus=corpus, config={}),
                    runtime_fingerprint='f'*64, model_sha256=sha(model), dictionary_sha256=sha(dictionary),
                    input_assets=inputs, config=dict(inputs=[item['id'] for item in inputs], max_output_bytes=64_000_000))
    identity = job(storage, snapshot=snapshot)
    monkeypatch.setattr(executor, 'select', lambda *a, **kw: (
        dict(id='runtime', version='3.3.8', path=str(registry), fingerprint='f'*64),
        dict(id='model', model=str(model), dictionary=str(dictionary), model_sha256=sha(model), dictionary_sha256=sha(dictionary))))
    raw = b'fixture TextGrid bytes'
    def aligned(_runtime, root, request, **kwargs):
        assert request['expected_files'] == len(corpus)
        for item in corpus:
            path = root/'output'/Path(encode_fs_name(item['name'])).with_suffix('.TextGrid')
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        kwargs['evidence']['group_cleaned'] = True
        return dict(versions={}, textgrids=[])
    monkeypatch.setattr(executor, 'run', aligned)
    with store.transaction() as tx:
        claim = dict(store._row(tx, identity[0]))
    executor.execute_claim(store, claim, identity[1], threading.Event())
    with store.transaction() as tx:
        result = public(dict(store._row(tx, identity[0])))
    assert result['state'] == 'succeeded', result
    grids = ['女童_平静.TextGrid', '女老_愉快1.TextGrid', '001_原始 录音.v1.TextGrid']
    expected = sorted([*grids, 'm11-provenance.json'])
    assert sorted(item['name'] for item in result['result_manifest']['files']) == expected

    provider = FileProvider(); output = tmp_path/'saved'; output.mkdir()
    original = output/grids[0]; original.write_bytes(b'original TextGrid')
    grant = provider.choose('output', lambda: str(output))
    def read_result(body):
        item = next(f for f in result['result_manifest']['files'] if f['id'] == body['id'])
        return dict(base64=base64.b64encode(files.read_result('local', item['id'], 0, item['size_bytes'])).decode('ascii'))
    bridge = M11Bridge(SimpleNamespace(service=SimpleNamespace(get=lambda _: result), provider=provider,
                                      invoke=read_result, record_export=lambda _: None))
    for _ in range(2):
        saved = bridge.invoke(dict(op='m11_save', job=result['id'], directory=grant['id']))
        assert sorted(saved['saved']) == expected
        for name in grids:
            assert (output/saved['directory']/name).read_bytes() == raw
        provenance = json.loads((output/saved['directory']/'m11-provenance.json').read_text('utf8'))
        assert provenance['inputs'] == corpus
    assert len(list(output.glob('MFA-*'))) == 2
    assert original.read_bytes() == b'original TextGrid'
