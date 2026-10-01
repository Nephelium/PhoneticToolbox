import hashlib
import io
import numpy as np
import pytest
import soundfile as sf
from ptb_desktop.file_provider import FileProvider, FileAccessError
from ptb_desktop.annotation_audio import preview_audio


def test_recursive_list_keeps_pair_paths_and_scoped_file_authority(tmp_path):
    for folder in (tmp_path, tmp_path/'甲', tmp_path/'乙'):
        folder.mkdir(exist_ok=True)
        (folder/'same.wav').write_bytes(b'audio')
        (folder/'same.TextGrid').write_bytes(b'grid')
    provider = FileProvider()
    grant = provider.choose('input', lambda: tmp_path)
    assert len(provider.list(grant['id'])) == 2
    recursive = provider.scan(grant['id'])
    assert len(recursive) == 6
    assert {f['name'] for f in recursive if f['kind']=='audio'} == {'same.wav', '甲/same.wav', '乙/same.wav'}
    assert all(provider.read(f['id'])[0] in (b'audio', b'grid') for f in recursive)
    assert recursive == provider.scan(grant['id'])
    item = next(f for f in recursive if f['name']=='乙/same.wav')
    (tmp_path/'乙/same.wav').write_bytes(b'changed')
    with pytest.raises(FileAccessError):
        provider.read(item['id'])


def test_multichannel_compact_preview_preserves_time_source_hash_and_channels():
    rate = 48000
    t = np.arange(rate*2+13)/rate
    samples = np.column_stack((.25*np.sin(2*np.pi*120*t), .4*np.sin(2*np.pi*250*t)))
    source = io.BytesIO()
    sf.write(source, samples, rate, format='WAV', subtype='FLOAT')
    original = source.getvalue()
    raw, sha, duration, note = preview_audio(source, 160_000, preserve_channels=True)
    preview, fs = sf.read(io.BytesIO(raw), always_2d=True)
    assert len(raw) <= 160_000 and preview.shape[1] == 2
    assert abs(len(preview)/fs-duration) <= 1/fs
    assert duration == len(samples)/rate
    assert sha == hashlib.sha256(original).hexdigest() and note
    assert np.max(np.abs(preview[:, 0])) < .26
    assert np.max(np.abs(preview[:, 1])) > .39


def test_recursive_results_save_beside_sources_and_never_clobber(tmp_path):
    from urllib.parse import urlsplit, parse_qs
    from uuid import uuid4
    from ptb_desktop.task_bridge import TaskBridge
    for name in ('甲','乙'):
        sub=tmp_path/name;sub.mkdir();(sub/'same.wav').write_bytes(name.encode())
    provider=FileProvider();grant=provider.choose('input',lambda:tmp_path)
    inputs=[f for f in provider.scan(grant['id']) if f['kind']=='audio']
    batch_id=str(uuid4());jobs={};payloads={}
    for index in range(2):
        job_id=str(uuid4());file_id=str(uuid4());raw=f'param {index}'.encode();payloads[file_id]=raw
        jobs[job_id]={'id':job_id,'state':'succeeded','operation':'acoustic_analysis','result_manifest':{'files':[{'id':file_id,'name':'result.xlsx','size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}]}}
    batch={'id':batch_id,'audio_names':['same.wav','same.wav'],'summary':{'items':[{'index':i,'job_id':job,'state':'succeeded'} for i,job in enumerate(jobs)]}}
    class Service:
        def import_input(self,raw,name,role):return {'asset_id':str(uuid4()),'sha256':hashlib.sha256(raw).hexdigest()}
        def request(self,*a,**k):return batch
        def get(self,path):return batch if '/batches/' in path else jobs[path.rsplit('/',1)[-1]]
        def binary(self,path):
            query=parse_qs(urlsplit(path).query);raw=payloads[urlsplit(path).path.rsplit('/',1)[-1]]
            offset=int(query['offset'][0]);return raw[offset:offset+int(query['size'][0])]
    bridge=TaskBridge(provider,Service())
    bridge.invoke({'op':'submit','operation':'acoustic_analysis','inputs':[{'audio':f['id']} for f in inputs],'config':None,'layer':None,'idempotency_key':'fixture'})
    result=bridge.save(batch_id,grant['id'],beside_sources=True)
    assert result['count']==2 and set(result['saved'])=={'甲/same.xlsx','乙/same.xlsx'}
    assert not (tmp_path/'same.xlsx').exists()
    assert bridge.save(batch_id,grant['id'],beside_sources=True)==result
    (tmp_path/'甲/same.xlsx').write_bytes(b'external')
    bridge.save(batch_id,grant['id'],beside_sources=True)
    assert (tmp_path/'甲/same.xlsx').read_bytes()==b'external'


def test_cached_preview_rechecks_source_identity(tmp_path):
    path=tmp_path/'short.wav';sf.write(path,np.zeros((8000,2)),8000,subtype='PCM_16')
    provider=FileProvider();grant=provider.choose('input',lambda:tmp_path);file=provider.list(grant['id'])[0]
    raw,sha,duration,_=provider.preview_payload(file['id'])
    assert duration==1 and provider.spectrogram_payload(file['id'])==(raw,sha)
    path.write_bytes(b'changed')
    with pytest.raises(FileAccessError):provider.preview_payload(file['id'])
    provider.close();assert provider.preview_cache is None


def test_spectrogram_refills_opted_in_preview_after_other_module_changes_audio(tmp_path):
    for name in ('first.wav','second.wav'):
        sf.write(tmp_path/name,np.zeros((48000,2)),48000,subtype='FLOAT')
    provider=FileProvider(max_bytes=160_000)
    grant=provider.choose('input',lambda:tmp_path);first,second=provider.list(grant['id'])
    first_preview=provider.audio_preview(first['id'])
    provider.audio_preview(second['id'])
    raw,sha=provider.spectrogram_payload(first['id'])
    assert len(raw)<=160_000 and sha==first_preview['sha256']
    assert provider.preview_cache[0]==first['id']
