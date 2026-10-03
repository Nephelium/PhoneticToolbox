"""R4 durable resynthesis, owned process, artifacts, and source identity."""
import hashlib
import io
import json
import numpy as np
import pytest
from scipy.io import wavfile
from test_m06 import runtime,run,body,ROOT


@pytest.mark.parametrize('method',['world','psola'])
def test_real_resynthesis_bundle(runtime,method):
    _,files=runtime
    raw=(ROOT/'tests/fixtures/m06/source.wav').read_bytes()
    c=body('resynthesize',.6)
    c['config']['render'].update(method=method,source_sha256=hashlib.sha256(raw).hexdigest())
    c['audio']=files.import_input(raw,'source.wav','audio')
    result,evidence=run(runtime,c)
    assert result['state']=='succeeded',result
    data={f['name']:files.read_result('local',f['id'],0,f['size_bytes']) for f in result['result_manifest']['files']}
    assert set(data)=={'synthesis.wav','m06.ptb.json','parameters.csv','analysis.npz'}
    rate,y=wavfile.read(io.BytesIO(data['synthesis.wav']));assert rate==16000 and len(y)==9600 and y.dtype==np.float32
    meta=json.loads(data['m06.ptb.json']);assert meta['computation_revision']=='m06-'+method+'/1'
    assert meta['config']['render']['source_sha256']==hashlib.sha256(raw).hexdigest()
    with np.load(io.BytesIO(data['analysis.npz']),allow_pickle=False) as arrays:
        assert len(arrays['source_f0_hz'])==len(arrays['source_times_s'])
        if method=='world':assert arrays['spectral_envelope_power'].shape==arrays['aperiodicity_amplitude_ratio'].shape
    assert evidence['cleaned'] and evidence['memory_peak_bytes']>0


def test_source_mismatch_fails_without_publishing(runtime):
    _,files=runtime;request=body('resynthesize',.6);request['config']['render']['method']='world'
    request['audio']=files.import_input((ROOT/'tests/fixtures/m06/source.wav').read_bytes(),'source.wav','audio')
    result,_=run(runtime,request)
    assert result['state']=='failed' and result['error_code']=='m06_resynthesis_source_mismatch'
    assert result['result_manifest'] is None


@pytest.mark.parametrize('method',['world','psola'])
def test_natural_extraction_preserves_config_and_binds_source(runtime,method):
    _,files=runtime;request=body('extract',.6);request['config']['render']['method']=method
    raw=(ROOT/'tests/fixtures/m06/source.wav').read_bytes()
    request['audio']=files.import_input(raw,'source.wav','audio')
    result,_=run(runtime,request);assert result['state']=='succeeded',result
    f=next(x for x in result['result_manifest']['files'] if x['name']=='m06.ptb.json')
    meta=json.loads(files.read_result('local',f['id'],0,f['size_bytes']))
    assert meta['config']['render']['source_sha256']==hashlib.sha256(raw).hexdigest()
    assert meta['diagnostics']['extraction_revision']=='m06-natural-extract/1'
    assert meta['config']['curves']['AV']==request['config']['curves']['AV']


@pytest.mark.parametrize('method',['world','psola'])
def test_running_resynthesis_cancel_releases_resources(runtime,method):
    import threading
    _,files=runtime;raw=(ROOT/'tests/fixtures/m06/source.wav').read_bytes()
    request=body('resynthesize',.6);request['config']['render'].update(method=method,source_sha256=hashlib.sha256(raw).hexdigest())
    request['audio']=files.import_input(raw,'source.wav','audio');stop=threading.Event()
    result,evidence=run(runtime,request,stop,on_started=lambda pid:stop.set())
    assert result['state']=='cancelled' and result['result_manifest'] is None
    assert evidence['cleaned']


def test_praat_task_seed_reproduces_waveform():
    from ptb_worker.m06_child import compute
    from phonetic_core.synthesis.klatt.api import export_parameters
    raw=(ROOT/'tests/fixtures/m06/source.wav').read_bytes();c=body(duration=.6)['config'];sha=hashlib.sha256(raw).hexdigest()
    c['render'].update(method='psola',source_sha256=sha)
    header=dict(parameters=export_parameters(c),action='resynthesize',seed=42,input_sha256=sha)
    a=dict(compute(header,raw));b=dict(compute(header,raw))
    assert a['synthesis.wav']==b['synthesis.wav']
    assert json.loads(a['m06.ptb.json'])['diagnostics']['praat_seed']==42
