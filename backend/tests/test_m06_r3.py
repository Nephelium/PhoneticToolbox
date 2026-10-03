"""Actual owned REAPER and persistent M06 tasks on a copied test schema."""
import json
import io
from pathlib import Path
import numpy as np
import pytest
from test_m06 import runtime,run,body,ROOT


@pytest.mark.parametrize('method',['praat_cc','praat_ac','reaper'])
def test_real_method_snapshot_and_cleanup(runtime,method):
    _,files=runtime
    manifest=json.loads((ROOT/'resources/manifests/acoustic.json').read_text('utf8'))
    files.reaper_binary=ROOT/manifest['resources'][0]['validation_source']
    request=body('extract');request['config']['f0_method']=method
    from scipy.io import wavfile
    t=np.arange(16000)/16000
    y=sum(np.sin(2*np.pi*150*k*t)/k for k in range(1,41))*.1
    stream=io.BytesIO();wavfile.write(stream,16000,y.astype(np.float32))
    request['audio']=files.import_input(stream.getvalue(),'harmonic150.wav','audio')
    result,evidence=run(runtime,request)
    assert result['state']=='succeeded',result
    output=next(f for f in result['result_manifest']['files'] if f['name']=='m06.ptb.json')
    meta=json.loads(files.read_result('local',output['id'],0,output['size_bytes']))
    assert meta['config']['f0_method']==method
    d=meta['diagnostics'];assert d['actual_f0_backend']==('native_reaper' if method=='reaper' else method)
    assert d['extraction_revision']=='m06-extract/2'
    measured=[v for v in d['measured_f0_hz'] if v is not None]
    assert measured and np.isfinite(measured).all()
    assert evidence['cleaned'] and evidence['memory_peak_bytes']>0
    if method=='reaper':
        from ptb_worker.native.reaper import REAPER_SHA256
        assert d['reaper_binary_sha256']==REAPER_SHA256 and 'SRC-REAPER' in meta['source_ids']


def test_missing_reaper_fails_without_fallback(runtime):
    _,files=runtime;request=body('extract');request['config']['f0_method']='reaper'
    request['audio']=files.import_input((ROOT/'tests/fixtures/m06/source.wav').read_bytes(),'source.wav','audio')
    result,_=run(runtime,request)
    assert result['state']=='failed' and result['error_code']=='m06_reaper_unavailable'
    assert result['result_manifest'] is None
