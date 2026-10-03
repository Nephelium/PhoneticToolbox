"""M01-D serialization boundary faults; actual native/goldens use verify_m01_contract."""
from dataclasses import asdict
import numpy as np
import pytest
from pydantic import ValidationError
from phonetic_core.models.audio import AudioInput
from phonetic_core.models.acoustic import AnalysisResult
from phonetic_core.catalog import PARAMETER_MAPPING
from phonetic_core.acoustic.catalog import PARAMETER_MAPPING as old_catalog
from ptb_worker.acoustic_result import build_acoustic_result,to_core_config,SOURCE_IDS
from ptb_api.acoustic_models import AcousticRequest,AcousticInputSnapshot


def context():
    req=AcousticRequest(project_id='00000000-0000-4000-8000-000000000001',idempotency_key='boundary-result-01',
        inputs={'audio':{'asset_id':'00000000-0000-4000-8000-000000000002','sha256':'a'*64}},
        config={'selection':{'keys':['pF0']}})
    inp=AcousticInputSnapshot(**req.inputs.audio.model_dump(),role='audio',expires_at=200.)
    audio=AudioInput(np.zeros((44100,2),dtype=np.int16),44100)
    result=AnalysisResult(time_axis=np.array([0.,.005,.01,.015]),
        f0_praat=np.array([0.,np.nan,np.inf,-np.inf]),sampling_rate=44100,
        config_snapshot=asdict(to_core_config(req.config)))
    return result,audio,req,[inp]


def test_actual_array_mask_decode_identity_and_request_isolation():
    result,audio,req,inputs=context()
    wire=build_acoustic_result(result,audio,req,inputs)
    assert wire.numeric[0].nonfinite==[0,1,2,3]
    assert wire.numeric[0].values==[0.,None,None,None]
    assert wire.metadata.decoded.sample_count==44100
    assert wire.metadata.decoded.channels==2
    req.config.settings.min_f0=70.
    assert wire.metadata.config.settings.min_f0==30.
    assert old_catalog is PARAMETER_MAPPING


@pytest.mark.parametrize('fault', ['config','sample_rate','input_hash','input_missing','native_hash','native_policy'])
def test_serialization_cannot_relabel_unverified_core_metadata(fault):
    result,audio,req,inputs=context()
    if fault=='config':result.config_snapshot['min_f0']=70.
    elif fault=='sample_rate':result.sampling_rate=16000
    elif fault=='input_hash':inputs[0].sha256='b'*64
    elif fault=='input_missing':inputs=[]
    elif fault=='native_hash':result.backend_events=[dict(stage='reaper',actual='native_reaper')]
    elif fault=='native_policy':
        req.config.backend_policy.reaper='native_required'
        result.backend_events=[dict(stage='reaper',actual='reaper_python')]
    with pytest.raises((ValueError,ValidationError)):build_acoustic_result(result,audio,req,inputs)


def test_unknown_error_text_never_crosses_wire_and_disabled_is_explicit():
    result,audio,req,inputs=context()
    result.backend_events=[dict(stage='wm_f0',actual='unavailable',reason='sensitive host exception text')]
    req.config.backend_policy.reaper='disabled'
    result.config_snapshot=asdict(to_core_config(req.config))
    wire=build_acoustic_result(result,audio,req,inputs)
    assert wire.metadata.backends[0].reason=='backend_failure'
    assert wire.metadata.backends[1].actual=='disabled'
    assert 'sensitive' not in wire.model_dump_json()


def test_recorded_sources_exist_in_registry():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    registry=json.loads((root/'third_party/source-registry.json').read_text('utf-8'))
    assert set(SOURCE_IDS)<={s['id'] for s in registry['sources']}
