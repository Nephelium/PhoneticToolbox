"""P16 actual Windows REAPER/Python computations and result roundtrip, synthetic only."""
from pathlib import Path
import hashlib
import sys
import numpy as np
import pytest
from phonetic_core.models.audio import AudioInput
from phonetic_core.ports.acoustic import AcousticBackends
from phonetic_core.services.acoustic import analyze_audio
from ptb_api.acoustic_models import AcousticRequest,AcousticInputSnapshot,AcousticResult
from ptb_worker.acoustic_result import to_core_config,build_acoustic_result
from ptb_worker.io.scratch import Scratch
from ptb_worker.native.reaper import Reaper
from ptb_worker.reaper_policy import PolicyReaper


@pytest.mark.skipif(sys.platform!='win32',reason='This resource is the registered Windows REAPER binary')
@pytest.mark.parametrize('policy',['native_required','python_only','native_then_python'])
def test_actual_backends_produce_identified_roundtrippable_results(tmp_path,policy):
    root=Path(__file__).resolve().parents[2]
    # Rich periodic excitation: a pure sine is legitimately unvoiced for REAPER.
    time=np.arange(44100)/44100
    waveform=sum(np.sin(2*np.pi*180*k*time)/k for k in range(1,16))
    samples=(waveform/np.max(np.abs(waveform))*16000).astype(np.int16)
    digest=hashlib.sha256(samples.tobytes()).hexdigest()
    request=AcousticRequest(project_id='00000000-0000-4000-8000-000000000001',idempotency_key='review-science-01',
        inputs={'audio':{'asset_id':'00000000-0000-4000-8000-000000000002','sha256':digest}},
        config={'selection':{'keys':['pF0','rF0','pF1']},'backend_policy':{'reaper':policy}})
    audio=AudioInput(samples,44100)
    with Scratch(tmp_path,16_000_000) as scratch:
        native=lambda:Reaper(root/'phonetic_toolbox/core/acoustic/reaper.exe',scratch)
        backend=PolicyReaper(policy,native)
        result=analyze_audio(audio,to_core_config(request.config),backends=AcousticBackends(reaper=backend))
        inputs=[AcousticInputSnapshot(**request.inputs.audio.model_dump(),role='audio',expires_at=200.)]
        wire=build_acoustic_result(result,audio,request,inputs,native_sha256=backend.native_sha256)
    assert wire.metadata.computation_revision=='acoustic/2'
    assert AcousticResult.model_validate_json(wire.model_dump_json())==wire
    observed=next(item for item in wire.metadata.backends if item.stage=='reaper')
    assert observed.actual==('reaper_python' if policy=='python_only' else 'native_reaper')
    assert np.isfinite(result.f0_praat).any()
    assert np.isfinite(result.f0_reaper).any()


def test_old_metadata_without_revision_still_reads_as_legacy():
    from ptb_api.acoustic_models import AcousticMetadata
    assert AcousticMetadata.model_fields['computation_revision'].default=='acoustic/1'
