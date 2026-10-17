import json
from types import SimpleNamespace
import pytest
from ptb_worker import spec2wav_preview
from ptb_worker.spectrogram_preview import PreviewError
from ptb_worker.io.limits import LimitError

def test_linux_preview_uses_bounded_fixed_child_and_keeps_error_code(monkeypatch,tmp_path):
    from ptb_worker.native import linux_runtime,posix
    seen={}
    monkeypatch.setattr(spec2wav_preview.sys,'platform','linux')
    monkeypatch.setattr(linux_runtime,'load_profile',lambda:(tmp_path/'profile.json',{'cache':str(tmp_path)}))
    monkeypatch.setattr(linux_runtime,'command',lambda entry,arg:[entry,arg])
    def run(argv,payload,cwd,limits):
        seen.update(argv=argv,payload=payload,cwd=cwd,limits=limits)
        return b'123\n'+json.dumps({'prediction':{'win_length_ms':10}}).encode()
    monkeypatch.setattr(posix,'run_bounded',run)
    assert spec2wav_preview.render(b'image',{'auto_parameters':True})['prediction']['win_length_ms']==10
    header,raw=seen['payload'].split(b'\n',1);assert raw==b'image' and json.loads(header)['config']['auto_parameters']
    assert seen['argv']==['spec2wav_preview','--preview'] and seen['limits'].process_bytes==1_000_000_000
    monkeypatch.setattr(posix,'run_bounded',lambda *a,**k:b'123\n{"error":"m09_parameter_model_invalid"}')
    with pytest.raises(PreviewError,match='m09_parameter_model_invalid'):spec2wav_preview.render(b'image',{})
    def timeout(*a,**k):raise LimitError('native_timeout')
    monkeypatch.setattr(posix,'run_bounded',timeout)
    with pytest.raises(PreviewError,match='preview_timeout'):spec2wav_preview.render(b'image',{})

def test_unverified_linux_reconstruction_is_rejected_before_queue(monkeypatch):
    from ptb_worker import spec2wav_jobs
    from ptb_worker.native import capabilities
    from ptb_worker.store import JobError
    from uuid import uuid4
    monkeypatch.setattr(spec2wav_jobs.sys,'platform','linux')
    monkeypatch.setattr(capabilities,'linux_capabilities',lambda store:([],[]))
    body=dict(project_id=str(uuid4()),idempotency_key=uuid4().hex,image=dict(asset_id=str(uuid4()),sha256='0'*64),config={})
    with pytest.raises(JobError,match='m09_runtime_unavailable'):spec2wav_jobs.submit(SimpleNamespace(files=object()),'owned',body)
    monkeypatch.setattr(capabilities,'linux_capabilities',lambda store:(['spectrogram_to_audio'],[]))
    body['config']={'algorithm':'neural_pipeline'}
    with pytest.raises(JobError,match='m09_vocoder_missing'):spec2wav_jobs.submit(SimpleNamespace(files=object()),'owned',body)
