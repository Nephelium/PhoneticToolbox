"""Reject ambiguous EGG modes before admitting a persistent task."""
import pytest
from pydantic import ValidationError
from ptb_api.egg_models import EggTaskConfig, EggManifest


@pytest.mark.parametrize('change', [
    {'roi_start': .2}, {'roi_start': .3, 'roi_end': .2},
    {'mode': 'batch', 'roi_end': 1.0}, {'mode': 'batch', 'signal_mode': 'raw'},
    {'mode': 'inverse', 'roi_end': 10.0001}, {'lp_order': 0},
    {'highpass_cutoff': 1000.0}, {'spec_vmin': -1.0}, {'flip_channels': 1},
])
def test_reject_inconsistent_settings(change):
    with pytest.raises(ValidationError): EggTaskConfig(**change)


def test_workbench_defaults():
    config = EggTaskConfig()
    assert (config.gci_method, config.goi_method) == ('slope', 'scale')
    assert config.export_policy == 'sample-aligned/1'


def test_preview_fields_do_not_change_historical_mode_idempotency_payload():
    from ptb_api.egg_models import EggRequest
    from ptb_worker.egg_jobs import request_payload
    from ptb_worker.store import LOCAL_PROJECT
    base=dict(project_id=LOCAL_PROJECT,idempotency_key='m03-compatible-hash',audio=dict(asset_id=LOCAL_PROJECT,sha256='a'*64))
    for config in [{},{'mode':'batch'},{'mode':'inverse','roi_end':.1}]:
        payload=request_payload(EggRequest(**base,config=config))
        assert 'micro_center' not in payload['config'] and 'micro_width_ms' not in payload['config']
    assert request_payload(EggRequest(**base,config={'mode':'preview','micro_center':.1}))['config']['micro_center']==.1


def test_partial_manifest_rejected():
    with pytest.raises(ValidationError): EggManifest(core_version='3.0.0a1', files=[])


def test_api_requires_local_identity_origin_and_asset_reference():
    from fastapi.testclient import TestClient
    from ptb_api.main import create_app
    from ptb_worker.store import LOCAL_PROJECT
    origin='http://127.0.0.1:5177';token='x'*40
    body=dict(project_id=LOCAL_PROJECT,idempotency_key='m03-boundary-01',
              audio=dict(asset_id=LOCAL_PROJECT,sha256='a'*64),config={})
    with TestClient(create_app('local',job_store=object(),local_token=token,local_origin=origin),base_url=origin) as client:
        url='/api/v1/jobs/egg/create'
        assert client.post(url,json=body).status_code==403
        assert client.post(url,json=body,headers={'Authorization':'Bearer '+token}).status_code==403
        headers={'Authorization':'Bearer '+token,'Origin':origin}
        assert client.post(url,json=body|dict(owner_id='other'),headers=headers).status_code==422
        assert client.post(url,json=body|dict(audio={'path':'C:/private.wav'}),headers=headers).status_code==422


def test_runtime_missing_or_wrong_build_is_explicit(monkeypatch):
    from ptb_worker.egg_runtime import command
    from ptb_worker.acoustic_errors import AcousticFailure
    from pathlib import Path
    monkeypatch.delenv('PTB_EGG_PYTHON',raising=False)
    with pytest.raises(AcousticFailure,match='egg_runtime_unavailable'):command('request','pipe')
    root=Path(__file__).resolve().parents[2]
    monkeypatch.setenv('PTB_EGG_PYTHON',str(root/'.venv/m03-ui/Scripts/python.exe'))
    with pytest.raises(AcousticFailure,match='egg_runtime_mismatch'):command('request','pipe')


def test_new_long_csv_manifest_extends_only_csv_budget():
    from uuid import uuid4
    files=[dict(id=str(uuid4()),name=name,kind='result',size_bytes=(65000000 if name.endswith('.csv') else 100),sha256='a'*64,expires_at=None)
        for name in ['egg_DATA.csv','egg.ptb.json']]
    assert EggManifest(core_version='3.0.0',files=files,format_revision='m03/2').complete
    with pytest.raises(ValidationError):EggManifest(core_version='3.0.0',files=files)
    files[1]['size_bytes']=65000000
    with pytest.raises(ValidationError):EggManifest(core_version='3.0.0',files=files,format_revision='m03/2')
