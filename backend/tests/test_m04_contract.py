"""M04 admission and complete result contracts (A06/A11/A17/A20)."""
import pytest
from pydantic import ValidationError


@pytest.mark.parametrize('change', [{'order':0},{'order':201},{'order':True},
    {'roi_start':.3,'roi_end':.2},{'roi_end':None},{'amp_min_db':35.},
    {'freq_max_hz':float('nan')},{'font':None},{'dynamic_y':1}])
def test_bad_config(change):
    from ptb_api.lpc_models import LpcTaskConfig
    with pytest.raises(ValidationError):LpcTaskConfig(**({'roi_end':.1}|change))


def test_complete_manifest():
    from ptb_api.lpc_models import LpcManifest
    with pytest.raises(ValidationError):LpcManifest(core_version='3.0.0a1',files=[])


def test_reference_and_identity_boundary():
    from fastapi.testclient import TestClient
    from ptb_api.main import create_app
    from ptb_worker.store import LOCAL_PROJECT
    origin='http://127.0.0.1:5177';token='x'*40
    body=dict(project_id=LOCAL_PROJECT,idempotency_key='m04-boundary-01',
              audio=dict(asset_id=LOCAL_PROJECT,sha256='a'*64),config={'roi_end':.1})
    with TestClient(create_app('local',job_store=object(),local_token=token,local_origin=origin),base_url=origin) as client:
        url='/api/v1/jobs/lpc/create'
        assert client.post(url,json=body).status_code==403
        headers={'Authorization':'Bearer '+token,'Origin':origin}
        assert client.post(url,json=body|dict(owner_id='other'),headers=headers).status_code==422
        assert client.post(url,json=body|dict(audio={'path':'C:/private.wav'}),headers=headers).status_code==422
