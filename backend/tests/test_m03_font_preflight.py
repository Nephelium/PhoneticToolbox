"""M03/P04-FONT: authenticated inspection in the actual bounded render runtime."""
from pathlib import Path
import pytest
from ptb_api.font_models import FigureFontSnapshot
from ptb_worker.font_preflight import inspect_fonts,_slot
from ptb_worker.spectrogram_preview import PreviewError

@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setenv('PTB_EGG_PYTHON',str(Path(__file__).resolve().parents[2]/'.venv/m03-compatible/python.exe'))

def test_real_runtime_reports_roles_and_no_paths(runtime):
    result=inspect_fonts(FigureFontSnapshot())
    assert result['available']
    assert [f['role'] for f in result['fonts']]==['zh','latin','ipa']
    assert result['fonts'][2]['family']=='Doulos SIL'
    assert all(len(f['sha256'])==64 and set(f)=={'role','requested','available','family','sha256'} for f in result['fonts'])
    missing=inspect_fonts(FigureFontSnapshot(zh='PTB missing Chinese 987654',latin='PTB missing Latin 987654'))
    assert not missing['available']
    assert [f['available'] for f in missing['fonts']]==[False,False,True]
    assert missing['fonts'][0]['family'] is None

def test_busy_fails_immediately_and_slot_recovers(runtime):
    _slot.acquire()
    try:
        with pytest.raises(PreviewError,match='font_preflight_busy'):inspect_fonts({})
    finally:_slot.release()
    assert inspect_fonts({})['available']

def test_timeout_reaps_child_and_releases_slot(runtime):
    with pytest.raises(PreviewError,match='font_preflight_timeout'):inspect_fonts({},timeout=.001)
    assert inspect_fonts({})['available']

def test_missing_runtime_releases_slot(monkeypatch,runtime):
    configured=__import__('os').environ['PTB_EGG_PYTHON']
    monkeypatch.delenv('PTB_EGG_PYTHON')
    with pytest.raises(PreviewError,match='egg_runtime_unavailable'):inspect_fonts({})
    monkeypatch.setenv('PTB_EGG_PYTHON',configured)
    assert inspect_fonts({})['available']

def test_api_auth_and_snapshot_boundaries(monkeypatch):
    from fastapi.testclient import TestClient
    from ptb_api.main import create_app
    calls=[]
    result=dict(available=True,fonts=[dict(role=r,requested=n,available=True,family=n,sha256='a'*64) for r,n in [('zh','SimSun'),('latin','Arial'),('ipa','Doulos SIL')]])
    monkeypatch.setattr('ptb_worker.font_preflight.inspect_fonts',lambda font:(calls.append(font) or result))
    origin='http://127.0.0.1:5177';token='x'*40;url='/api/v1/jobs/egg/fonts'
    with TestClient(create_app('local',job_store=object(),local_token=token,local_origin=origin),base_url=origin) as client:
        assert client.post(url,json={}).status_code==403
        assert client.post(url,json={},headers={'Authorization':'Bearer '+token}).status_code==403
        headers={'Authorization':'Bearer '+token,'Origin':origin}
        for body in [dict(zh='C:/font.ttf'),dict(ipa='Arial'),dict(size_px=25),dict(owner_id='other')]:
            assert client.post(url,json=body,headers=headers).status_code==422
        assert not calls
        assert client.post(url,json={},headers=headers).json()==result
        assert len(calls)==1

def test_server_session_csrf_and_owner_switch(runtime):
    # Exercise real HTTP auth and real font child; only account lookup is in-memory.
    from fastapi.testclient import TestClient
    from ptb_api.main import create_app
    from ptb_api.auth import AuthSettings
    from ptb_api.account_store import hash_token
    class Accounts:
        def session(self,token):
            return {hash_token('a'*40):dict(id='owner-a',csrf_token='csrf-a'),
                    hash_token('b'*40):dict(id='owner-b',csrf_token='csrf-b')}.get(token)
    origin='http://127.0.0.1:5177';url='/api/v1/jobs/egg/fonts'
    settings=AuthSettings(origin,'x'*40,allow_insecure_loopback=True)
    with TestClient(create_app('server',account_store=Accounts(),auth_settings=settings,job_store=object()),base_url=origin) as client:
        headers={'Origin':origin,'X-PTB-Account':'owner-a','X-CSRF-Token':'csrf-a'}
        assert client.post(url,json={},headers=headers).status_code==401
        client.cookies.set(settings.cookie,'a'*40)
        assert client.post(url,json={},headers=headers|{'X-CSRF-Token':'wrong'}).status_code==403
        assert client.post(url,json={},headers=headers|{'Origin':'https://other.invalid'}).status_code==403
        assert client.post(url,json={'latin':'PTB missing owner A'},headers=headers).json()['available'] is False
        client.cookies.set(settings.cookie,'b'*40)
        assert client.post(url,json={},headers=headers).status_code==409
        response=client.post(url,json={},headers=headers|{'X-PTB-Account':'owner-b','X-CSRF-Token':'csrf-b'})
        assert response.status_code==200 and response.json()['available'] is True
        assert client.post(url,content=b'x'*17000,headers=headers).status_code==413
