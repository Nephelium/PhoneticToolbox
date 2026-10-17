"""Server-only policy checks plus unchanged desktop mode boundaries."""
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from ptb_api.main import create_app
from ptb_worker.m11_task import catalog, submit
from ptb_worker.store import JobError


def test_web_mfa_catalog_never_reads_desktop_models(monkeypatch):
    monkeypatch.setattr('ptb_worker.m11_task.load_registry',lambda:pytest.fail('Web must not load local registry'))
    value=catalog(local=False)
    assert not value['execution_available'] and value['waiting_reason']=='m11_web_disabled'
    assert value['runtimes']==value['models']==[]
    with pytest.raises(JobError,match='m11_web_disabled'):
        submit(SimpleNamespace(postgres=True), 'owner', {})


def test_local_cannot_enable_public_accounts_and_default_server_is_closed():
    with pytest.raises(ValueError,match='server-only'):
        create_app('local',registration_service=object())
    with TestClient(create_app('local')) as client:
        assert client.get('/api/v1/auth/options').status_code==404
        assert client.post('/api/v1/auth/register',json={}).status_code==404
    with TestClient(create_app()) as client:
        assert client.get('/api/v1/auth/options').status_code==503
