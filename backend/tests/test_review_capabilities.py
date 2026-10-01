from types import SimpleNamespace
from fastapi.testclient import TestClient
from ptb_api.main import create_app


def test_small_profile_does_not_advertise_rejected_archives(monkeypatch):
    from ptb_worker import resource_profiles
    monkeypatch.setattr(resource_profiles,'selected_profile',lambda:resource_profiles.PROFILES['server-small'])
    store=SimpleNamespace(batches=object(),files=SimpleNamespace(reaper_binary=None))
    with TestClient(create_app(job_store=store,storage=SimpleNamespace(ready=True))) as client:
        operations=client.get('/api/v1/capabilities').json()['task_operations']
    assert 'storage_check' in operations
    assert 'archive_zip' not in operations and 'extract_zip' not in operations
