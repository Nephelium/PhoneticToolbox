"""P11: queue configuration alone must not advertise a runnable science backend."""
from types import SimpleNamespace

from fastapi.testclient import TestClient

from ptb_api.main import create_app
from ptb_worker.native import capabilities as runtime


def test_linux_batches_without_verified_science_runtime_are_unavailable(monkeypatch):
    monkeypatch.setattr(runtime, 'platform_name', lambda: 'linux')
    store = SimpleNamespace(batches=object(), files=object())
    with TestClient(create_app(job_store=store)) as client:
        result = client.get('/api/v1/capabilities').json()
    assert result['algorithms'] == []
    assert 'acoustic_analysis' not in result['task_operations']
    assert 'textgrid_segment' not in result['task_operations']
    assert any('linux_scientific_runtime_unverified' in item for item in result['limitations'])


def test_windows_missing_native_resource_is_not_available(monkeypatch):
    monkeypatch.setattr(runtime, 'platform_name', lambda: 'win32')
    store = SimpleNamespace(batches=object(), files=SimpleNamespace(reaper_binary=None))
    with TestClient(create_app(job_store=store)) as client:
        result = client.get('/api/v1/capabilities').json()
    assert 'M01' not in result['algorithms']
    assert 'acoustic_analysis' not in result['task_operations']
    assert any('registered_reaper_unavailable' in item for item in result['limitations'])


def test_missing_queue_stays_unconfigured():
    assert runtime.m01_capability(None) == (False, 'acoustic_batches_not_configured')
