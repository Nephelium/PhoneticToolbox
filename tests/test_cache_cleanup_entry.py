"""Uninstall must not initialize scientific runtimes, including its API child."""
from pathlib import Path
import importlib.util
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cache_cleanup_entry',ROOT/'scripts/v3_local_preview_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)

@pytest.mark.parametrize('complete,code',[(True,0),(False,21)])
def test_clear_dispatch_skips_scientific_bindings(monkeypatch,complete,code):
    from ptb_desktop import cache_cleanup
    modes=[]
    monkeypatch.setattr(entry,'configure',lambda *,runtime:modes.append(runtime) or ROOT)
    monkeypatch.setattr(cache_cleanup,'result_caches',lambda:{'complete':complete})
    monkeypatch.setattr(sys,'argv',['entry','--ptb-clear-user-caches'])
    assert entry.main()==code
    assert modes==[False]

@pytest.mark.parametrize('argument,runtime',[
    ('--local-cache-cleanup-service',False),('--local-service',True)])
def test_cleanup_service_only_skips_bindings_for_maintenance(monkeypatch,argument,runtime):
    import research_entry
    modes=[];calls=[]
    monkeypatch.setattr(entry,'configure',lambda *,runtime:modes.append(runtime) or ROOT)
    monkeypatch.setattr(research_entry,'main',lambda:calls.append(sys.argv[:]) or 0)
    monkeypatch.setattr(sys,'argv',['entry',argument,'--mode','local'])
    assert entry.main()==0
    assert modes==[runtime]
    assert calls==[['entry','--local-service','--mode','local']]

def test_cleanup_service_clears_owned_workspace_and_denies_other_routes(tmp_path):
    from urllib.error import HTTPError
    from ptb_worker.local_workspace import prepare_workspace
    from ptb_desktop.local_service import LocalService
    database,files=prepare_workspace(tmp_path/'owned-workspace',ROOT/'backend/migrations')
    with LocalService(database,local_files_root=files,cache_cleanup=True) as service:
        assert service.request('/api/v1/jobs/local-storage/clear-all','POST')['complete'] is True
        with pytest.raises(HTTPError) as error:
            service.request('/api/v1/jobs/local-storage')
        assert error.value.code==403
