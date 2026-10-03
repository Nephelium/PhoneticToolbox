"""Host-side compatibility and owned native resources (no science in API)."""
from pathlib import Path
import pytest
from ptb_api.egg_models import EggRequest
from ptb_worker.egg_jobs import request_payload
from ptb_worker.store import LOCAL_PROJECT
from ptb_worker.egg_interactive import InteractivePreview

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(r'C:\Users\13680\Desktop\project\音频数据\EGG测试\3.wav')
CONFIG=dict(mode='preview',roi_start=40,roi_end=40.5,micro_center=40.25,
    f0_policy='audio-f0/2',keep_praat_f0=True,keep_reaper_f0=True,keep_gci_f0=False)


def test_legacy_idempotency_omits_additive_defaults():
    for mode in ['single','batch','preview','inverse']:
        request=EggRequest(project_id=LOCAL_PROJECT,idempotency_key='m03-r5-old',audio=dict(asset_id=LOCAL_PROJECT,sha256='a'*64),config=dict(mode=mode,**({'roi_end':.5} if mode=='inverse' else {})))
        c=request_payload(request)['config'];assert 'f0_policy' not in c and 'keep_reaper_f0' not in c
        request.config.f0_policy='audio-f0/2'
        assert request_payload(request)['config']['f0_policy']=='audio-f0/2'


def test_native_preview_caches_and_releases_owned_slot(monkeypatch):
    monkeypatch.setenv('PTB_EGG_PYTHON',str(ROOT/'.venv/m03-compatible/python.exe'))
    manager=InteractivePreview(reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe')
    try:
        token=manager.open('owned',SOURCE.read_bytes())['session_id'];scratch=manager.native_scratch.root;process=manager.process
        one=manager.update('owned',token,CONFIG)
        assert any(v and v>0 for v in one['preview']['reaper']['values'])
        assert all(p.stat().st_size==0 for p in scratch.iterdir())
        two=manager.update('owned',token,CONFIG|dict(micro_center=40.3))
        assert one['preview']['reaper']==two['preview']['reaper']
        assert manager.process is process
    finally:manager.close()
    assert not scratch.exists() and process.poll() is not None


def test_bad_native_resource_does_not_break_unselected_preview(monkeypatch,tmp_path):
    monkeypatch.setenv('PTB_EGG_PYTHON',str(ROOT/'.venv/m03-compatible/python.exe'))
    manager=InteractivePreview(reaper_binary=tmp_path/'missing.exe')
    try:
        token=manager.open('owned',SOURCE.read_bytes())['session_id']
        manager.update('owned',token,CONFIG|dict(keep_reaper_f0=False))
        with pytest.raises(ValueError,match='egg_reaper_unavailable'):manager.update('owned',token,CONFIG)
        assert manager.update('owned',token,CONFIG|dict(keep_reaper_f0=False))['preview']['praat']['times']
    finally:manager.close()
