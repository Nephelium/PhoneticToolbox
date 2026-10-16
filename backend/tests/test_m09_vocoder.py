"""Optional vocoder keeps legacy tasks stable and never opens web install APIs."""
import json
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ptb_api.spec2wav_models import Spec2WavConfig
from ptb_api.m09_components import component_router
from ptb_worker import vocoder_components as component


def test_default_snapshot_remains_legacy_and_bigvgan_is_explicit():
    old = Spec2WavConfig().snapshot()
    assert 'vocoder' not in old
    assert Spec2WavConfig(vocoder='none').snapshot() == old
    assert Spec2WavConfig(vocoder='bigvgan').snapshot() == old | {'vocoder':'bigvgan'}


def test_original_phase_audio_editing_cannot_enable_vocoder():
    with pytest.raises(ValueError):
        Spec2WavConfig(mode='audio_draw',vocoder='bigvgan')


def test_missing_component_is_explicit_and_status_does_not_install(tmp_path, monkeypatch):
    monkeypatch.setenv('PTB_VOCODER_ROOT',str(tmp_path / 'component'))
    with pytest.raises(ValueError, match='m09_vocoder_missing'):
        component.installed()
    assert component.status()['ready'] is False
    assert not (tmp_path / 'component').exists()


def test_registration_cannot_escape_component_versions(tmp_path,monkeypatch):
    monkeypatch.setenv('PTB_VOCODER_ROOT',str(tmp_path))
    monkeypatch.setattr(component.vocoder_catalog,'RELEASE',{'id':'expected'})
    (tmp_path / 'current.json').write_text(json.dumps(dict(id='expected',path=str(tmp_path.parent))),'utf-8')
    with pytest.raises(ValueError, match='m09_vocoder_invalid'):
        component.installed()


@pytest.mark.parametrize('method,url',[('get','/spec2wav/vocoder'),('post','/spec2wav/vocoder/download')])
def test_web_mode_has_no_component_operation(method,url):
    app=FastAPI();app.include_router(component_router('server',lambda:{'id':'owner'},lambda:{'id':'owner'}))
    with TestClient(app) as client:
        assert getattr(client,method)(url).status_code == 404


def test_download_failure_is_bounded_public_code(monkeypatch):
    def failed():raise RuntimeError('secret-internal-path-must-not-leak')
    monkeypatch.setattr(component,'install',failed)
    app=FastAPI();app.include_router(component_router('local',lambda:{'id':'local'},lambda:{'id':'local'}))
    with TestClient(app) as client:
        response=client.post('/spec2wav/vocoder/download')
        assert response.status_code==422 and response.json()['detail']=='m09_vocoder_download_failed'


def test_ready_component_download_reuses_existing_installation(monkeypatch):
    monkeypatch.setattr(component,'status',lambda:dict(ready=True,pipeline_ready=True))
    def unexpected(*args,**kwargs):raise AssertionError('Must reuse current component')
    monkeypatch.setattr(component,'ComponentManager',unexpected)
    assert component.install()['ready'] is True


def test_component_manager_legacy_defaults_remain_separate_from_model_version(tmp_path):
    from ptb_worker.mfa.components import ComponentManager
    legacy=ComponentManager(tmp_path / 'mfa')
    vocoder=ComponentManager(tmp_path / 'vocoder',schema='ptb-vocoder-component/1',version_field='model_revision',version=component.REVISION,prefix='bigvgan-r1')
    assert legacy.schema=='m11-component/1' and legacy.version_field=='mfa_version' and legacy.version=='3.3.8'
    with pytest.raises(ValueError,match='m11_manifest_invalid'):
        vocoder.import_archive(tmp_path / 'no.zip',dict(schema=legacy.schema,id='candidate'),lambda _:dict(success=True))
