"""A source bootstrap must not combine new adapters with an old installed core."""
import sys
import pytest
from ptb_worker.source_runtime import use_matching_core_source


def test_source_and_frozen_snapshot_use_their_matching_core(tmp_path,monkeypatch):
    bootstrap=tmp_path/'backend/src/ptb_worker/egg_bootstrap.py'
    core=tmp_path/'packages/phonetic_core/src'
    (core/'phonetic_core').mkdir(parents=True)
    (core/'phonetic_core/__init__.py').write_text('',encoding='utf-8')
    monkeypatch.setattr(sys,'path',['installed-environment'])
    use_matching_core_source(bootstrap)
    assert sys.path==[str(core),'installed-environment']


def test_missing_matching_source_is_explicit_and_wheel_uses_its_environment(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'path',['installed-environment'])
    with pytest.raises(RuntimeError,match='Matching core'):
        use_matching_core_source(tmp_path/'backend/src/ptb_worker/lpc_bootstrap.py')
    use_matching_core_source(tmp_path/'site-packages/ptb_worker/lpc_bootstrap.py')
    assert sys.path==['installed-environment']


def test_core_version_is_from_snapshot_only_when_that_sibling_is_loaded(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from ptb_worker.source_runtime import bound_core_version
    sibling=tmp_path/'packages/phonetic_core/src/phonetic_core/__init__.py'
    sibling.parent.mkdir(parents=True);sibling.write_text('',encoding='utf8')
    bootstrap=tmp_path/'backend/src/ptb_worker/egg_runtime.py'
    monkeypatch.setitem(sys.modules,'phonetic_core',SimpleNamespace(__file__=str(sibling)))
    assert bound_core_version(bootstrap,'3.0.0a1','3.0.3')=='3.0.3'
    assert bound_core_version(tmp_path/'site-packages/ptb_worker/egg_runtime.py','3.0.0a1','3.0.3')=='3.0.0a1'
    monkeypatch.setitem(sys.modules,'phonetic_core',SimpleNamespace(__file__=str(tmp_path/'unrelated/phonetic_core/__init__.py')))
    assert bound_core_version(bootstrap,'3.0.0a1','3.0.3')=='3.0.0a1'
