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
