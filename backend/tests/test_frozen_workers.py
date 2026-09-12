import sys
import pytest
from ptb_worker.process_entry import command, dispatch, MODULES


@pytest.mark.parametrize('module', sorted(MODULES))
def test_frozen_worker_never_invokes_python_flags(monkeypatch, module):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', 'owned-app.exe')
    assert command(module, 'input with spaces', 'pipe') == ['owned-app.exe', '--ptb-worker', module, 'input with spaces', 'pipe']


def test_development_worker_preserves_module_protocol(monkeypatch):
    monkeypatch.delattr(sys, 'frozen', raising=False)
    assert command('ptb_worker.cli')[1:] == ['-B', '-m', 'ptb_worker.cli']


@pytest.mark.parametrize('module', ['os', 'runpy', '', 'ptb_desktop.host', '../../evil'])
def test_unknown_worker_rejected_before_running_code(module):
    with pytest.raises(ValueError):
        command(module)
    with pytest.raises(ValueError):
        dispatch(module, [])
