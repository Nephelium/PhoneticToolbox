import io
import json
import sys
from types import SimpleNamespace


def test_worker_startup_failure_returns_protocol_error_and_logs_traceback(monkeypatch,capsys):
    from ptb_desktop.vocal_tract import worker
    def fail(*args,**kwargs):
        raise RuntimeError('native_platform_unavailable')
    monkeypatch.setattr(worker,'watch_parent',lambda pid:None)
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(dict(parent_pid=123,resources='fixture',profile='fixture'))+'\n'))
    monkeypatch.setitem(sys.modules,'ptb_desktop.vocal_tract.runtime',SimpleNamespace(Runtime=fail))
    assert worker.main()==1
    captured=capsys.readouterr()
    assert json.loads(captured.out)==dict(ready=None,error='native_platform_unavailable')
    assert 'RuntimeError: native_platform_unavailable' in captured.err
