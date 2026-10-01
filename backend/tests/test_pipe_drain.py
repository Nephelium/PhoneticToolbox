"""Ready native output must drain without per-chunk throttling."""
import sys
import pytest
from ptb_worker.native import reaper
from ptb_worker.io.limits import Limits, LimitError, Cancelled

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows named-pipe adapter')


@pytest.fixture
def process(monkeypatch):
    from ptb_worker.native import windows
    class Process:
        pid = 123
        closed = False
        instances = []
        def __init__(self, *args): self.instances.append(self)
        def poll(self): return 0
        def close(self): self.closed = True
    monkeypatch.setattr(windows, 'OwnedProcess', Process)
    return Process


class Pipe:
    def __init__(self, chunks): self.chunks = iter(chunks); self.closed = False
    def read(self, size, owner): return next(self.chunks, b''), False
    def close(self): self.closed = True


def test_ready_output_does_not_sleep_between_chunks(process, monkeypatch):
    sleeps = []
    monkeypatch.setattr(reaper.time, 'sleep', sleeps.append)
    pipe = Pipe([b'a'*4096]*1024)
    data, _ = reaper.collect_pipe([], pipe, '.', Limits(output_bytes=5_000_000))
    assert data == b'a'*(4096*1024)
    assert sleeps == []
    assert pipe.closed
    assert process.instances[-1].closed


def test_drain_still_checks_cancel_every_chunk(process):
    pipe = Pipe([b'x']*100)
    chunks = []
    with pytest.raises(Cancelled):
        reaper.collect_pipe([], pipe, '.', Limits(), stop=lambda: len(chunks)>=3, on_chunk=chunks.append)
    assert chunks == [b'x']*3 and pipe.closed


def test_drain_still_rejects_output_over_budget(process):
    pipe = Pipe([b'ab', b'cd'])
    with pytest.raises(LimitError):
        reaper.collect_pipe([], pipe, '.', Limits(output_bytes=3))
    assert pipe.closed


def test_idle_pipe_waits_and_timeout_still_closes_process(process, monkeypatch):
    monkeypatch.setattr(process, 'poll', lambda self: None)
    clock = [0.]
    sleeps = []
    def sleep(seconds): sleeps.append(seconds); clock[0] += seconds
    monkeypatch.setattr(reaper.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(reaper.time, 'sleep', sleep)
    pipe = Pipe([])
    with pytest.raises(LimitError, match='native_timeout'):
        reaper.collect_pipe([], pipe, '.', Limits(timeout_seconds=.01))
    assert sleeps and pipe.closed and process.instances[-1].closed
