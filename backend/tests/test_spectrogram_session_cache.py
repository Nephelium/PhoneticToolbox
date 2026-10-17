"""Real owned Windows process: cache hits skip IPC and preserve exact output."""
from pathlib import Path
import pytest
from ptb_worker.spectrogram_session import SpectrogramSession
from ptb_worker.spectrogram_preview import render, PreviewError

WAV=(Path(__file__).resolve().parents[2]/'frontend/src/assets/SYN-EGG-44100.wav').read_bytes()


def test_same_view_skips_exchange_and_close_releases():
    session=SpectrogramSession()
    try:
        view=dict(channel=0,start=0,end=.8,width=300)
        expected=render(WAV,**view)
        first=session.render(WAV,**view)
        assert first==expected
        original=session.process.stdin
        class Input:
            writes=0
            def write(self,value):self.writes+=1;return original.write(value)
            def flush(self):return original.flush()
            def close(self):return original.close()
        counted=Input();session.process.stdin=counted
        first['pixels_base64']='modified by caller'
        assert session.render(WAV,**view)==expected
        assert counted.writes==0
        assert len(session.cached[1])<=512_000
        assert session.render(WAV,**{**view,'width':320})==render(WAV,**{**view,'width':320})
        assert counted.writes==1
        assert session.render(WAV,**{**view,'channel':1})==render(WAV,**{**view,'channel':1})
        assert counted.writes==2
        changed=bytearray(WAV);changed[-100:]=b'\0'*100
        assert session.render(changed,**view)==render(changed,**view)
        assert counted.writes==4
        with pytest.raises(PreviewError):session.render(b'bad',**view)
        assert session.cached is None and session.process is None
        assert session.render(WAV,**view)==expected
    finally:session.close()
    assert session.cached is None and session.digest is None and session.process is None


def test_dead_worker_and_idle_expiry_drop_cached_response():
    import time
    session=SpectrogramSession()
    try:
        view=dict(channel=0,start=0,end=.8,width=300)
        expected=session.render(WAV,**view)
        dead=session.process;dead.terminate();dead.wait(timeout=5)
        assert session.render(WAV,**view)==expected and session.process is not dead
        session.touched=time.monotonic()-session.idle_seconds-1
        session._expire()
        assert session.cached is None and session.process is None
    finally:session.close()
