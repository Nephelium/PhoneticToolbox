from types import SimpleNamespace
import numpy as np
from phonetic_core.vocal_tract.monitor import OutputHistory
from phonetic_core.vocal_tract import session


def test_unchanged_output_reuses_analysis_but_generation_and_view_invalidate(monkeypatch):
    history=OutputHistory(48000);history.append(np.arange(240,dtype='float32')/1000)
    live=SimpleNamespace(output_history=history,heartbeat=0)
    runtime=session.Runtime(SimpleNamespace(section_count=33,sr=48000),None,live)
    calls=[];original=session.analyze_output
    def analyze(*args,**kwargs):calls.append(kwargs);return original(*args,**kwargs)
    monkeypatch.setattr(session,'analyze_output',analyze)
    first=runtime.invoke('audio/monitor');second=runtime.invoke('audio/monitor');assert first==second and len(calls)==1
    second['waveform'].clear();assert runtime.invoke('audio/monitor')==first
    runtime.invoke('audio/monitor',{'window_ms':10});assert len(calls)==2
    history.append(np.zeros(240,dtype='float32'));runtime.invoke('audio/monitor',{'window_ms':10});assert len(calls)==3
    history.reset();runtime.invoke('audio/monitor',{'window_ms':10});assert len(calls)==4
