"""Time the existing isolated Praat preview on authorized real recordings."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCES=[str(ROOT/p) for p in ('backend/src','packages/phonetic_core/src')]
sys.path[:0]=SOURCES
os.environ['PYTHONPATH']=os.pathsep.join(SOURCES)
from ptb_worker.spectrogram_preview import render
from ptb_worker.spectrogram_session import SpectrogramSession

def main():
    inventory=json.loads((ROOT/'output/validation/p17/natural-inventory.json').read_text(encoding='utf8'))
    rows=[];session=SpectrogramSession();processes=[]
    try:
        for kind in ('short','medium','egg'):
            source=inventory['selected'][kind]
            raw=Path(source['path']).read_bytes()
            for index in range(3):
                start=min(source['duration']/2,2)+index*.01
                end=min(start+.1,source['duration'])
                view=dict(channel=0,start=start,end=end,width=800)
                t=time.perf_counter();expected=render(raw,**view);original=time.perf_counter()-t
                t=time.perf_counter();result=session.render(raw,**view);elapsed=time.perf_counter()-t
                assert result==expected, 'New process must preserve every byte and scalar'
                if session.process not in processes:processes.append(session.process)
                rows.append(dict(kind=kind,iteration=index,original_seconds=original,seconds=elapsed,
                                 byte_exact=True,width=result['width'],height=result['height']))
        timings=[]
        for i in range(24):
            t=time.perf_counter();session.render(raw,channel=i%2,start=2+i*.01,end=2.1+i*.01,width=800)
            timings.append((time.perf_counter()-t)*1000)
        try:session.render(raw,channel=99,start=0,end=.1,width=800)
        except ValueError:pass
        else:raise AssertionError('Invalid channel must fail')
        assert session.process is None,'A failed session is discarded'
        session.render(raw,channel=0,start=0,end=.1,width=800)
        processes.append(session.process)
        try:session.render(raw,channel=0,start=0,end=10,width=1000,timeout=.000001)
        except ValueError as exc:assert str(exc)=='preview_timeout'
        else:raise AssertionError('Controlled short deadline must time out')
        assert session.process is None,'Timed-out child must stop'
    finally:session.close()
    assert all(p.poll() is not None for p in processes),'All owned children must stop'
    idle=SpectrogramSession(idle_seconds=.15)
    try:
        idle.render(raw,channel=0,start=0,end=.1,width=800)
        process=idle.process
        deadline=time.monotonic()+3
        while (idle.process is not None or process.poll() is None) and time.monotonic()<deadline:time.sleep(.05)
        assert idle.process is None and process.poll() is not None,'Idle child expiry'
    finally:idle.close()
    timings.sort()
    report=dict(parity=rows,warm_ms=timings,median=timings[12],p95=timings[22],maximum=timings[-1],
                lifecycle=['invalid request resets child','recovery','controlled deadline','owned exit','idle expiry'])
    out=ROOT/'output/validation/p17/spectrogram-session.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
