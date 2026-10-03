"""Read-only natural input: display parity and scientific sensitivity audit.

No threshold is declared optimal without labelled events / source truth.
"""
from pathlib import Path
import base64
import contextlib
import hashlib
import io
import json
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend/src'),str(ROOT/'packages/phonetic_core/src')]
SOURCE=Path(r'C:\Users\13680\Desktop\project\音频数据\EGG测试\3.wav')


def main():
    import numpy as np
    from scipy.io import wavfile
    from phonetic_core.egg import EGGConfig,prepare,analyze_events,events_segment
    from phonetic_core.egg.metrics import calculate_cq_sq
    from phonetic_core.egg.inverse import inverse_filter
    from ptb_worker.egg_interactive_child import Session
    from ptb_worker import egg_preview
    raw=SOURCE.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    out=ROOT/'output/m03-r3';out.mkdir(parents=True,exist_ok=True)
    old=types.ModuleType('ptb_worker.r3_reference')
    exec((out/'before/backend__src__ptb_worker__egg_preview.py').read_text('utf-8'),old.__dict__)
    current=egg_preview.preview_files
    a=Session(raw);b=Session(raw)
    config=dict(mode='preview',roi_start=40,roi_end=40.5,micro_center=40.25,
                keep_praat_f0=False,keep_gci_f0=False,lowpass_cutoff=2000)
    parity=[]
    for change in [{},{'micro_width_ms':100},{'spec_vmin':-130,'spec_vmax':-50},
                   {'roi_start':20,'roi_end':20.5,'micro_center':20.25},
                   {'signal_mode':'raw'},{'flip_channels':True}]:
        config.update(change);new=a.update(config)
        egg_preview.preview_files=old.preview_files
        try:reference=b.update(config)
        finally:egg_preview.preview_files=current
        new['preview'].pop('suggested_db_range',None);reference['preview'].pop('suggested_db_range',None)
        assert new==reference,change
        parity.append(change)
    fs,samples=wavfile.read(io.BytesIO(raw));base=EGGConfig.for_workbench(lowpass_cutoff=2000)
    result=prepare(samples,fs,base)
    thresholds=[]
    for start in [20.,40.,55.]:
        for auto in [True,False]:
            for threshold in [.003,.005,.01,.02,.03,.05,.08,.1]:
                cfg=EGGConfig.for_workbench(lowpass_cutoff=2000,auto_prominence=auto,
                    peak_prominence=threshold,valley_prominence=threshold)
                gci,goi,peaks=events_segment(result,start,start+.5,cfg)
                # The legacy event helper includes 50 ms padding; UI crops it.
                gci,goi,peaks=([v for v in values if start<=v<start+.5] for values in (gci,goi,peaks))
                t,cq,sq=calculate_cq_sq(gci,goi,peaks)
                delta=np.diff(gci);f0=1/delta if len(delta) else []
                finite=lambda v:float(np.nanmedian(v)) if v is not None and np.any(np.isfinite(v)) else None
                thresholds.append(dict(start=start,auto=auto,threshold=threshold,
                    gci=len(gci),goi=len(goi),peaks=len(peaks),median_f0=finite(f0),
                    median_cq=finite(cq),median_sq=finite(sq),
                    periods_above_500hz=int(np.count_nonzero(np.asarray(f0)>500))))
    result=analyze_events(result,base)
    start,end=40.,40.5;audio=result.audio_signal[int(start*fs):int(end*fs)]
    gci=np.asarray(result.gci_times);gci=gci[(gci>=start)&(gci<end)]-start
    goi=np.asarray(result.goi_times);goi=goi[(goi>=start)&(goi<end)]-start
    inverse=[];outputs={}
    for order in [20,30,40,50,60,80]:
        with contextlib.redirect_stdout(io.StringIO()):values=inverse_filter(audio,fs,gci,lp_order=order)
        outputs[order]=values
    ref=outputs[50]
    for order,values in outputs.items():
        inverse.append(dict(order=order,rms=float(np.sqrt(np.mean(values**2))),
            peak=float(np.max(np.abs(values))),correlation_with_50=float(np.corrcoef(ref,values)[0,1])))
    crossings=sum(1 for event in gci if any((goi>event)&(goi<event+.003)))
    report=dict(input_sha256=sha,parity=parity,thresholds=thresholds,inverse=inverse,
        fixed_window=dict(samples=int(fs*.003),default_order=int(fs/1000)+6,gci_count=len(gci),
            crosses_detected_goi=crossings,crosses_next_gci=int(np.count_nonzero(np.diff(gci)<.003))),
        qualification='One natural recording; no labelled events or glottal flow truth. Sensitivity is not accuracy.')
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==sha
    (out/'natural-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='thresholds'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
