"""Compare visual frame IDs against decoded PTS, independently of lip export logic."""
from pathlib import Path
import json
import sys
import av
import numpy as np

root=Path(__file__).resolve().parents[1]
reports=[]
for relative in sys.argv[1:]:
    directory=root/relative
    capture=json.loads((directory/'capture.json').read_text('utf8'))
    decoded={}
    with av.open(str(directory/'source.webm')) as media:
        for frame in media.decode(video=0):
            image=frame.to_ndarray(format='rgb24')
            code=sum(1<<bit for bit in range(8) if image[32,bit*8+4,0]>128)
            decoded.setdefault(code,[]).append(float(frame.pts*frame.time_base))
    samples={round(s['time_ms'],6):s for s in capture['samples']}
    pairs=[]
    for row in capture['frames']:
        sample=samples.get(round(row['media_time_s']*1000,6))
        if sample and len(decoded.get(sample['id'],[]))==1:
            pts=decoded[sample['id']][0]
            pairs.append(dict(frame_id=sample['id'],candidate=row['time_s'],encoded=pts,error_ms=(row['time_s']-pts)*1000))
    assert len(pairs)>=10, 'insufficient unambiguous frame IDs'
    with av.open(str(directory/'source.webm')) as media:
        start=None;position=0;pulse=None;rate=None
        for frame in media.decode(audio=0):
            start=float(frame.pts*frame.time_base) if start is None else start;rate=frame.sample_rate
            values=frame.to_ndarray().ravel();indices=np.flatnonzero(np.abs(values)>.05)
            if pulse is None and len(indices):pulse=start+(position+int(indices[0]))/rate
            position+=frame.samples
    # Canvas event records use the same synthetic AudioContext as the pulse.
    event=next((e for e in capture['events'] if e['audio']>=capture['onset'] and len(decoded.get(e['id'],[]))==1),None)
    delta=None if event is None or pulse is None else (decoded[event['id']][0]-pulse)*1000
    report=dict(source=str(directory.relative_to(root)),config=capture.get('config'),pairs=pairs,
                matched_frames=len(pairs),error_ms_min_median_max=np.percentile([p['error_ms'] for p in pairs],[0,50,100]).tolist(),
                audio_first_decoded_pts_s=start,pulse_decoded_pcm_time_s=pulse,first_visual_event_after_scheduled_pulse_delta_ms=delta,
                limits='Synthetic Canvas/AudioContext clocks, fake inference latency, real MediaRecorder and PyAV decoding. Not physical camera/microphone latency.')
    (directory/'independent-clock-audit.json').write_text(json.dumps(report,indent=2),'utf8');reports.append({k:v for k,v in report.items() if k!='pairs'})
(root/'output/validation/m05-r3/clock-audit.json').write_text(json.dumps(reports,indent=2),'utf8')
print(json.dumps(reports,indent=2))
