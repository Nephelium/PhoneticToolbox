"""M01-F1 actual synthetic artifacts; no persistent task schema or UI writes."""
from pathlib import Path
import hashlib
import io
import json
import sqlite3
import sys
import time
from uuid import uuid4
import numpy as np
from scipy.io import wavfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests/support'))
from baseline_support import RECIPES,create_fixture
from phonetic_core.models.associations import AcousticAssociations
from phonetic_core.services.acoustic import analyze_audio
from ptb_api.acoustic_models import AcousticRequest,AcousticInputSnapshot
from ptb_worker.acoustic_result import build_acoustic_result,to_core_config
from ptb_worker.io.audio import decode_wav
from ptb_worker.io.annotations import decode_textgrid
from ptb_worker.io.parameter_exports import ExportPair,verify_pair
from ptb_worker.io.scratch import Scratch
from ptb_worker.segmentation import prepare_segments,SEGMENT_LIMITS


def uid(n):return f'00000000-0000-4000-8000-{n:012d}'
def sha(raw):return hashlib.sha256(raw).hexdigest()


def grid(duration,intervals):
    parts=['File type = "ooTextFile short"','"TextGrid"','0',str(duration),'<exists>','1',
           '"IntervalTier"','"音节"','0',str(duration),str(len(intervals))]
    for start,end,label in intervals:parts.extend([str(start),str(end),'"'+label.replace('"','""')+'"'])
    return '\n'.join(parts).encode('utf-8')


def persist_bundle(target,bundle):
    for entry,raw in zip(bundle.manifest['files'],bundle.payloads):
        path=target/entry['name']
        with path.open('xb') as output:output.write(raw)
        assert sha(path.read_bytes())==entry['sha256']
    (target/'manifest.json').write_text(json.dumps(bundle.manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    folder=ROOT/'output/validation/m01'/('segments-'+uuid4().hex);folder.mkdir(parents=True)
    target=folder/'actual-parent';target.mkdir()
    source=create_fixture(target,RECIPES[0]);audio_raw=source.read_bytes()
    grid_raw=grid(.8,[(0,.203,'ɑ̃'),(.203,.401,'sil'),(.401,.8,'=literal')])
    (target/'labels.TextGrid').write_bytes(grid_raw)
    audio=decode_wav(audio_raw)
    snapshots=[AcousticInputSnapshot(asset_id=uid(2+i),sha256=sha(raw),role=role,expires_at=None)
               for i,(role,raw) in enumerate((('audio',audio_raw),('textgrid',grid_raw)))]
    request=AcousticRequest(project_id=uid(1),idempotency_key='m01-f1-artifact',
        inputs={s.role:dict(asset_id=s.asset_id,sha256=s.sha256) for s in snapshots},
        config=dict(selection=dict(keys=['pF0']),backend_policy=dict(reaper='disabled')))
    # Tiny new analytic fixture only; full user science isolation/publication remains F2.
    result=analyze_audio(audio,to_core_config(request.config),AcousticAssociations(tiers=decode_textgrid(grid_raw)))
    wire=build_acoustic_result(result,audio,request,snapshots)
    parent_raw=wire.model_dump_json().encode();(target/'parent-result.json').write_bytes(parent_raw)
    beats=[];started=time.monotonic()
    with Scratch(target,10_000_000) as scratch:
        bundle=prepare_segments(audio_raw,grid_raw,'音节',scratch,audio_name=source.name,parent_result=parent_raw,
                                heartbeat=lambda:beats.append(time.monotonic()))
        assert scratch.used==0
    persist_bundle(target,bundle)
    frame=result.to_dataframe().rename(columns={c.key:c.label for c in wire.numeric})
    segments=[]
    for item in bundle.manifest['segments']:
        files=[(f,b) for f,b in zip(bundle.manifest['files'],bundle.payloads) if f['segment_index']==item['interval_index']]
        fs,values=wavfile.read(io.BytesIO(files[0][1]))
        np.testing.assert_array_equal(values,audio.samples[item['first_sample']:item['last_sample']])
        assert fs==audio.sample_rate_hz
        # Independent expected rows from original frame, without the slicing implementation.
        first,last=item['first_sample']/fs,item['last_sample']/fs
        selected=frame[(frame['Time_s']>=first)&(frame['Time_s']<last)]
        cols=['Time_s','Source_Time_s',*list(frame.columns[1:])]
        rows=[]
        for row in selected.itertuples(index=False,name=None):
            rows.append([row[0]-first,row[0],*[v if isinstance(v,str) or np.isfinite(v) else None for v in row[1:]]])
        table=dict(columns=cols,kinds=['text' if c.startswith('text_') else 'number' for c in cols],rows=rows)
        pair=ExportPair(files[1][1],files[2][1],tuple(cols),len(rows))
        verify_pair(pair,table)
        conn=sqlite3.connect((target/files[2][0]['name']).as_uri()+'?mode=ro',uri=True)
        try:assert conn.execute('SELECT COUNT(*) FROM params').fetchone()[0]==len(rows)
        finally:conn.close()
        segments.append(dict(first_sample=item['first_sample'],last_sample=item['last_sample'],rows=len(rows),
                             first_local_time=rows[0][0],first_source_time=rows[0][1],files=[f for f,_ in files]))
    assert sha(source.read_bytes())==snapshots[0].sha256
    cases=[dict(case='actual-praat-parent',source_frames=len(frame),segments=segments,
                elapsed_seconds=time.monotonic()-started,heartbeat_calls=len(beats),wave_samples_equal=True,
                both_parameter_formats_equal=True,parent_sha256=sha(parent_raw),backends=[b.model_dump() for b in wire.metadata.backends])]
    for name,fs,samples,spans in (
        ('float32-stereo',8000,np.column_stack((np.linspace(-.75,.75,8000,dtype=np.float32),np.zeros(8000,dtype=np.float32))),[(.125,.875,'float')]),
        ('600-seconds',8000,(np.arange(4_800_000,dtype=np.int32)%30000).astype(np.int16),[(0,.125,'start'),(599.875,600,'end')])):
        case=folder/name;case.mkdir();stream=io.BytesIO();wavfile.write(stream,fs,samples);raw=stream.getvalue()
        (case/'input.wav').write_bytes(raw);tg=grid(len(samples)/fs,spans);(case/'labels.TextGrid').write_bytes(tg)
        start=time.monotonic()
        with Scratch(case,16_000_000) as scratch:
            bundle=prepare_segments(raw,tg,'音节',scratch,audio_name=name+'.wav')
            assert scratch.used==0
        persist_bundle(case,bundle)
        for item,blob in zip(bundle.manifest['segments'],bundle.payloads):
            actual_fs,actual=wavfile.read(io.BytesIO(blob))
            assert actual_fs==fs and actual.dtype==samples.dtype
            np.testing.assert_array_equal(actual,samples[item['first_sample']:item['last_sample']])
        cases.append(dict(case=name,duration_s=len(samples)/fs,input_bytes=len(raw),sample_dtype=str(samples.dtype),
                          elapsed_seconds=time.monotonic()-start,samples_equal=True,files=bundle.manifest['files']))
    report=dict(task='M01-F1',platform='Windows',python=sys.version.split()[0],cases=cases,
                scope='Owned preparation and new synthetic artifact readback only; persistent jobs/publication pending F2',
                limits=vars(SEGMENT_LIMITS))
    path=folder/'report.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(path.relative_to(ROOT)),cases=len(cases)),ensure_ascii=False))


if __name__=='__main__':main()
