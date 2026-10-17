"""Real Linux Python VTL session, budgets and finite long-render checks; no DB/files."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from ptb_worker.vocal_sessions import VocalSessions,VocalError


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resources',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--long',action='store_true')
    args=parser.parse_args()
    if sys.platform!='linux':raise ValueError('Linux validation only')
    started=time.perf_counter();service=VocalSessions(args.resources);identity=('native-qa','owned-session','public-synthetic');checks=[];cases=[]
    try:
        sid=service.open(identity)['id'];call=lambda op,body={}:service.invoke(sid,identity,op,body)
        meta=call('meta');assert meta['section_count']==33 and meta['section_options']==[33,65]
        assert meta['geometry_method']=='m10/r18';revision=0
        for count in (33,65):
            for name in ('a','i','u'):
                revision+=1;t=time.perf_counter();state=call('pose',dict(params=meta['presets'][name],revision=revision,section_count=count))
                assert len(state['centerline'])==len(state['airway_sections'])==count
                assert np.isfinite(state['tube_areas']).all()
                cases.append(dict(name=name,section_count=count,areas=state['tube_areas'],lengths=state['tube_lengths'],resonances=state['resonances'],seconds=time.perf_counter()-t))
        for count in (129,257):
            try:call('pose',dict(params=meta['presets']['a'],revision=100,section_count=count))
            except VocalError:pass
            else:raise AssertionError('Web sampling bypass')
        checks.append('real 33/65 geometry, a/i/u, server rejects 129/257')
        revision+=1;call('pose',dict(params=meta['presets']['a'],revision=revision,section_count=33))
        try:call('live',{'active':True})
        except VocalError:pass
        else:raise AssertionError('Continuous phonation bypass')
        call('preview',{'duration':1});position=0;peaks=[];pull_times=[]
        try:call('pose',dict(params=meta['presets']['i'],revision=revision+1))
        except VocalError:pass
        else:raise AssertionError('Pose changed during one-second audition')
        while True:
            t=time.perf_counter();packet=call('audio/pull',{'position':position});pull_times.append(time.perf_counter()-t)
            audio=np.frombuffer(base64.b64decode(packet['base64']),'<f4');assert 0<len(audio)<=23040 and np.isfinite(audio).all();peaks.append(float(np.max(np.abs(audio))));position+=len(audio)/48000
            if packet['eof']:break
        assert max(peaks)>0
        assert position==1
        call('audio/monitor',dict(seconds=2,position=position,completed=True));call('animation/stop');checks.append('48kHz one-second PCM, pose lock, continuous rejection, acknowledged monitor, stop')
        frame=lambda name,duration:dict(params=meta['presets'][name],name=name,duration=duration,lip_width=1.,f0=150.)
        frames=[frame('a',.05),{'aspiration':True,'name':'aspiration','duration':.01},{**frame('i',.05),'silent':True}]
        call('keyframes',dict(frames=frames,pitch_curve=[[0.,100.],[1.,200.]]));doc=call('document/create',call('keyframes/load'));assert doc['section_count']==33
        assert call('document/validate',doc)==doc
        prepared=call('animation/prepare',call('keyframes/load'));packet=call('animation/audio',{'id':prepared['id']});assert packet['samples']==5280
        call('animation/picture',dict(id=prepared['id'],index=prepared['count']-1));checks.append('ordinary/aspiration/silent sequence, F0, version5 document, actual audio/picture')
        if args.long:
            long_frames=[frame(name,1.25) for name in ('a','i','u','a')]
            with ThreadPoolExecutor(max_workers=1) as executor:
                future=executor.submit(call,'animation/prepare',dict(frames=long_frames));time.sleep(.15)
                t=time.perf_counter();call('animation/stop');cancel_seconds=time.perf_counter()-t
                try:future.result(timeout=20)
                except VocalError:pass
                else:raise AssertionError('Long render finished before cancellation could be tested')
            checks.append('cancel in-flight five-second render; heartbeat/stop bypass compute slot')
            t=time.perf_counter();long_result=call('animation/prepare',dict(frames=long_frames));long_seconds=time.perf_counter()-t
            packet=call('animation/audio',{'id':long_result['id']});audio=np.frombuffer(base64.b64decode(packet['base64']),'<f4')
            assert len(audio)==240000 and np.isfinite(audio).all() and np.std(audio)>0
            assert long_result['duration']==5 and long_result['times'][-1]==5
            checks.append('complete maximum five-second native animation and 240000 samples')
        child=service.get(sid,identity);limits=Path(f'/proc/{child.process.pid}/limits').read_text();status=Path(f'/proc/{child.process.pid}/status').read_text()
        assert '1073741824' in limits and '600' in limits
        memory={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in status.splitlines() if line.startswith(('VmPeak:','VmHWM:','VmRSS:','Threads:'))}
        scratch=Path(child.scratch.name);service.release(sid,identity);assert child.process.poll() is not None and not scratch.exists();checks.append('hard process limits, exit and owned scratch cleanup')
        report=dict(success=True,scope='actual Linux Python VTL worker, no DB/browser claim',checks=checks,cases=cases,continuous_pull_seconds=pull_times,memory=memory,
            long_render_seconds=long_seconds if args.long else None,cancel_seconds=cancel_seconds if args.long else None,
            native_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.resources/'linux-x86_64').glob('*.so')},elapsed_seconds=time.perf_counter()-started)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n','utf-8');print(json.dumps(dict(success=True,checks=checks,memory=memory,elapsed_seconds=report['elapsed_seconds'])))
    finally:service.close()


if __name__=='__main__':main()
