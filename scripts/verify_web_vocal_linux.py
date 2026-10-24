"""Owned VTL session: fixed half-second web audition and existing safety limits."""
from pathlib import Path
import base64,hashlib,json,sys
import numpy as np
from ptb_worker.vocal_sessions import VocalSessions,VocalError
from ptb_worker.native.linux_runtime import load_profile

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);profile,_=load_profile();service=VocalSessions(Path(sys.argv[2]));identity=('synthetic-vocal','test-session','owned-project');checks=[]
    try:
        sid=service.open(identity)['id'];call=lambda op,body={}:service.invoke(sid,identity,op,body)
        meta=call('meta');assert meta['preview_duration']==.5 and meta['continuous_audio'] is False
        call('pose',dict(params=meta['presets']['a'],revision=1));started=call('preview',{'duration':.5});assert started['active'];checks.append('half_second_start')
        try:call('pose',dict(params=meta['presets']['i'],revision=2));raise AssertionError('pose unlocked')
        except VocalError as e:assert '等待测试发声' in str(e)
        parts=[];position=0
        while True:
            packet=call('audio/pull',dict(position=position));audio=np.frombuffer(base64.b64decode(packet['base64']),dtype='<f4');assert len(audio)==packet['samples'];assert packet['sample_rate']==48000;parts.append(audio);position+=len(audio)/48000
            if packet['eof']:break
        pcm=np.concatenate(parts);assert len(pcm)==24000 and np.isfinite(pcm).all() and np.max(np.abs(pcm))>0
        assert not call('heartbeat',dict(position=.5,completed=True))['active'];call('pose',dict(params=meta['presets']['i'],revision=2));checks.append('24000_samples_pose_unlock')
        video=call('video/begin',{'duration':60,'width':1280,'height':720,'fps':30});call('video/cancel',{'id':video['id']});checks.append('video_sixty_seconds_accepted')
        for op,body in [('preview',{'duration':1}),('live',{'active':True}),('video/begin',{'duration':60.01,'width':1280,'height':720,'fps':30})]:
            try:call(op,body);raise AssertionError('web policy bypass')
            except VocalError:pass
        checks.append('web_duration_live_video_limits')
        service.release(sid,identity)
    finally:service.close()
    report=dict(success=True,profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),checks=checks)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
