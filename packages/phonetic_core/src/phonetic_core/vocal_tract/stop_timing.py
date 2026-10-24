"""Explicit stop gestures. Times are seconds; VOT is relative to release onset.

These are commanded articulator/phonation landmarks, not an acoustic VOT
measurement. The native pressure/flow solver supplies any release transient.
"""
import math
from .source import validate_source, interpolate_source

STOP_TIMING_METHOD='m10/stop-timing/2'


def default_stop_timing(duration):
    release=min(.008,duration*.1)
    closure=max(min(.06,duration*.7),duration-release-1.)
    aspiration=duration-closure-release
    return dict(closure=closure,release=release,aspiration=aspiration,vot=release+aspiration)


def validate_stop_timing(value,duration):
    if value is None:return None
    keys={'closure','release','aspiration','vot'}
    if not isinstance(value,dict) or set(value)!=keys or any(type(v) not in (int,float) or not math.isfinite(v) for v in value.values()):
        raise ValueError('无效的爆破音时序')
    c,r,a,v=(float(value[k]) for k in ('closure','release','aspiration','vot'))
    # A negative-VOT stop has no aspiration: its closure must accommodate the
    # existing 3 s frame limit while reserving at least 1 ms for release.
    if not .01<=c<=2.999 or not .001<=r<=.05 or not 0<=a<=1. or not -c<=v<=r+a or c+r+a>3. or abs(c+r+a-duration)>1e-8:
        raise ValueError('闭塞 10–2999 ms、释放 1–50 ms、送气 0–1000 ms；三者之和须等于帧时长且不超过 3 秒，浊音起始须位于该帧内')
    return dict(closure=c,release=r,aspiration=a,vot=v)


def gesture_at(frames,seconds):
    elapsed=0.
    for i,frame in enumerate(frames):
        start=elapsed;elapsed+=frame['duration']
        timing=frame.get('stop_timing')
        if timing is None:continue
        following=next((j for j in range(i+1,len(frames)) if not frames[j].get('aspiration')),None)
        if following is None:continue
        end=elapsed+sum(f['duration'] for f in frames[i+1:following])
        # An onset exactly at the frame boundary still needs its 2 ms ramp.
        # Carry only that source ramp into an ordinary following pose.
        tail=max(0.,timing['closure']+timing['vot']+.002-(end-start))
        if frames[following].get('stop_timing') or frames[following].get('silent'):tail=0.
        if start<=seconds<end+tail:return frame,frames[following],seconds-start,end-start,following
    return None


def sample_stop(frames,seconds,interpolate):
    found=gesture_at(frames,seconds)
    if found is None:return None
    frame,target,t,duration,following=found;timing=frame['stop_timing']
    start={k:v for k,v in frame.items() if k!='stop_timing'}
    end={k:v for k,v in target.items() if k!='stop_timing'}
    start['duration']=timing['release']
    pose=interpolate(frames[following:],t-duration) if t>=duration else interpolate([start,end],max(0.,min(timing['release'],t-timing['closure'])))
    # Preserve authored driving pressure/opening. Voicing begins independently
    # of the tract release, with a 2 ms activation ramp to avoid a digital step.
    voiced=validate_source(target.get('source'))
    authored=validate_source(frame.get('source'))
    if timing['vot']<0 and voiced['vibration']==0:
        voiced=authored if authored['vibration']>0 else validate_source(None)
    voice=voiced['vibration']
    onset=timing['closure']+timing['vot']
    vibration=voice*max(0.,min(1.,(t-onset)/.002))
    pressure=validate_source(pose.get('source'))['pressure_pa']
    if timing['vot']<0:
        # Prephonatory adduction prevents an open, non-vibrating glottis from
        # equalizing oral/subglottal pressure before the commanded prevoicing.
        unvoiced={**authored,'mode':'voiceless','vibration':0.,'opening_mm':0.,'posterior_gap_mm2':0.}
    else:
        unvoiced=authored if authored['vibration']==0 else {**validate_source({'mode':'voiceless'}),'pressure_pa':authored['pressure_pa'],'audition_gain_db':authored['audition_gain_db']}
    source=interpolate_source(unvoiced,voiced,max(0.,min(1.,(t-onset)/.002)))
    source['pressure_pa']=pressure
    pose['source']={**source,'vibration':vibration,'mode':'voiceless' if vibration==0 else 'voiced' if vibration==1 else 'transition'}
    pose['stop_phase']='闭塞前浊' if t<timing['closure'] and vibration>0 else '闭塞' if t<timing['closure'] else '释放' if t<timing['closure']+timing['release'] else '浊音起始' if vibration>0 else '送气'
    return pose


def timing_events(frames):
    events=set();fine=[];elapsed=0.
    for frame in frames:
        if (t:=frame.get('stop_timing')) is not None:
            release=elapsed+t['closure'];onset=release+t['vot']
            events.update((elapsed,elapsed+frame['duration'],release,release+t['release'],onset,onset+.002))
            fine.extend(((release,release+t['release']),(onset,onset+.002)))
        elapsed+=frame['duration']
    return events,fine
