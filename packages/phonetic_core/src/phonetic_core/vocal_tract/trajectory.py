"""One interpolated trajectory for native tube synthesis and cached display frames."""
import numpy as np
from .source import validate_source, interpolate_source
from .lips import validate_lip_shape
from .stop_timing import validate_stop_timing, sample_stop
from .shape import validate_tongue_shape

MAX_SEQUENCE_SECONDS = 60.


def bounded_lerp(a, b, t):
    """A convex interpolation cannot exceed either endpoint, even by one ULP."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.clip(a + (b - a) * t, np.minimum(a, b), np.maximum(a, b)).tolist()


def validate_pitch_curve(curve):
    """Normalized time keeps an authored contour when frame durations change."""
    if not isinstance(curve,list) or len(curve)>201:
        raise ValueError('F0 曲线最多 201 个点')
    if not curve:
        return []
    points=np.asarray(curve,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
        raise ValueError('无效的 F0 曲线')
    if points[0,0]!=0 or points[-1,0]!=1 or np.any(np.diff(points[:,0])<=0) or np.any((points[:,1]<60)|(points[:,1]>350)):
        raise ValueError('F0 曲线需覆盖完整时间，频率为 60–350 Hz')
    return points.tolist()


def validate_pressure_curve(curve):
    """Authored driving pressure in Pa, independent of audition gain."""
    if not isinstance(curve,list) or len(curve)>201:
        raise ValueError('驱动气压曲线最多 201 个点')
    if not curve:return []
    points=np.asarray(curve,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
        raise ValueError('无效的驱动气压曲线')
    if points[0,0]!=0 or points[-1,0]!=1 or np.any(np.diff(points[:,0])<=0) or np.any((points[:,1]<0)|(points[:,1]>1600)):
        raise ValueError('驱动气压曲线需覆盖完整时间，范围为 0–1600 Pa')
    return points.tolist()


def sample_trajectory(frames, seconds, pitch_curve=None, pressure_curve=None):
    pose=sample_frame(frames,seconds)
    if pitch_curve and not pose.get('silent',False) and not pose.get('aspiration',False):
        total=sum(f['duration'] for f in frames)
        pose['f0']=float(np.interp(seconds/total,[p[0] for p in pitch_curve],[p[1] for p in pitch_curve]))
    if pressure_curve and not pose.get('silent',False):
        total=sum(f['duration'] for f in frames)
        pose['source']={**validate_source(pose.get('source')),
            'pressure_pa':float(np.interp(seconds/total,[p[0] for p in pressure_curve],[p[1] for p in pressure_curve]))}
    return pose

def validate_frames(engine,frames,*,for_storage=False):
    if not isinstance(frames,list) or len(frames)<(0 if for_storage else 1):raise ValueError('播放至少需要 1 个关键帧')
    result=[]
    for frame in frames:
        if not isinstance(frame,dict):raise ValueError('Invalid keyframe')
        aspiration=frame.get('aspiration',False)
        if type(aspiration) is not bool:raise ValueError('无效的送气帧类型')
        if aspiration:
            duration=float(frame.get('duration',.05))
            if not np.isfinite(duration) or not .01<=duration<=3:raise ValueError('送气时长应为 0.01–3 秒')
            if frame.get('silent') or any(k in frame for k in ('params','source','tongue_shape')):raise ValueError('送气帧不保存独立构形或声源')
            result.append({'aspiration':True,'duration':duration})
            if 'name' in frame:
                name=frame['name']
                if not isinstance(name,str) or len(name)>80 or any(ord(c)<32 for c in name):raise ValueError('帧名称须为不超过 80 字的单行文本')
                result[-1]['name']=name
            continue
        width=float(frame.get('lip_width',1));f0=float(frame.get('f0',150));duration=float(frame.get('duration',.2))
        minimum=.011 if frame.get('stop_timing') else .05
        if not np.isfinite([width,f0,duration]).all() or not .55<=width<=1.6 or not minimum-1e-10<=duration<=3:raise ValueError(f'无效的关键帧参数（时长 {minimum:g}–3 秒）')
        silent=frame.get('silent',False)
        if type(silent) is not bool:raise ValueError('无效的静音帧类型')
        engine.glottis(f0)
        result.append({'params':engine.validated(frame['params']).tolist(),'lip_width':width,'f0':f0,'duration':duration,'preset':frame.get('preset','') if frame.get('preset','') in engine.presets else ''})
        name=frame.get('name','')
        if not isinstance(name,str) or len(name)>80 or any(ord(c)<32 for c in name):raise ValueError('姿势名称须为不超过 80 字的单行文本')
        manual=frame.get('manual_root',False)
        if type(manual) is not bool:raise ValueError('无效的舌根模式')
        result[-1].update(name=name,manual_root=manual)
        if 'larynx_height' in frame:
            result[-1]['larynx_height']=engine.validated_larynx(frame['larynx_height'])
        if silent:result[-1]['silent']=True
        if 'source' in frame:result[-1]['source']=validate_source(frame['source'])
        if 'tongue_shape' in frame:result[-1]['tongue_shape']=validate_tongue_shape(frame['tongue_shape'])
        if 'lip_shape' in frame:result[-1]['lip_shape']=validate_lip_shape(frame['lip_shape'])
        if 'stop_timing' in frame:
            if silent:raise ValueError('静音帧不能添加爆破时序')
            result[-1]['stop_timing']=validate_stop_timing(frame['stop_timing'],duration)
    for frame,validated in zip(frames,result):
        if 'frame_id' in frame:
            key=frame['frame_id']
            if not isinstance(key,str) or not 1<=len(key)<=64 or any(ord(c)<33 for c in key):raise ValueError('无效的关键帧标识')
            validated['frame_id']=key
    ids=[f['frame_id'] for f in result if 'frame_id' in f]
    if len(ids)!=len(set(ids)):raise ValueError('关键帧标识重复')
    if not for_storage and result and next((f for f in reversed(result) if not f.get('aspiration')),{}).get('stop_timing'):
        raise ValueError('爆破时序需要后续释放目标构型，请添加元音或其他构型')
    if not for_storage and not any(not f.get('aspiration') for f in result):raise ValueError('送气帧需要至少一个构形帧')
    if not for_storage and round(sum(f['duration'] for f in result),9)>MAX_SEQUENCE_SECONDS:raise ValueError(f'关键帧总时长不能超过 {MAX_SEQUENCE_SECONDS:g} 秒')
    return result

def sample_frame(frames,seconds):
    if any(f.get('stop_timing') for f in frames):
        stop=sample_stop(frames,seconds,sample_frame)
        if stop is not None:return stop
    # Pose knots stay at ordinary frame starts. Intervening aspiration extends
    # a single smooth motion between those knots, without an invented posture.
    if any(f.get('aspiration') for f in frames):
        starts=np.cumsum([0.]+[f['duration'] for f in frames]).tolist()
        active=next((i for i in range(len(frames)) if seconds<starts[i+1]),len(frames)-1)
        knots=[i for i,f in enumerate(frames) if not f.get('aspiration')]
        if not knots:raise ValueError('送气帧需要至少一个构形帧')
        left=next((i for i in reversed(knots) if starts[i]<=seconds),knots[0])
        right=next((i for i in knots if starts[i]>seconds),knots[-1])
        if left==right:result=sample_frame([frames[left]],0)
        else:
            a={**frames[left],'duration':starts[right]-starts[left]}
            result=sample_frame([a,frames[right]],max(0.,seconds-starts[left]))
        if frames[active].get('aspiration'):
            result.pop('silent',None)
            result['aspiration']=True
            result['source']={**validate_source(result.get('source')),'mode':'voiceless','vibration':0.}
        return result
    elapsed=0.
    for i,frame in enumerate(frames[:-1]):
        if seconds<elapsed+frame['duration']:
            t=float(np.clip((seconds-elapsed)/frame['duration'],0,1));t=t*t*(3-2*t);end=frames[i+1]
            result={'params':bounded_lerp(frame['params'],end['params'],t),
                    'lip_width':bounded_lerp(frame['lip_width'],end['lip_width'],t),'f0':bounded_lerp(frame['f0'],frame['f0'] if end.get('silent') else end['f0'],t),
                    'manual_root':frame.get('manual_root',False) or end.get('manual_root',False)}
            if 'source' in frame or 'source' in end:result['source']=interpolate_source(frame.get('source'),frame.get('source') if end.get('silent') else end.get('source'),t)
            if 'larynx_height' in frame or 'larynx_height' in end:result['larynx_height']=bounded_lerp(frame.get('larynx_height',0.),end.get('larynx_height',0.),t)
            if 'tongue_shape' in frame or 'tongue_shape' in end:result['tongue_shape']=bounded_lerp(validate_tongue_shape(frame.get('tongue_shape')),validate_tongue_shape(end.get('tongue_shape')),t)
            if 'lip_shape' in frame or 'lip_shape' in end:result['lip_shape']=bounded_lerp(validate_lip_shape(frame.get('lip_shape')),validate_lip_shape(end.get('lip_shape')),t)
            # Silence gates the sound, not articulator motion. Keep its source
            # and F0 while the geometry reaches the following pose continuously.
            if frame.get('silent',False):
                result['silent']=True;result['f0']=frame['f0']
                if 'source' in frame:result['source']=frame['source']
                else:result.pop('source',None)
            return result
        elapsed+=frame['duration']
    return {key:frames[-1][key] for key in ['params','lip_width','f0','source','manual_root','silent','larynx_height','tongue_shape','lip_shape'] if key in frames[-1]}

