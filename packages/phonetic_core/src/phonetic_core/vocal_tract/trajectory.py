"""One interpolated trajectory for native tube synthesis and cached display frames."""
import numpy as np
from .source import validate_source, interpolate_source


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


def sample_trajectory(frames, seconds, pitch_curve=None):
    pose=sample_frame(frames,seconds)
    if pitch_curve and not pose.get('silent',False):
        total=sum(f['duration'] for f in frames)
        pose['f0']=float(np.interp(seconds/total,[p[0] for p in pitch_curve],[p[1] for p in pitch_curve]))
    return pose

def validate_frames(engine,frames,*,for_storage=False):
    if not isinstance(frames,list) or len(frames)<(0 if for_storage else 2):raise ValueError('播放至少需要 2 个关键帧')
    result=[]
    for frame in frames:
        if not isinstance(frame,dict):raise ValueError('Invalid keyframe')
        width=float(frame.get('lip_width',1));f0=float(frame.get('f0',150));duration=float(frame.get('duration',.2))
        if not np.isfinite([width,f0,duration]).all() or not .55<=width<=1.6 or not .05<=duration<=3:raise ValueError('无效的关键帧参数（时长 0.05–3 秒）')
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
    if not for_storage and round(sum(f['duration'] for f in result),9)>12:raise ValueError('关键帧总时长不能超过 12 秒')
    return result

def sample_frame(frames,seconds):
    elapsed=0.
    for i,frame in enumerate(frames[:-1]):
        if seconds<elapsed+frame['duration']:
            if frame.get('silent',False):return {key:frame[key] for key in ['params','lip_width','f0','source','manual_root','silent','larynx_height'] if key in frame}
            t=float(np.clip((seconds-elapsed)/frame['duration'],0,1));t=t*t*(3-2*t);end=frames[i+1]
            result={'params':((1-t)*np.array(frame['params'])+t*np.array(end['params'])).tolist(),
                    'lip_width':(1-t)*frame['lip_width']+t*end['lip_width'],'f0':(1-t)*frame['f0']+t*(frame['f0'] if end.get('silent') else end['f0']),
                    'manual_root':frame.get('manual_root',False) or end.get('manual_root',False)}
            if 'source' in frame or 'source' in end:result['source']=interpolate_source(frame.get('source'),frame.get('source') if end.get('silent') else end.get('source'),t)
            if 'larynx_height' in frame or 'larynx_height' in end:result['larynx_height']=(1-t)*frame.get('larynx_height',0.)+t*end.get('larynx_height',0.)
            return result
        elapsed+=frame['duration']
    return {key:frames[-1][key] for key in ['params','lip_width','f0','source','manual_root','silent','larynx_height'] if key in frames[-1]}

