"""Source validation and continuous interpolation, independent of playback."""
import math
from phonetic_core.vocal_tract.source_models import SOURCE_PRESETS, SOURCE_LIMITS


def validate_source(source=None):
    if source is None:return dict(SOURCE_PRESETS['voiced'])
    if not isinstance(source,dict):raise ValueError('无效的声源控制')
    mode=source.get('mode','voiced')
    if mode not in (*SOURCE_PRESETS,'transition'):raise ValueError('未知声源模式')
    result=dict(SOURCE_PRESETS.get(mode,SOURCE_PRESETS['voiced']));result['mode']=mode
    for name,(lo,hi) in SOURCE_LIMITS.items():
        value=float(source.get(name,result[name]))
        if not math.isfinite(value) or not lo<=value<=hi:raise ValueError('无效的声源参数：'+name)
        result[name]=value
    if mode in ('voiceless','whisper') and result['vibration']!=0:raise ValueError('清声与耳语模式不能包含周期振动')
    return result


def interpolate_source(a,b,t):
    a=validate_source(a);b=validate_source(b)
    if t<=0:return a
    if t>=1:return b
    return {'mode':a['mode'] if a['mode']==b['mode'] else 'transition',
            **{k:a[k]*(1-t)+b[k]*t for k in SOURCE_LIMITS}}
