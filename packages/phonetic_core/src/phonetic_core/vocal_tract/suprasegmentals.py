"""Composable regional controls over an authored sequence, without pose mutation."""
import copy
import math
from .sequence_controls import validate_global_controls, apply_global_controls, EXTRA_LIMITS
from .lips import LIP_KEYS, validate_lip_shape
from .source import validate_source
from .perturbation import DEFAULT_PERTURBATION, validate_perturbation


def validate_features(engine,value=None):
    if value is None:return []
    if not isinstance(value,list) or len(value)>200:raise ValueError('超音段特征最多 200 项')
    result=[];seen=set()
    for item in value:
        if not isinstance(item,dict):raise ValueError('无效的超音段特征')
        key=item.get('id');enabled=item.get('enabled',True)
        if not isinstance(key,str) or not 1<=len(key)<=64 or key in seen or type(enabled) is not bool:raise ValueError('超音段特征标识无效或重复')
        parameter=item.get('parameter');control=validate_global_controls(engine,{parameter:{'mode':item.get('mode'),'value':item.get('value')}})[parameter]
        scope=item.get('selection')
        if not isinstance(scope,dict):raise ValueError('请选择关键帧或起止时间')
        if scope.get('kind')=='frames':
            ids=scope.get('ids')
            if not isinstance(ids,list) or not 1<=len(ids)<=2000 or any(not isinstance(s,str) or not 1<=len(s)<=64 for s in ids) or len(ids)!=len(set(ids)):
                raise ValueError('请选择至少一个关键帧')
            selection={'kind':'frames','ids':list(ids)}
        elif scope.get('kind')=='time':
            start,end=scope.get('start'),scope.get('end')
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in (start,end)) or not 0<=start<end<=86400:raise ValueError('起止时间须满足 0 ≤ 起点 < 终点，单位为秒')
            selection={'kind':'time','start':float(start),'end':float(end)}
        else:raise ValueError('无效的超音段特征范围')
        seen.add(key);result.append(dict(id=key,enabled=enabled,parameter=parameter,**control,selection=selection))
    return result


def feature_intervals(feature,frames):
    scope=feature['selection']
    if scope['kind']=='time':return [(scope['start'],scope['end'])]
    wanted=set(scope['ids']);intervals=[];start=0.
    for frame in frames:
        end=start+frame['duration']
        if frame.get('frame_id') in wanted:intervals.append((start,end))
        start=end
    return intervals


def compile_features(features,frames):
    """Resolve frame identities after ordering/duration edits, once per render."""
    return [(f,feature_intervals(f,frames)) for f in features if f['enabled']]


def apply_features(engine,pose,compiled,seconds,total):
    active=[f for f,intervals in compiled if any(a<=seconds<b or seconds==total==b for a,b in intervals)]
    if not active:return pose
    fixed={};offset={}
    for f in active:
        if f['mode']=='fixed':fixed[f['parameter']]=f['value']
        else:offset[f['parameter']]=offset.get(f['parameter'],0.)+f['value']
    controls={}
    for key in fixed.keys()|offset.keys():
        if key in fixed:base=fixed[key]
        elif key in LIP_KEYS:base=validate_lip_shape(pose.get('lip_shape'))[LIP_KEYS.index(key)]
        elif key=='pressure_pa':base=validate_source(pose.get('source'))['pressure_pa']
        elif key in ('jitter','shimmer'):base=(validate_perturbation(pose.get('perturbation')) or DEFAULT_PERTURBATION)[key]
        elif key in EXTRA_LIMITS:base=pose.get(key,1. if key=='lip_width' else 0.)
        else:base=pose['params'][engine.names.index(key)]
        controls[key]={'mode':'fixed','value':base+offset.get(key,0.)}
    # Individual authored values were validated already. Sum first, clamp once.
    return apply_global_controls(engine,pose,controls)
