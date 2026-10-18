"""Portable M10 sequence contract; no paths or platform dependencies."""
from .trajectory import validate_frames, validate_pitch_curve, validate_pressure_curve
from .perturbation import validate_perturbation, PERTURBATION_METHOD
from .suprasegmentals import validate_features
from .stop_timing import STOP_TIMING_METHOD
from .sequence_controls import validate_global_controls

FORMAT = 'phonetic-toolbox-vocal-tract'


def sequence_document(engine, obj):
    if not isinstance(obj, dict) or obj.get('format') != FORMAT or obj.get('version') not in (1,2,3,4,5,6,7,8):
        raise ValueError('不支持此声道关键帧文件版本（支持 1–8）')
    speaker=obj.get('speaker') if obj['version']>=7 else 'JD2'
    if speaker not in ('JD2','W02') or speaker!=getattr(engine,'speaker','JD2'):
        raise ValueError('关键帧属于另一声道模型，请先切换为'+('女性' if speaker=='W02' else '男性')+'模型后导入')
    if obj['version']<7 and obj.get('global_controls'):
        raise ValueError('全局构音调整需要版本 7 文件')
    model={1:'VTL-2.4-JD2',2:'VTL-2.4-JD2-M10-3',3:'VTL-2.4-JD2-M10-R16',4:'VTL-2.4-JD2-M10-R17',5:'VTL-2.4-JD2-M10-R18',6:'VTL-2.4-JD2-M10-R18'}.get(obj['version'])
    if obj['version']>=7:
        if obj.get('geometry_method') not in ('m10/r18','m10/r19','m10/r20','m10/r21'):raise ValueError('不支持此声道几何方法')
        model=f"VTL-2.4-{speaker}-M10-{obj['geometry_method'].split('/')[-1].upper()}"
    if obj['version']<6 and (obj.get('pressure_curve') or obj.get('perturbation') is not None):
        raise ValueError('驱动气压曲线和自定义声源抖动需要版本 6 文件')
    if obj.get('model') != model or obj.get('parameters') != engine.names:
        raise ValueError('关键帧文件的模型或参数顺序不匹配')
    if obj['version']<5 and any(f.get('aspiration') or any(f.get('tongue_shape',[])) for f in obj.get('frames',[]) if isinstance(f,dict)):
        raise ValueError('独立舌面、舌底控制和送气帧需要版本 5 文件')
    if obj['version']<8 and (obj.get('features') or any(any(f.get('lip_shape',[])) or f.get('stop_timing') for f in obj.get('frames',[]) if isinstance(f,dict))):raise ValueError('独立唇部、爆破时序和区段特征需要版本 8 文件')
    if obj['version']<8 and obj.get('geometry_method') in ('m10/r20','m10/r21'):raise ValueError('此几何方法需要版本 8 文件')
    frames = validate_frames(engine, obj.get('frames'), for_storage=True)
    # Imported positions must not silently clamp to a different articulation.
    for original, validated in zip(obj['frames'], frames):
        if not validated.get('aspiration') and original['params'] != validated['params']:
            raise ValueError('导入的器官参数超出模型范围')
    if obj['version']==1 and any(f.get('larynx_height',0)!=0 or
            any(v>hi for v,hi in zip(f['params'],engine.legacy_maximum)) for f in frames):
        raise ValueError('扩展舌位或独立声门高低需要版本 2 文件')
    method=obj.get('geometry_method') if obj['version']>=7 else {3:'m10/r16',4:'m10/r17',5:'m10/r18',6:'m10/r18'}.get(obj['version'])
    if method and obj.get('geometry_method')!=method:
        raise ValueError('关键帧文件的几何方法版本不匹配')
    result={'format': FORMAT, 'version': obj['version'], 'model': model,
            'parameters': engine.names, 'frames': frames,
            'pitch_curve': validate_pitch_curve(obj.get('pitch_curve', []))}
    if method:result['geometry_method']=method
    if obj['version']>=6:result['pressure_curve']=validate_pressure_curve(obj.get('pressure_curve',[]))
    if obj['version']>=6:result['perturbation']=validate_perturbation(obj.get('perturbation'))
    if obj['version']>=6 and result['perturbation'] is not None:
        if obj.get('perturbation_method')!=PERTURBATION_METHOD:raise ValueError('关键帧文件的声源抖动方法不匹配')
        result['perturbation_method']=PERTURBATION_METHOD
    if obj['version']>=7:
        result.update(speaker=speaker,global_controls=validate_global_controls(engine,obj.get('global_controls')))
        if obj.get('audition_method','m10/pitch-level/1')!='m10/pitch-level/1':raise ValueError('不支持此基频音量补偿方法')
        result['audition_method']='m10/pitch-level/1'
    if obj['version']>=8:
        result['features']=validate_features(engine,obj.get('features'))
        if obj.get('stop_timing_method') not in ('m10/stop-timing/1',STOP_TIMING_METHOD):raise ValueError('不支持此爆破时序方法')
        result['stop_timing_method']=STOP_TIMING_METHOD
    current=getattr(engine,'geometry_method','m10/r18')
    if obj['version']<5 or (method and method!=current):result['geometry_notice']=f"旧文件参数已保留，按当前 {current.split('/')[-1].upper()} 几何重新计算，声音及面积可能与旧版不同。"
    if obj.get('stop_timing_method')=='m10/stop-timing/1' and any(f.get('stop_timing') for f in frames):
        result['geometry_notice']=(result.get('geometry_notice','')+' 旧爆破时序数值与总时长保留，按修正后的起声与前浊方法重新合成。').strip()
    if 'section_count' in obj:
        count=obj['section_count']
        if type(count) is not int or count not in (33,65,129,257):raise ValueError('关键帧文件的截面数无效')
        result['section_count']=count
        if count!=engine.section_count:result['geometry_notice']=(result.get('geometry_notice','')+f' 文件记录 {count} 截面，当前使用 {engine.section_count} 截面重新计算。').strip()
    return result


def make_document(engine, frames, curve, pressure_curve=None, perturbation=None, global_controls=None,features=None):
    # New readers explicitly identify the changed geometry. Older readers
    # reject newer versions instead of silently producing their old geometry.
    pressure=validate_pressure_curve(pressure_curve or [])
    perturbation=validate_perturbation(perturbation)
    controls=validate_global_controls(engine,global_controls)
    speaker=getattr(engine,'speaker','JD2');method=getattr(engine,'geometry_method','m10/r18')
    extended=bool(controls) or speaker!='JD2' or method!='m10/r18'
    return sequence_document(engine, {'format': FORMAT, 'version': 8 if method in ('m10/r20','m10/r21') or features else 7 if extended else 6 if pressure or perturbation is not None else 5,
        'model': f"VTL-2.4-{speaker}-M10-{method.split('/')[-1].upper()}", 'geometry_method':method,'parameters': engine.names,'section_count':engine.section_count,
        **({'speaker':speaker,'global_controls':controls} if extended else {}),
        'features':features or [],'stop_timing_method':STOP_TIMING_METHOD,
        'frames': frames, 'pitch_curve': curve, **({'pressure_curve':pressure} if pressure else {}),
        **({'perturbation':perturbation,'perturbation_method':PERTURBATION_METHOD} if perturbation is not None else {})})
