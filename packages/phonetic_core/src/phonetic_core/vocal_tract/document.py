"""Portable M10 sequence contract; no paths or platform dependencies."""
from .trajectory import validate_frames, validate_pitch_curve

FORMAT = 'phonetic-toolbox-vocal-tract'


def sequence_document(engine, obj):
    if not isinstance(obj, dict) or obj.get('format') != FORMAT or obj.get('version') not in (1,2,3,4,5):
        raise ValueError('不支持此声道关键帧文件版本（支持 1–5）')
    model={1:'VTL-2.4-JD2',2:'VTL-2.4-JD2-M10-3',3:'VTL-2.4-JD2-M10-R16',4:'VTL-2.4-JD2-M10-R17',5:'VTL-2.4-JD2-M10-R18'}[obj['version']]
    if obj.get('model') != model or obj.get('parameters') != engine.names:
        raise ValueError('关键帧文件的模型或参数顺序不匹配')
    if obj['version']<5 and any(f.get('aspiration') or any(f.get('tongue_shape',[])) for f in obj.get('frames',[]) if isinstance(f,dict)):
        raise ValueError('独立舌面、舌底控制和送气帧需要版本 5 文件')
    frames = validate_frames(engine, obj.get('frames'), for_storage=True)
    # Imported positions must not silently clamp to a different articulation.
    for original, validated in zip(obj['frames'], frames):
        if not validated.get('aspiration') and original['params'] != validated['params']:
            raise ValueError('导入的器官参数超出模型范围')
    if obj['version']==1 and any(f.get('larynx_height',0)!=0 or
            any(v>hi for v,hi in zip(f['params'],engine.legacy_maximum)) for f in frames):
        raise ValueError('扩展舌位或独立声门高低需要版本 2 文件')
    method={3:'m10/r16',4:'m10/r17',5:'m10/r18'}.get(obj['version'])
    if method and obj.get('geometry_method')!=method:
        raise ValueError('关键帧文件的几何方法版本不匹配')
    result={'format': FORMAT, 'version': obj['version'], 'model': model,
            'parameters': engine.names, 'frames': frames,
            'pitch_curve': validate_pitch_curve(obj.get('pitch_curve', []))}
    if method:result['geometry_method']=method
    if obj['version']<5:result['geometry_notice']='旧文件参数已保留，按当前 R18 几何重新计算，声音及面积可能与旧版不同。'
    return result


def make_document(engine, frames, curve):
    # New readers explicitly identify the changed geometry. Older readers
    # reject version 5 instead of silently producing their old geometry.
    return sequence_document(engine, {'format': FORMAT, 'version': 5,
        'model': 'VTL-2.4-JD2-M10-R18', 'geometry_method':'m10/r18','parameters': engine.names,
        'frames': frames, 'pitch_curve': curve})
