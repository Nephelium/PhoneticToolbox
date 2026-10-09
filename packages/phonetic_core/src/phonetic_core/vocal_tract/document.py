"""Portable M10 sequence contract; no paths or platform dependencies."""
from .trajectory import validate_frames, validate_pitch_curve

FORMAT = 'phonetic-toolbox-vocal-tract'


def sequence_document(engine, obj):
    if not isinstance(obj, dict) or obj.get('format') != FORMAT or obj.get('version') not in (1,2):
        raise ValueError('不是受支持的声道关键帧文件（版本 1 或 2）')
    model='VTL-2.4-JD2' if obj['version']==1 else 'VTL-2.4-JD2-M10-3'
    if obj.get('model') != model or obj.get('parameters') != engine.names:
        raise ValueError('关键帧文件的模型或参数顺序不匹配')
    frames = validate_frames(engine, obj.get('frames'), for_storage=True)
    # Imported positions must not silently clamp to a different articulation.
    for original, validated in zip(obj['frames'], frames):
        if original['params'] != validated['params']:
            raise ValueError('导入的器官参数超出模型范围')
    if obj['version']==1 and any(f.get('larynx_height',0)!=0 or
            any(v>hi for v,hi in zip(f['params'],engine.legacy_maximum)) for f in frames):
        raise ValueError('扩展舌位或独立声门高低需要版本 2 文件')
    return {'format': FORMAT, 'version': obj['version'], 'model': model,
            'parameters': engine.names, 'frames': frames,
            'pitch_curve': validate_pitch_curve(obj.get('pitch_curve', []))}


def make_document(engine, frames, curve):
    # Version 2 prevents older readers silently discarding extended geometry.
    extended=any(f.get('larynx_height',0)!=0 or any(v>hi for v,hi in zip(f['params'],getattr(engine,'legacy_maximum',engine.maximum))) for f in frames)
    return sequence_document(engine, {'format': FORMAT, 'version': 2 if extended else 1,
        'model': 'VTL-2.4-JD2-M10-3' if extended else 'VTL-2.4-JD2', 'parameters': engine.names,
        'frames': frames, 'pitch_curve': curve})
