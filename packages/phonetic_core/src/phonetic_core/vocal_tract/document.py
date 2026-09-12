"""Portable M10 sequence contract; no paths or platform dependencies."""
from .trajectory import validate_frames, validate_pitch_curve

FORMAT = 'phonetic-toolbox-vocal-tract'


def sequence_document(engine, obj):
    if not isinstance(obj, dict) or obj.get('format') != FORMAT or obj.get('version') != 1:
        raise ValueError('不是受支持的声道关键帧文件（版本 1）')
    if obj.get('model') != 'VTL-2.4-JD2' or obj.get('parameters') != engine.names:
        raise ValueError('关键帧文件的模型或参数顺序不匹配')
    frames = validate_frames(engine, obj.get('frames'), for_storage=True)
    # Imported positions must not silently clamp to a different articulation.
    for original, validated in zip(obj['frames'], frames):
        if original['params'] != validated['params']:
            raise ValueError('导入的器官参数超出模型范围')
    return {'format': FORMAT, 'version': 1, 'model': 'VTL-2.4-JD2',
            'parameters': engine.names, 'frames': frames,
            'pitch_curve': validate_pitch_curve(obj.get('pitch_curve', []))}


def make_document(engine, frames, curve):
    return sequence_document(engine, {'format': FORMAT, 'version': 1,
        'model': 'VTL-2.4-JD2', 'parameters': engine.names,
        'frames': frames, 'pitch_curve': curve})
