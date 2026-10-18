"""Independent native lip offsets in centimetres, relative to common LP/LD."""
import math

LIP_KEYS = ('upper_lip_x','upper_lip_y','lower_lip_x','lower_lip_y')
LIP_LIMITS = [[-.8,.8] for _ in LIP_KEYS]


def validate_lip_shape(value=None):
    if value is None:return [0.,0.,0.,0.]
    if not isinstance(value,(list,tuple)) or len(value)!=4 or any(
        type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi
        for v,(lo,hi) in zip(value,LIP_LIMITS)):
        raise ValueError('上下唇位移各为前后、上下两项，范围 −0.8–0.8 cm')
    return [float(v) for v in value]
