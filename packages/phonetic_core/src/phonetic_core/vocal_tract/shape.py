"""Local tongue offsets in cm; zero follows the original articulator model."""
import numpy as np

TONGUE_SHAPE_LIMITS=((-0.5,0.5),(-0.3,0.8),(-0.5,0.5),(-0.4,0.4))


def validate_tongue_shape(value=None):
    values=np.asarray([0.,0.,0.,0.] if value is None else value,dtype=float)
    if values.shape!=(4,) or not np.isfinite(values).all():
        raise ValueError('无效的舌面与舌底局部偏移')
    if any(not lo<=v<=hi for v,(lo,hi) in zip(values,TONGUE_SHAPE_LIMITS)):
        raise ValueError('舌面或舌底局部偏移超出范围')
    return values.tolist()
