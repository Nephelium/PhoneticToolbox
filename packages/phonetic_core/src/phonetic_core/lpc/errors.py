"""M04 stable errors and input checks outside the preserved numerical function."""
from numbers import Integral, Real
import math
import numpy as np

MAX_ROI_SAMPLES = 48_000


class LPCError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class LPCCancelled(LPCError):
    def __init__(self):
        super().__init__('cancelled', 'LPC 计算已取消。')


def check_cancelled(callback):
    if callback is not None and callback():
        raise LPCCancelled()


def real_number(value):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_rate(fs):
    if isinstance(fs, (bool, np.bool_)) or not isinstance(fs, Integral) or not 1 <= fs <= 768000:
        raise LPCError('invalid_sample_rate', '采样率须为 1–768000 Hz 的整数。')


def numeric_array(samples, dimensions):
    try:
        data = np.asarray(samples)
    except (ValueError, TypeError) as exc:
        raise LPCError('invalid_audio', '音频须为规则实数数组。') from exc
    if data.ndim not in dimensions or not data.size or data.dtype.kind not in 'iuf':
        raise LPCError('invalid_audio', '音频数组的形状或类型无效。')
    return data


def finite_audio(data):
    if not np.isfinite(data).all():
        raise LPCError('nonfinite_audio', '音频包含 NaN 或无穷值。')
