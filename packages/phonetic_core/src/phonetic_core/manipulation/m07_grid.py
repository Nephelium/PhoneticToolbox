"""V2 service numeric helpers, unchanged. SRC-ZAIWA."""
import math
import numpy as np
from scipy import signal

def track_to_millisecond_grid(times_sec: np.ndarray, values_hz: np.ndarray, duration_sec: float) -> np.ndarray:
    grid_times = np.arange(int(math.ceil(duration_sec * 1000.0)) + 1) / 1000.0
    output = np.zeros_like(grid_times)
    times = np.asarray(times_sec, dtype=np.float64)
    values = np.asarray(values_hz, dtype=np.float64)
    if times.shape != values.shape:
        raise ValueError('F0 时间轴和值的长度不一致。')
    valid = np.isfinite(times) & np.isfinite(values) & (values > 0)
    if not np.any(valid):
        return output
    valid_times = times[valid]
    valid_values = values[valid]
    order = np.argsort(valid_times)
    valid_times = valid_times[order]
    valid_values = valid_values[order]
    output = np.interp(grid_times, valid_times, valid_values, left=0.0, right=0.0)
    output[grid_times < valid_times[0]] = 0.0
    output[grid_times > valid_times[-1]] = 0.0
    return output

def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64)
    if source_rate == target_rate:
        return values.copy()
    divisor = math.gcd(int(source_rate), int(target_rate))
    return signal.resample_poly(values, target_rate // divisor, source_rate // divisor).astype(np.float64)
