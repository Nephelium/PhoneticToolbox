"""M01-B source_id SRC-REAPER: array conversion extracted without formula changes.

No process, temporary file or executable discovery here. Empty conversions are
rejected explicitly; the service retains its empty-result behavior.
"""
import numpy as np
from scipy import signal


def reaper_pcm16(audio):
    data = audio.samples.copy()
    fs = audio.sample_rate_hz
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    target_fs = 16000
    if fs != target_fs:
        num_samples = int(len(data) * target_fs / fs)
        data = signal.resample(data, num_samples)
        fs = target_fs
    if data.dtype.kind == 'f':
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val * 32000.0
        data = data.astype(np.int16)
    elif data.dtype == np.int32:
        data = (data / 65536.0).astype(np.int16)
    elif data.dtype == np.uint8:
        data = ((data.astype(float) - 128.0) * 256.0).astype(np.int16)
    if data.dtype != np.int16:
        data = data.astype(np.int16)
    return data


def parse_est_f0(text):
    """Legacy EST numeric rows; resource/size enforcement belongs to the adapter."""
    times, voiced, values = [], [], []
    in_header = True
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if in_header:
            if s == 'EST_Header_End':
                in_header = False
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        try:
            t, v, val = float(parts[0]), int(float(parts[1])), float(parts[2])
        except (ValueError, OverflowError):
            continue
        times.append(t)
        voiced.append(v)
        values.append(val)
    return np.array(times, dtype=float), np.array(voiced, dtype=int), np.array(values, dtype=float)
