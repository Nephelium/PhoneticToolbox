"""Array-only M04 service rules; no file decoding or rendering."""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from ._legacy import compute_lpc_spectrum
from .errors import (LPCError, MAX_ROI_SAMPLES, check_cancelled, real_number,
                     validate_rate, numeric_array, finite_audio)


@dataclass(frozen=True)
class LPCConfig:
    order: int = 50
    freq_max_hz: int = 8000
    amp_min_db: float = -5.0
    amp_max_db: float = 35.0
    dynamic_y: bool = False

    def __post_init__(self):
        if (isinstance(self.order, (bool, np.bool_)) or not isinstance(self.order, Integral)
                or not 1 <= self.order <= 200):
            raise LPCError('invalid_config', 'LPC 阶数须为 1–200 的整数。')
        if not real_number(self.freq_max_hz) or not 100 <= self.freq_max_hz <= 48000:
            raise LPCError('invalid_config', '频率上限须为 100–48000 Hz。')
        if (not real_number(self.amp_min_db) or not real_number(self.amp_max_db)
                or not -200 <= self.amp_min_db < self.amp_max_db <= 100):
            raise LPCError('invalid_config', '纵轴范围须递增，且位于 −200 至 100 dB。')
        if not isinstance(self.dynamic_y, bool):
            raise LPCError('invalid_config', '动态纵轴须为布尔值。')


@dataclass(frozen=True)
class LPCResult:
    frequencies_hz: np.ndarray
    magnitude_db: np.ndarray
    amp_min_db: float
    amp_max_db: float


def compute_spectrum(audio_segment, fs, config=LPCConfig(), *, cancelled=None):
    """Compute unchanged V2 spectrum; cooperative cancellation before/after native work.

    A host must isolate this call to enforce a deadline during np.correlate.
    Analysis accepts a mono segment; PCM conversion is explicit in mono_samples.
    """
    check_cancelled(cancelled)
    validate_rate(fs)
    if not isinstance(config, LPCConfig):
        raise LPCError('invalid_config', '需要 LPCConfig 参数快照。')
    audio = numeric_array(audio_segment, (1,))
    if audio.size > MAX_ROI_SAMPLES:
        raise LPCError('roi_too_large', f'单次 LPC 分析最多 {MAX_ROI_SAMPLES} 个样本，请缩短选区。')
    if audio.size <= max(config.order+1, 2):
        raise LPCError('segment_too_short', '音频长度不足，无法计算当前阶数的 LPC。')
    finite_audio(audio)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            frequency, magnitude = compute_lpc_spectrum(audio, fs, config.order)
    except (ValueError, FloatingPointError) as exc:
        raise LPCError('solver_failed', 'LPC 求解失败，请检查静音、幅值或降低阶数。') from exc
    check_cancelled(cancelled)
    if not np.isfinite(magnitude).all():
        raise LPCError('solver_failed', 'LPC 求解未得到有限谱值。')
    minimum, maximum = config.amp_min_db, config.amp_max_db
    if config.dynamic_y:
        visible = magnitude[frequency <= config.freq_max_hz]
        if visible.size > 0:
            minimum = float(np.min(visible) - 5.0)
            maximum = float(np.max(visible) + 5.0)
    return LPCResult(frequency, magnitude, minimum, maximum)


def mono_samples(samples):
    data = numeric_array(samples, (1, 2))
    finite_audio(data)
    if data.dtype == np.int16:
        y = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        y = (data.astype(np.float64) - 128) / 128.0
    else:
        y = data.astype(np.float64)
    if y.ndim > 1:
        with np.errstate(over='ignore', invalid='ignore'):
            y = np.mean(y, axis=1)
    finite_audio(y)
    return y


def select_segment(mono, fs, start_sec, end_sec):
    """Copy the explicit ROI using V2 int(t*fs) half-open sample indexing."""
    validate_rate(fs)
    audio = numeric_array(mono, (1,))
    if (not real_number(start_sec) or not real_number(end_sec)
            or not 0 <= start_sec < end_sec <= audio.size/fs):
        raise LPCError('invalid_roi', '选区须位于音频范围内，且结束时间大于起始时间。')
    start, end = int(start_sec*fs), int(end_sec*fs)
    if end <= start:
        raise LPCError('invalid_roi', '选区不足一个样本。')
    if end-start > MAX_ROI_SAMPLES:
        raise LPCError('roi_too_large', f'单次 LPC 分析最多 {MAX_ROI_SAMPLES} 个样本，请缩短选区。')
    segment = audio[start:end].copy()
    finite_audio(segment)
    return segment


def next_tier_name(tiers, current_tier_name):
    if not tiers:
        return None
    names = [tier.name for tier in tiers]
    if not current_tier_name or current_tier_name not in names:
        return names[0]
    return names[(names.index(current_tier_name)+1) % len(names)]


def extract_label(tiers, tier_name, start_sec, end_sec):
    if not tiers or not tier_name:
        return ''
    tier = next((tier for tier in tiers if tier.name == tier_name), None)
    if tier is None:
        return ''
    labels = []
    for interval in tier.intervals:
        if interval.xmax <= start_sec or interval.xmin >= end_sec:
            continue
        if (start_sec < end_sec and interval.xmin < end_sec <= interval.xmax
                and not (interval.xmin <= start_sec < interval.xmax)):
            continue
        label = interval.text.strip()
        if label and label not in labels:
            labels.append(label)
    return '+'.join(labels)
