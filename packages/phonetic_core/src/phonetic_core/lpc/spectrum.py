"""Array-only M04 service rules; no file decoding or rendering."""
from dataclasses import dataclass
import numpy as np
from ._legacy import compute_lpc_spectrum


@dataclass(frozen=True)
class LPCConfig:
    order: int = 50
    freq_max_hz: int = 8000
    amp_min_db: float = -5.0
    amp_max_db: float = 35.0
    dynamic_y: bool = False


@dataclass(frozen=True)
class LPCResult:
    frequencies_hz: np.ndarray
    magnitude_db: np.ndarray
    amp_min_db: float
    amp_max_db: float


def compute_spectrum(audio_segment, fs, config=LPCConfig()):
    frequency, magnitude = compute_lpc_spectrum(audio_segment, fs, config.order)
    minimum, maximum = config.amp_min_db, config.amp_max_db
    if config.dynamic_y:
        visible = magnitude[frequency <= config.freq_max_hz]
        if visible.size > 0:
            minimum = float(np.min(visible) - 5.0)
            maximum = float(np.max(visible) + 5.0)
    return LPCResult(frequency, magnitude, minimum, maximum)


def mono_samples(samples):
    data = np.asarray(samples)
    if data.dtype == np.int16:
        y = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        y = (data.astype(np.float64) - 128) / 128.0
    else:
        y = data.astype(np.float64)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y


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
