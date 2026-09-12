"""M03 numerical compatibility core. No GUI, file, database or device access."""
from dataclasses import replace

from .config import EGGConfig
from .model import EGGAnalysisResult
from .preprocess import prepare
from ._legacy import LegacyCalculations

SOURCE_IDS = ('PENDING-EGG', 'SRC-PRAAT')
METHOD_VERSION = 'egg-legacy/1'


def analyze_events(result, config, cancel_event=None):
    return LegacyCalculations().analyze_events(replace(result), config, cancel_event)


def cq_segment(result, start_s, end_s, config, use_raw_signal=False):
    """Compatibility: +/-100ms and a second filtering of the processed signal."""
    return LegacyCalculations().calculate_cq_sq_segment(result, start_s, end_s, config, use_raw_signal)


def events_segment(result, start_s, end_s, config, use_raw_signal=False):
    """Compatibility: +/-50ms, always starting from normalized raw EGG."""
    return LegacyCalculations().get_events_segment(result, start_s, end_s, config, use_raw_signal)
