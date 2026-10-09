"""M03 numerical compatibility core. No GUI, file, database or device access."""
from dataclasses import replace

from .config import EGGConfig
from .model import EGGAnalysisResult
from .preprocess import prepare
from ._legacy import LegacyCalculations
from .errors import check_cancel, roi, cutoffs

SOURCE_IDS = ('PENDING-EGG', 'SRC-PRAAT')
METHOD_VERSION = 'egg-legacy/1'


def analyze_events(result, config, cancel_event=None):
    if result.method_version == 'egg-bounded/2':
        from .bounded import analyze_events as bounded_events
        return bounded_events(result, config, cancel_event)
    check_cancel(cancel_event)
    cutoffs(config, result.fs)
    updated = LegacyCalculations().analyze_events(replace(result, analysis_config=config), config, cancel_event)
    check_cancel(cancel_event)
    return updated


def cq_segment(result, start_s, end_s, config, use_raw_signal=False, *, cancel_event=None):
    """Compatibility: +/-100ms and a second filtering of the processed signal."""
    check_cancel(cancel_event)
    roi(start_s, end_s)
    cutoffs(config, result.fs)
    value = LegacyCalculations().calculate_cq_sq_segment(result, start_s, end_s, config, use_raw_signal, cancel_event)
    check_cancel(cancel_event)
    return value


def events_segment(result, start_s, end_s, config, use_raw_signal=False, *, cancel_event=None):
    """Compatibility: +/-50ms, always starting from normalized raw EGG."""
    check_cancel(cancel_event)
    roi(start_s, end_s)
    cutoffs(config, result.fs)
    value = LegacyCalculations().get_events_segment(result, start_s, end_s, config, use_raw_signal, cancel_event)
    check_cancel(cancel_event)
    return value
