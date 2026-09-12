"""Simplified closed-phase inverse filter, source PENDING-EGG."""
from ._inverse_legacy import apply_simplified_cp_inverse_filtering


def inverse_filter(audio, fs, gci_times, *, lp_order=None):
    return apply_simplified_cp_inverse_filtering(audio, fs, gci_times, lp_order=lp_order)
