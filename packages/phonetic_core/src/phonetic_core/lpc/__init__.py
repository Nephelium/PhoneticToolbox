"""LPC spectrum arrays and immutable analysis configuration."""
from .spectrum import LPCConfig, LPCResult, compute_spectrum, mono_samples, extract_label, next_tier_name, select_segment
from .errors import LPCError, LPCCancelled, MAX_ROI_SAMPLES

__all__ = ['LPCConfig', 'LPCResult', 'compute_spectrum', 'mono_samples', 'extract_label',
           'next_tier_name', 'select_segment', 'LPCError', 'LPCCancelled', 'MAX_ROI_SAMPLES']
