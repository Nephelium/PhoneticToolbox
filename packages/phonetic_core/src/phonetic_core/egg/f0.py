"""Array-only Praat and original GCI heuristics. Sources SRC-PRAAT / PENDING-EGG."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ._legacy import LegacyCalculations
from .errors import EggError, check_cancel, sample_rate, vector


@dataclass(frozen=True)
class PitchTrack:
    times: np.ndarray
    values: np.ndarray
    legacy_times: np.ndarray
    source_id: str = 'SRC-PRAAT'


def praat_pitch(audio: np.ndarray, fs: int, *, cancel_event=None) -> PitchTrack:
    """v2's 10ms / 75–600Hz autocorrelation, preserving both actual and old grids."""
    check_cancel(cancel_event)
    sample_rate(fs)
    audio = vector(audio)
    try:
        import parselmouth
        sound = parselmouth.Sound(np.asarray(audio, dtype=float), sampling_frequency=fs)
        pitch = sound.to_pitch(time_step=.01, pitch_floor=75., pitch_ceiling=600.)
    except Exception as exc:
        raise EggError('pitch_failed') from exc
    check_cancel(cancel_event)
    raw = np.asarray(pitch.selected_array['frequency'], dtype=float)
    values = np.where(raw <= 0, np.nan, raw)
    return PitchTrack(np.asarray(pitch.xs(), dtype=float), values, np.arange(len(values))*.01+.005)


def gci_pitch(gci_times):
    holder = SimpleNamespace(gci_times=vector(gci_times, allow_empty=True).tolist())
    LegacyCalculations()._calculate_gci_f0(holder)
    return holder.gci_f0_times, holder.gci_f0_values


def glottal_movement(times: np.ndarray, values: np.ndarray):
    """Legacy F0-slope heuristic, not a measurement of anatomical displacement."""
    times = vector(times, allow_empty=True)
    values = np.asarray(values)
    if times.shape != values.shape or values.dtype.kind not in 'iuf' or np.isinf(values).any() or np.any(np.diff(times) <= 0):
        raise EggError('invalid_pitch_grid')
    holder = SimpleNamespace(audio_f0_times=times, audio_f0_values=values)
    LegacyCalculations().detect_glottal_movement(holder)
    return holder.glottal_movement_events
