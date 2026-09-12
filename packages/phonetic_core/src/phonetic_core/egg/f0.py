"""Array-only Praat and original GCI heuristics. Sources SRC-PRAAT / PENDING-EGG."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ._legacy import LegacyCalculations


@dataclass(frozen=True)
class PitchTrack:
    times: np.ndarray
    values: np.ndarray
    legacy_times: np.ndarray
    source_id: str = 'SRC-PRAAT'


def praat_pitch(audio: np.ndarray, fs: int) -> PitchTrack:
    """v2's 10ms / 75–600Hz autocorrelation, preserving both actual and old grids."""
    import parselmouth
    sound = parselmouth.Sound(np.asarray(audio, dtype=float), sampling_frequency=fs)
    pitch = sound.to_pitch(time_step=.01, pitch_floor=75., pitch_ceiling=600.)
    raw = np.asarray(pitch.selected_array['frequency'], dtype=float)
    values = np.where(raw <= 0, np.nan, raw)
    return PitchTrack(np.asarray(pitch.xs(), dtype=float), values, np.arange(len(values))*.01+.005)


def gci_pitch(gci_times):
    holder = SimpleNamespace(gci_times=list(gci_times))
    LegacyCalculations()._calculate_gci_f0(holder)
    return holder.gci_f0_times, holder.gci_f0_values


def glottal_movement(times: np.ndarray, values: np.ndarray):
    """Legacy F0-slope heuristic, not a measurement of anatomical displacement."""
    holder = SimpleNamespace(audio_f0_times=np.asarray(times), audio_f0_values=np.asarray(values))
    LegacyCalculations().detect_glottal_movement(holder)
    return holder.glottal_movement_events
