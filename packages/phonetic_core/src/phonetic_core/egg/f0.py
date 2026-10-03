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


def praat_pitch(audio: np.ndarray, fs: int, *, pitch_floor=75., pitch_ceiling=600., cancel_event=None) -> PitchTrack:
    """10 ms AC; explicit search bounds, with legacy defaults for old requests."""
    check_cancel(cancel_event)
    sample_rate(fs)
    audio = vector(audio)
    try:
        import parselmouth
        sound = parselmouth.Sound(np.asarray(audio, dtype=float), sampling_frequency=fs)
        if not 0 < pitch_floor < pitch_ceiling < fs/2:
            raise ValueError('invalid_pitch_range')
        # A 30 Hz AC window needs 100 ms. Short files contain no complete frame.
        if pitch_floor == 30. and len(audio)/fs < 3/pitch_floor:
            return PitchTrack(np.array([]),np.array([]),np.array([]))
        pitch = sound.to_pitch(time_step=.01, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    except Exception as exc:
        raise EggError('pitch_failed') from exc
    check_cancel(cancel_event)
    raw = np.asarray(pitch.selected_array['frequency'], dtype=float)
    values = np.where(raw <= 0, np.nan, raw)
    return PitchTrack(np.asarray(pitch.xs(), dtype=float), values, np.arange(len(values))*.01+.005)


def reaper_pitch(audio, fs, backend):
    """SRC-REAPER: use the injected native engine on the normalized audio track."""
    from ..models.audio import AudioInput
    audio = vector(audio); sample_rate(fs)
    # Native REAPER needs enough context for its epoch lattice. Empty/silent
    # tracks have no voiced estimate; do not invent a fallback frequency.
    if len(audio)/fs < .1 or not np.any(audio):
        return PitchTrack(np.array([]),np.array([]),np.array([]),'SRC-REAPER')
    track = backend(AudioInput(audio,fs),.01,30.,800.,hilbert=False,no_highpass=False)
    if track.actual_backend != 'native_reaper':raise EggError('egg_reaper_failed')
    times=np.asarray(track.times,dtype=float);values=np.asarray(track.values,dtype=float)
    if times.shape!=values.shape or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or np.isinf(values).any():
        raise EggError('egg_reaper_failed')
    return PitchTrack(times,np.where(values>0,values,np.nan),times.copy(),'SRC-REAPER')


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
