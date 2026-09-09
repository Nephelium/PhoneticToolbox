"""Explicit per-call F0 backends. Core never silently starts a native process."""
from dataclasses import dataclass
from typing import Callable, Protocol
import numpy as np
from ..models.audio import AudioInput


@dataclass(frozen=True)
class ReaperTrack:
    times: np.ndarray
    values: np.ndarray
    actual_backend: str
    reason: str | None = None


class ReaperBackend(Protocol):
    def __call__(self, audio: AudioInput, frame_interval_sec: float,
                 min_f0: float, max_f0: float, *, hilbert: bool,
                 no_highpass: bool) -> ReaperTrack: ...


def unavailable_reaper(*args, **kwargs):
    raise RuntimeError('REAPER backend must be provided by the adapter')


def python_reaper(audio, frame_interval_sec, min_f0, max_f0, *, hilbert, no_highpass):
    """Explicit Python policy, never presented as the native backend."""
    from ..acoustic.reaper_codec import reaper_pcm16
    from ..acoustic.reaper_python import run_python_samples
    times, _, values = run_python_samples(reaper_pcm16(audio), frame_interval_sec,
                                          min_f0, max_f0, hilbert, no_highpass)
    a = np.array(values, dtype=float)
    return ReaperTrack(np.array(times, dtype=float), np.where(a > 0., a, np.nan), 'reaper_python')


@dataclass(frozen=True)
class AcousticBackends:
    reaper: ReaperBackend = unavailable_reaper
    wm_f0: Callable | None = None  # None uses migrated IRAPT with original Praat fallback
