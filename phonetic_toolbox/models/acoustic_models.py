from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchTrack:
    """A pitch track with explicit sample times in seconds."""

    times: np.ndarray
    values: np.ndarray

