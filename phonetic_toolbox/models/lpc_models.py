from dataclasses import dataclass

import numpy as np


@dataclass
class LPCSpectrumConfig:
    order: int = 50
    freq_max_hz: int = 8000
    amp_min_db: float = -5.0
    amp_max_db: float = 35.0
    dynamic_y: bool = False


@dataclass
class LPCSpectrumResult:
    frequencies_hz: np.ndarray
    magnitude_db: np.ndarray
    amp_min_db: float
    amp_max_db: float
