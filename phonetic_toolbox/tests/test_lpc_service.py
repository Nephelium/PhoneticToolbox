import numpy as np

from phonetic_toolbox.models.lpc_models import LPCSpectrumConfig
from phonetic_toolbox.services.io.textgrid import Interval, TextGrid, Tier
from phonetic_toolbox.services.lpc_service import LPCSpectrumService


def test_extract_label_in_range_returns_joined_labels():
    service = LPCSpectrumService()
    tg = TextGrid(
        xmin=0.0,
        xmax=1.0,
        tiers=[
            Tier(
                name="phones",
                xmin=0.0,
                xmax=1.0,
                intervals=[
                    Interval(0.0, 0.2, "a"),
                    Interval(0.2, 0.4, "b"),
                    Interval(0.4, 0.6, ""),
                    Interval(0.6, 0.8, "b"),
                    Interval(0.8, 1.0, "c"),
                ],
            )
        ],
    )
    label = service.extract_label_in_range(tg, "phones", 0.15, 0.85)
    assert label == "a+b"


def test_compute_spectrum_returns_expected_shape():
    service = LPCSpectrumService()
    fs = 16000
    t = np.arange(fs) / fs
    y = np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    config = LPCSpectrumConfig(order=24, freq_max_hz=8000, amp_min_db=-20.0, amp_max_db=30.0)
    result = service.compute_spectrum(y, fs, config)
    assert result.frequencies_hz.shape == result.magnitude_db.shape
    assert result.frequencies_hz.size == 1024
