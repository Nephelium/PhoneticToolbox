from __future__ import annotations

import numpy as np

from phonetic_toolbox.core.acoustic import f0_praat as f0_module
from phonetic_toolbox.core.acoustic.common import segment_for_frame
from phonetic_toolbox.services.acoustic_service import align_track_to_grid


class _FakePitch:
    def __init__(self) -> None:
        self.selected_array = {
            "frequency": np.array([0.0, 120.0, 121.0], dtype=float),
        }

    def xs(self) -> np.ndarray:
        return np.array([0.0175, 0.0225, 0.0275], dtype=float)


class _FakeSound:
    def __init__(self) -> None:
        self.cc_calls = 0
        self.ac_calls = 0

    def to_pitch_cc(self, **_kwargs) -> _FakePitch:
        self.cc_calls += 1
        return _FakePitch()

    def to_pitch(self, **_kwargs) -> _FakePitch:
        self.ac_calls += 1
        return _FakePitch()


def test_praat_cc_uses_cross_correlation_and_preserves_times(monkeypatch):
    fake_sound = _FakeSound()
    monkeypatch.setattr(f0_module.parselmouth, "Sound", lambda _path: fake_sound)

    track = f0_module.compute_praat_f0_track(
        "sample.wav",
        frameshift_ms=5.0,
        min_f0=60.0,
        max_f0=880.0,
        method="cc",
    )

    assert fake_sound.cc_calls == 1
    assert fake_sound.ac_calls == 0
    np.testing.assert_allclose(track.times, [0.0175, 0.0225, 0.0275])
    np.testing.assert_allclose(track.values, [np.nan, 120.0, 121.0], equal_nan=True)


def test_legacy_praat_f0_api_still_returns_values(monkeypatch):
    fake_sound = _FakeSound()
    monkeypatch.setattr(f0_module.parselmouth, "Sound", lambda _path: fake_sound)

    values = f0_module.compute_praat_f0(
        "sample.wav",
        frameshift_ms=5.0,
        min_f0=60.0,
        max_f0=880.0,
        method="ac",
    )

    assert fake_sound.ac_calls == 1
    np.testing.assert_allclose(values, [np.nan, 120.0, 121.0], equal_nan=True)


def test_align_track_to_grid_does_not_extrapolate():
    source_times = np.array([0.025, 0.030], dtype=float)
    source_values = np.array([100.0, 110.0], dtype=float)
    target_times = np.arange(8, dtype=float) * 0.005

    aligned = align_track_to_grid(source_times, source_values, target_times)

    np.testing.assert_allclose(
        aligned,
        [np.nan, np.nan, np.nan, np.nan, np.nan, 100.0, 110.0, np.nan],
        equal_nan=True,
    )


def test_align_track_to_grid_preserves_unvoiced_gaps():
    source_times = np.array([0.000, 0.005, 0.010, 0.015], dtype=float)
    source_values = np.array([100.0, np.nan, np.nan, 110.0], dtype=float)
    target_times = np.array([0.000, 0.0025, 0.0075, 0.0125, 0.015], dtype=float)

    aligned = align_track_to_grid(source_times, source_values, target_times)

    np.testing.assert_allclose(
        aligned,
        [100.0, np.nan, np.nan, np.nan, 110.0],
        equal_nan=True,
    )


def test_segment_for_frame_uses_zero_based_grid_center():
    signal = np.arange(1000, dtype=float)

    segment = segment_for_frame(
        signal,
        fs=1000,
        frameshift_ms=10.0,
        k=5,
        N_periods=2,
        f0_curr=100.0,
    )

    np.testing.assert_array_equal(segment, signal[40:60])

