from __future__ import annotations

import numpy as np
import pytest

from phonetic_toolbox.core.manipulation.phonation_synthesis import (
    build_lpc_residual,
    enframe,
    interpolate_f0_control_points,
    make_residual_continuum,
    sample_f0_control_points,
    synthesize_from_residual,
    trim_edge_silence,
)
from phonetic_toolbox.models.phonation_synthesis_models import (
    ContinuumType,
    F0AlignmentMode,
    PhonationAnalysisConfig,
    PhonationAnalysisResult,
    PhonationGenerationConfig,
)


def _analysis(*, residual_scale: float = 1.0, f0_hz: float = 120.0) -> PhonationAnalysisResult:
    sample_rate = 1000
    samples = 500
    time = np.arange(samples, dtype=float) / sample_rate
    signal = 0.4 * np.sin(2.0 * np.pi * 100.0 * time)
    residual = residual_scale * np.sin(2.0 * np.pi * f0_hz * time)
    pulses = np.arange(50, 451, max(2, int(round(sample_rate / f0_hz))), dtype=int)
    config = PhonationAnalysisConfig(
        target_sample_rate=sample_rate,
        min_f0_hz=50.0,
        max_f0_hz=300.0,
        frame_length=64,
        frame_shift=16,
        lpc_order=12,
    )
    return PhonationAnalysisResult(
        sample_rate=sample_rate,
        signal=signal,
        start_sample=40,
        end_sample=460,
        f0_hz=np.full(501, f0_hz, dtype=float),
        lpc_coefficients=np.tile(
            np.r_[1.0, np.zeros(config.lpc_order)],
            (1 + (421 - config.frame_length) // config.frame_shift, 1),
        ),
        residual=residual,
        pulses=pulses,
        config=config,
    )


def test_analysis_config_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="最低 F0"):
        PhonationAnalysisConfig(min_f0_hz=300.0, max_f0_hz=200.0).validate()

    with pytest.raises(ValueError, match="帧移"):
        PhonationAnalysisConfig(frame_length=64, frame_shift=64).validate()

    with pytest.raises(ValueError, match="脉冲外窗"):
        PhonationAnalysisConfig(pulse_inner_periods=1.0, pulse_outer_periods=0.8).validate()


def test_generation_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="合成步数"):
        PhonationGenerationConfig(step_count=1).validate()

    with pytest.raises(ValueError, match="峰值限制"):
        PhonationGenerationConfig(output_peak_limit=0.0).validate()


def test_trim_edge_silence_keeps_requested_padding():
    sample_rate = 1000
    audio = np.r_[np.zeros(100), np.ones(200) * 0.5, np.zeros(100)]

    trimmed = trim_edge_silence(audio, sample_rate, threshold_db=-20.0, pad_ms=10.0)

    assert 210 <= len(trimmed) <= 230
    assert np.max(np.abs(trimmed)) == pytest.approx(0.5)


def test_lpc_residual_has_finite_expected_shapes():
    sample_rate = 8000
    time = np.arange(800) / sample_rate
    audio = 0.6 * np.sin(2 * np.pi * 180 * time)

    coefficients, residual = build_lpc_residual(
        audio,
        start_sample=0,
        end_sample=len(audio) - 1,
        frame_length=128,
        frame_shift=32,
        order=20,
        preemphasis=0.98,
        window_name="hamming",
    )

    assert coefficients.shape[1] == 21
    assert residual.shape == audio.shape
    assert np.isfinite(coefficients).all()
    assert np.isfinite(residual).all()


@pytest.mark.parametrize("mode", list(F0AlignmentMode))
def test_f0_control_points_round_trip(mode: F0AlignmentMode):
    original = np.r_[np.zeros(5), np.linspace(100.0, 140.0, 41), np.zeros(4)]
    axis, sampled = sample_f0_control_points(original, point_count=21, mode=mode)
    edited = interpolate_f0_control_points(axis, sampled, original, mode=mode)

    assert edited.shape == original.shape
    assert np.count_nonzero(edited > 0) == 41
    assert edited[5] == pytest.approx(100.0)
    assert edited[45] == pytest.approx(140.0)


@pytest.mark.parametrize("continuum", list(ContinuumType))
def test_all_continuum_types_produce_finite_steps(continuum: ContinuumType):
    source = _analysis(residual_scale=1.0, f0_hz=120.0)
    target = _analysis(residual_scale=0.45, f0_hz=160.0)
    generation = PhonationGenerationConfig(step_count=5)

    residual_steps = make_residual_continuum(
        source,
        target,
        continuum,
        generation.step_count,
        energy_match=generation.energy_match,
    )
    audio_steps = synthesize_from_residual(
        source,
        residual_steps,
        normalize_to_source=generation.normalize_to_source,
        output_peak_limit=generation.output_peak_limit,
    )

    assert residual_steps.shape == (len(source.signal), generation.step_count)
    assert audio_steps.shape == residual_steps.shape
    assert np.isfinite(audio_steps).all()
    assert np.max(np.abs(audio_steps)) <= generation.output_peak_limit + 1e-12


@pytest.mark.parametrize("sample_count", [1, 63, 128, 129, 160, 1000])
def test_frames_cover_every_input_sample(sample_count):
    audio = np.arange(1, sample_count + 1, dtype=float)
    frames = enframe(audio, 128, 32)
    restored = np.zeros(sample_count)
    weights = np.zeros(sample_count)
    for index, frame in enumerate(frames):
        start = index * 32
        end = min(start + 128, sample_count)
        restored[start:end] += frame[:end-start]
        weights[start:end] += 1
    assert np.all(weights > 0)
    np.testing.assert_allclose(restored / weights, audio)


@pytest.mark.parametrize("sample_count", [63, 129, 1000])
def test_lpc_partial_frame_preserves_tail_and_surrounding_audio(sample_count):
    start = 17
    end = start + sample_count - 1
    audio = np.full(sample_count + 50, 0.1)
    audio[end] = 0.7
    config = PhonationAnalysisConfig(frame_length=128, frame_shift=32, lpc_order=20)
    coefficients, residual = build_lpc_residual(audio, start, end, 128, 32, 20)
    np.testing.assert_array_equal(residual[:start], audio[:start])
    np.testing.assert_array_equal(residual[end+1:], audio[end+1:])
    assert abs(residual[end]) > 1e-8
    source = PhonationAnalysisResult(
        11025, audio, start, end, np.full(100, 120.0), coefficients,
        residual, np.array([start, end]), config,
    )
    output = synthesize_from_residual(source, residual[:, None], False, 0.0)[:, 0]
    assert len(output) == len(audio)
    assert abs(output[end]) > 1e-8
    np.testing.assert_array_equal(output[:start], audio[:start])
    np.testing.assert_array_equal(output[end+1:], audio[end+1:])
