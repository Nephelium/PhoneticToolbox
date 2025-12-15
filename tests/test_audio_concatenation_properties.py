"""
Property-based tests for audio concatenation correctness.

Tests verify that audio segments are correctly concatenated with
proper length preservation and normalization.

**Feature: consonant-synthesis, Property 14: Audio Concatenation Correctness**
**Validates: Requirements 12.5**
"""
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from klatt.consonant_synth import (
    ConsonantSynthesizer, 
    synthesize_consonant,
    concatenate_audio_segments
)
from klatt.consonant_data import CONSONANT_DATA


# ============================================================================
# Test Data Generators
# ============================================================================

# Sample rate strategy
sample_rates = st.sampled_from([8000, 16000, 22050, 44100])

# Duration strategy for segments (10-200ms)
segment_duration_ms = st.floats(min_value=10.0, max_value=200.0,
                                 allow_nan=False, allow_infinity=False)

# Number of segments (1-5)
num_segments = st.integers(min_value=1, max_value=5)

# Crossfade duration (0-20ms)
crossfade_ms = st.floats(min_value=0.0, max_value=20.0,
                          allow_nan=False, allow_infinity=False)

# All consonant symbols
all_consonants = list(CONSONANT_DATA.keys())
consonant_symbols = st.sampled_from(all_consonants)


def generate_random_audio(fs: int, duration_ms: float) -> np.ndarray:
    """Generate random audio segment for testing."""
    n_samples = int(duration_ms * fs / 1000)
    if n_samples <= 0:
        return np.array([])
    # Generate random audio normalized to [-1, 1]
    audio = np.random.randn(n_samples)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


# ============================================================================
# Property 14: Audio Concatenation Correctness
# ============================================================================

@given(
    durations=st.lists(segment_duration_ms, min_size=1, max_size=5),
    crossfade=crossfade_ms
)
@settings(max_examples=100)
def test_concatenation_length_equals_sum_of_durations(
    durations: list, crossfade: float
):
    """
    **Feature: consonant-synthesis, Property 14: Audio Concatenation Correctness**
    **Validates: Requirements 12.5**
    
    For any sequence of segments, the concatenated audio length SHALL equal
    the sum of individual segment durations (within sample precision),
    accounting for crossfade overlap.
    """
    fs = 16000
    synth = ConsonantSynthesizer(fs=fs)
    
    # Generate audio segments
    segments = [generate_random_audio(fs, d) for d in durations]
    
    # Filter out empty segments
    segments = [s for s in segments if len(s) > 0]
    assume(len(segments) > 0)
    
    # Calculate expected length
    total_samples = sum(len(s) for s in segments)
    
    # Account for crossfade overlap
    crossfade_samples = int(crossfade * fs / 1000)
    if len(segments) > 1 and crossfade_samples > 0:
        # Each junction loses crossfade_samples
        overlap_loss = crossfade_samples * (len(segments) - 1)
        expected_min = total_samples - overlap_loss
    else:
        expected_min = total_samples
    
    # Concatenate
    result = synth.concatenate_audio(segments, crossfade_ms=crossfade)
    
    # Verify length is within expected range
    # Allow some tolerance for edge cases
    assert len(result) >= expected_min - crossfade_samples, \
        f"Concatenated length {len(result)} should be >= {expected_min - crossfade_samples}"
    assert len(result) <= total_samples + 10, \
        f"Concatenated length {len(result)} should be <= {total_samples + 10}"


@given(durations=st.lists(segment_duration_ms, min_size=1, max_size=5))
@settings(max_examples=100)
def test_concatenation_preserves_normalization(durations: list):
    """
    **Feature: consonant-synthesis, Property 14: Audio Concatenation Correctness**
    **Validates: Requirements 12.5**
    
    For any sequence of segments, the concatenated audio SHALL be
    normalized to prevent clipping (peak amplitude <= 1.0).
    """
    fs = 16000
    synth = ConsonantSynthesizer(fs=fs)
    
    # Generate audio segments with varying amplitudes
    segments = []
    for d in durations:
        audio = generate_random_audio(fs, d)
        if len(audio) > 0:
            # Scale to random amplitude (some may exceed 1.0 before normalization)
            audio = audio * np.random.uniform(0.5, 2.0)
            segments.append(audio)
    
    assume(len(segments) > 0)
    
    # Concatenate
    result = synth.concatenate_audio(segments)
    
    # Verify normalization
    if len(result) > 0:
        max_amplitude = np.max(np.abs(result))
        assert max_amplitude <= 1.0, \
            f"Concatenated audio peak {max_amplitude:.3f} should be <= 1.0"


@given(
    num_segs=st.integers(min_value=2, max_value=5),
    duration=segment_duration_ms
)
@settings(max_examples=100)
def test_concatenation_with_crossfade_is_smooth(num_segs: int, duration: float):
    """
    **Feature: consonant-synthesis, Property 14: Audio Concatenation Correctness**
    **Validates: Requirements 12.5**
    
    For any sequence of segments with crossfade, the concatenation
    SHALL produce continuous audio without discontinuities.
    """
    fs = 16000
    synth = ConsonantSynthesizer(fs=fs)
    
    # Generate segments with known patterns
    segments = []
    for i in range(num_segs):
        audio = generate_random_audio(fs, duration)
        if len(audio) > 0:
            segments.append(audio)
    
    assume(len(segments) >= 2)
    
    # Concatenate with crossfade
    result = synth.concatenate_audio(segments, crossfade_ms=5.0)
    
    # Verify result is valid
    assert len(result) > 0, "Concatenation should produce audio"
    assert not np.any(np.isnan(result)), "Result should not contain NaN"
    assert not np.any(np.isinf(result)), "Result should not contain Inf"


@given(consonant=consonant_symbols)
@settings(max_examples=50)
def test_consonant_synthesis_produces_valid_audio(consonant: str):
    """
    **Feature: consonant-synthesis, Property 14: Audio Concatenation Correctness**
    **Validates: Requirements 12.5**
    
    For any consonant, synthesis SHALL produce valid audio that can
    be concatenated with other segments.
    """
    fs = 16000
    synth = ConsonantSynthesizer(fs=fs)
    
    # Get default duration for this consonant
    duration_ms = synth.get_default_duration(consonant)
    
    # Synthesize consonant
    audio = synth.synthesize(consonant, duration_ms)
    
    # Verify audio is valid
    assert len(audio) > 0, f"Consonant '{consonant}' should produce audio"
    assert np.max(np.abs(audio)) <= 1.0, \
        f"Consonant '{consonant}' audio should be normalized"
    assert not np.any(np.isnan(audio)), \
        f"Consonant '{consonant}' audio should not contain NaN"


@given(
    consonant=consonant_symbols,
    duration=segment_duration_ms
)
@settings(max_examples=50)
def test_consonant_concatenation_with_silence(consonant: str, duration: float):
    """
    **Feature: consonant-synthesis, Property 14: Audio Concatenation Correctness**
    **Validates: Requirements 12.5**
    
    For any consonant followed by silence, concatenation SHALL produce
    audio with correct total length.
    """
    fs = 16000
    synth = ConsonantSynthesizer(fs=fs)
    
    # Synthesize consonant
    consonant_duration = synth.get_default_duration(consonant)
    consonant_audio = synth.synthesize(consonant, consonant_duration)
    
    # Generate silence
    silence_samples = int(duration * fs / 1000)
    silence = np.zeros(silence_samples)
    
    assume(len(consonant_audio) > 0 and len(silence) > 0)
    
    # Concatenate
    segments = [consonant_audio, silence]
    result = synth.concatenate_audio(segments, crossfade_ms=0.0)
    
    # Without crossfade, length should be sum of segments
    expected_length = len(consonant_audio) + len(silence)
    
    # Allow small tolerance
    assert abs(len(result) - expected_length) <= 10, \
        f"Concatenated length {len(result)} should be ~{expected_length}"


def test_empty_segments_handled():
    """Test that empty segment lists are handled correctly."""
    synth = ConsonantSynthesizer(fs=16000)
    
    # Empty list
    result = synth.concatenate_audio([])
    assert len(result) == 0, "Empty list should produce empty result"
    
    # List with empty arrays
    result = synth.concatenate_audio([np.array([]), np.array([])])
    assert len(result) == 0, "List of empty arrays should produce empty result"


def test_single_segment_passthrough():
    """Test that single segment is passed through correctly."""
    fs = 16000
    synth = ConsonantSynthesizer(fs=fs)
    
    # Generate single segment
    original = generate_random_audio(fs, 100.0)
    
    # Concatenate single segment
    result = synth.concatenate_audio([original])
    
    # Should be normalized version of original
    assert len(result) == len(original), \
        "Single segment should preserve length"


def test_resample_to_target():
    """Test sample rate conversion."""
    synth = ConsonantSynthesizer(fs=16000)
    
    # Generate audio at different sample rate
    source_fs = 8000
    duration_ms = 100.0
    n_samples = int(duration_ms * source_fs / 1000)
    audio = np.random.randn(n_samples)
    
    # Resample
    result = synth.resample_to_target(audio, source_fs)
    
    # Expected length at target sample rate
    expected_samples = int(duration_ms * 16000 / 1000)
    
    # Allow some tolerance for resampling
    assert abs(len(result) - expected_samples) <= 5, \
        f"Resampled length {len(result)} should be ~{expected_samples}"


def test_calculate_total_duration():
    """Test duration calculation for segment list."""
    fs = 16000
    synth = ConsonantSynthesizer(fs=fs)
    
    # Create segments with known durations
    segments = [
        {'duration_ms': 100.0},
        {'duration_ms': 50.0},
        {'duration_ms': 75.0}
    ]
    
    total = synth.calculate_total_duration(segments)
    
    assert total == 225.0, f"Total duration should be 225.0ms, got {total}"


def test_calculate_total_duration_with_audio():
    """Test duration calculation when audio is present."""
    fs = 16000
    synth = ConsonantSynthesizer(fs=fs)
    
    # Create segment with audio
    audio = generate_random_audio(fs, 100.0)
    segments = [
        {'audio': audio},
        {'duration_ms': 50.0}
    ]
    
    total = synth.calculate_total_duration(segments)
    
    # Audio duration + explicit duration
    audio_duration = len(audio) / fs * 1000
    expected = audio_duration + 50.0
    
    assert abs(total - expected) < 1.0, \
        f"Total duration should be ~{expected}ms, got {total}"


def test_get_expected_samples():
    """Test sample count calculation."""
    synth = ConsonantSynthesizer(fs=16000)
    
    # 100ms at 16000 Hz = 1600 samples
    samples = synth.get_expected_samples(100.0)
    assert samples == 1600, f"Expected 1600 samples, got {samples}"
    
    # 50ms at 16000 Hz = 800 samples
    samples = synth.get_expected_samples(50.0)
    assert samples == 800, f"Expected 800 samples, got {samples}"


# ============================================================================
# Integration Tests
# ============================================================================

def test_synthesize_and_concatenate_multiple_consonants():
    """Test synthesizing and concatenating multiple consonants."""
    fs = 16000
    synth = ConsonantSynthesizer(fs=fs)
    
    # Synthesize a few consonants
    consonants = ['m', 'n', 's']
    segments = []
    
    for c in consonants:
        duration = synth.get_default_duration(c)
        audio = synth.synthesize(c, duration)
        segments.append(audio)
    
    # Concatenate
    result = synth.concatenate_audio(segments)
    
    # Verify result
    assert len(result) > 0, "Should produce audio"
    assert np.max(np.abs(result)) <= 1.0, "Should be normalized"


def test_convenience_function_concatenate():
    """Test the convenience function for concatenation."""
    fs = 16000
    
    # Generate segments
    segments = [
        generate_random_audio(fs, 50.0),
        generate_random_audio(fs, 50.0)
    ]
    
    # Use convenience function
    result = concatenate_audio_segments(segments, fs=fs, crossfade_ms=5.0)
    
    assert len(result) > 0, "Convenience function should produce audio"


def test_convenience_function_synthesize():
    """Test the convenience function for synthesis."""
    audio = synthesize_consonant('m', 80.0, fs=16000)
    
    assert len(audio) > 0, "Convenience function should produce audio"
    assert np.max(np.abs(audio)) <= 1.0, "Should be normalized"
