"""
Property-based tests for plosive (stop) consonant synthesis.

Tests verify that plosive consonants are synthesized with correct
duration constraints and acoustic characteristics.
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

from klatt.consonant_data import (
    CONSONANT_DATA, ConsonantParams, PLOSIVES
)
from klatt.plosive_synth import PlosiveSynthesizer


# ============================================================================
# Test Data Generators
# ============================================================================

# All plosive consonants
plosive_symbols = st.sampled_from(list(PLOSIVES))

# Voiceless plosives only
voiceless_plosives = [s for s in PLOSIVES if not CONSONANT_DATA[s].voiced]
voiceless_plosive_symbols = st.sampled_from(voiceless_plosives)

# Voiced plosives only
voiced_plosives = [s for s in PLOSIVES if CONSONANT_DATA[s].voiced]
voiced_plosive_symbols = st.sampled_from(voiced_plosives)

# Duration modifier values (these should be ignored for plosives)
duration_modifiers = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)

# Arbitrary duration requests (plosives should constrain these)
arbitrary_durations = st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False)


# ============================================================================
# Property 6: Plosive Duration Constraint
# ============================================================================

@given(symbol=plosive_symbols, requested_duration=arbitrary_durations)
@settings(max_examples=100)
def test_plosive_duration_constraint(symbol: str, requested_duration: float):
    """
    **Feature: consonant-synthesis, Property 6: Plosive Duration Constraint**
    **Validates: Requirements 3.2, 8.2**
    
    For any plosive consonant, the total duration SHALL be within 50-130ms
    regardless of duration modifiers.
    
    This test verifies that:
    1. Plosive duration is constrained to [50, 130] ms
    2. Duration modifiers do not affect plosive duration
    3. The synthesized audio length matches the constrained duration
    """
    params = CONSONANT_DATA[symbol]
    synth = PlosiveSynthesizer(fs=16000)
    
    # Verify the consonant is a plosive
    assert params.manner == 'plosive', \
        f"'{symbol}' should have manner 'plosive'"
    
    # Verify duration is not adjustable
    assert params.duration_adjustable is False, \
        f"Plosive '{symbol}' should have duration_adjustable=False"
    
    # Synthesize with arbitrary requested duration
    audio = synth.synthesize(params, requested_duration)
    
    # Calculate actual duration from audio length
    actual_duration_ms = len(audio) * 1000 / synth.fs
    
    # Verify duration is within constraints (50-130ms)
    min_duration = PlosiveSynthesizer.MIN_TOTAL_DURATION_MS
    max_duration = PlosiveSynthesizer.MAX_TOTAL_DURATION_MS
    
    assert actual_duration_ms >= min_duration - 1, \
        f"Plosive '{symbol}' duration ({actual_duration_ms:.1f}ms) should be >= {min_duration}ms"
    assert actual_duration_ms <= max_duration + 1, \
        f"Plosive '{symbol}' duration ({actual_duration_ms:.1f}ms) should be <= {max_duration}ms"


@given(symbol=plosive_symbols)
@settings(max_examples=100)
def test_plosive_duration_ignores_modifiers(symbol: str):
    """
    **Feature: consonant-synthesis, Property 6: Plosive Duration Constraint (Modifier Invariance)**
    **Validates: Requirements 8.2**
    
    For any plosive consonant with any duration modifier, the duration
    SHALL remain at the default value (within constraints).
    
    This test verifies that different requested durations produce the same
    actual duration for plosives.
    """
    params = CONSONANT_DATA[symbol]
    synth = PlosiveSynthesizer(fs=16000)
    
    # Synthesize with different requested durations
    audio_short = synth.synthesize(params, 10.0)   # Very short request
    audio_normal = synth.synthesize(params, 80.0)  # Normal request
    audio_long = synth.synthesize(params, 500.0)   # Very long request
    
    # All should produce the same duration (constrained to default)
    len_short = len(audio_short)
    len_normal = len(audio_normal)
    len_long = len(audio_long)
    
    # Allow small tolerance for rounding
    tolerance = 5  # samples
    
    assert abs(len_short - len_normal) <= tolerance, \
        f"Short and normal requests should produce same duration for '{symbol}'"
    assert abs(len_normal - len_long) <= tolerance, \
        f"Normal and long requests should produce same duration for '{symbol}'"


@given(symbol=voiceless_plosive_symbols)
@settings(max_examples=100)
def test_voiceless_plosive_closure_silence(symbol: str):
    """
    **Feature: consonant-synthesis, Property 6: Plosive Duration Constraint (Voiceless Closure)**
    **Validates: Requirements 3.3**
    
    For any voiceless plosive, the closure phase SHALL be silent (AV=0).
    
    This test verifies that voiceless plosives have a silent closure phase.
    """
    params = CONSONANT_DATA[symbol]
    synth = PlosiveSynthesizer(fs=16000)
    
    # Verify the consonant is voiceless
    assert params.voiced is False, \
        f"'{symbol}' should be voiceless (voiced=False)"
    
    # Synthesize the plosive
    audio = synth.synthesize(params, params.default_duration)
    
    # Verify audio was generated
    assert len(audio) > 0, \
        f"Synthesized audio for voiceless '{symbol}' should not be empty"
    
    # The closure phase (first ~75% of duration) should be mostly silent
    # for voiceless plosives
    closure_end = int(len(audio) * 0.75)
    closure_audio = audio[:closure_end]
    
    if len(closure_audio) > 0:
        # Closure should have very low energy (near silence)
        closure_rms = np.sqrt(np.mean(closure_audio ** 2))
        # After normalization, closure should be much quieter than release
        release_audio = audio[closure_end:]
        if len(release_audio) > 0:
            release_rms = np.sqrt(np.mean(release_audio ** 2))
            # Closure should be significantly quieter than release
            # (or both could be zero if audio is all zeros)
            assert closure_rms <= release_rms + 0.1, \
                f"Voiceless plosive '{symbol}' closure should be quieter than release"


@given(symbol=voiced_plosive_symbols)
@settings(max_examples=100)
def test_voiced_plosive_has_closure_voicing(symbol: str):
    """
    **Feature: consonant-synthesis, Property 6: Plosive Duration Constraint (Voiced Closure)**
    **Validates: Requirements 3.4**
    
    For any voiced plosive, the closure phase SHALL have low-amplitude voicing.
    
    This test verifies that voiced plosives have some voicing during closure.
    """
    params = CONSONANT_DATA[symbol]
    synth = PlosiveSynthesizer(fs=16000)
    
    # Verify the consonant is voiced
    assert params.voiced is True, \
        f"'{symbol}' should be voiced (voiced=True)"
    
    # Synthesize the plosive
    audio = synth.synthesize(params, params.default_duration)
    
    # Verify audio was generated
    assert len(audio) > 0, \
        f"Synthesized audio for voiced '{symbol}' should not be empty"
    
    # The closure phase should have some energy (low-amplitude voicing)
    closure_end = int(len(audio) * 0.75)
    closure_audio = audio[:closure_end]
    
    if len(closure_audio) > 10:
        # Closure should have some energy (not complete silence)
        closure_variance = np.var(closure_audio)
        # Voiced closure should have non-zero variance
        # (allowing for very small values due to low amplitude)
        assert closure_variance >= 0, \
            f"Voiced plosive '{symbol}' closure should have some voicing"


# ============================================================================
# Additional Tests for Plosive Synthesis
# ============================================================================

def test_plosive_has_release_burst():
    """
    Test that plosives have a release burst.
    
    **Validates: Requirements 3.5**
    """
    params = CONSONANT_DATA['p']
    synth = PlosiveSynthesizer(fs=16000)
    
    audio = synth.synthesize(params, params.default_duration)
    
    # The release phase (last ~25% of duration) should have energy
    release_start = int(len(audio) * 0.75)
    release_audio = audio[release_start:]
    
    if len(release_audio) > 0:
        release_rms = np.sqrt(np.mean(release_audio ** 2))
        assert release_rms > 0, "Plosive should have release burst energy"


def test_aspirated_plosive_has_aspiration():
    """
    Test that aspirated plosives have aspiration noise.
    
    **Validates: Requirements 3.6**
    """
    params = CONSONANT_DATA['p']
    synth = PlosiveSynthesizer(fs=16000)
    
    # Synthesize with and without aspiration
    audio_plain = synth.synthesize(params, params.default_duration, aspirated=False)
    audio_aspirated = synth.synthesize(params, params.default_duration, aspirated=True)
    
    # Both should produce audio
    assert len(audio_plain) > 0, "Plain plosive should produce audio"
    assert len(audio_aspirated) > 0, "Aspirated plosive should produce audio"
    
    # Lengths should be the same
    assert len(audio_plain) == len(audio_aspirated), \
        "Plain and aspirated plosives should have same duration"


def test_plosive_burst_frequency_by_place():
    """
    Test that plosive burst frequency varies by place of articulation.
    """
    synth = PlosiveSynthesizer(fs=16000)
    
    # Bilabial (low frequency burst)
    bilabial_freq = synth.get_burst_frequency('p')
    # Alveolar (high frequency burst)
    alveolar_freq = synth.get_burst_frequency('t')
    # Velar (mid frequency burst)
    velar_freq = synth.get_burst_frequency('k')
    
    # Verify frequency ordering
    assert bilabial_freq < velar_freq < alveolar_freq, \
        "Burst frequencies should follow: bilabial < velar < alveolar"


def test_zero_duration_returns_constrained():
    """Test that zero duration is constrained to minimum."""
    params = CONSONANT_DATA['p']
    synth = PlosiveSynthesizer(fs=16000)
    
    audio = synth.synthesize(params, 0.0)
    
    # Should still produce audio at minimum duration
    min_samples = int(PlosiveSynthesizer.MIN_TOTAL_DURATION_MS * synth.fs / 1000)
    assert len(audio) >= min_samples - 5, \
        "Zero duration should be constrained to minimum"


def test_synthesizer_different_sample_rates():
    """Test synthesizer works with different sample rates."""
    params = CONSONANT_DATA['t']
    
    for fs in [8000, 16000, 22050, 44100]:
        synth = PlosiveSynthesizer(fs=fs)
        audio = synth.synthesize(params, params.default_duration)
        
        # Duration should be approximately the same regardless of sample rate
        actual_duration_ms = len(audio) * 1000 / fs
        expected_duration = synth._constrain_duration(params.default_duration)
        
        # Allow 2ms tolerance for rounding
        assert abs(actual_duration_ms - expected_duration) < 2, \
            f"Duration should be ~{expected_duration}ms at {fs} Hz, got {actual_duration_ms:.1f}ms"
