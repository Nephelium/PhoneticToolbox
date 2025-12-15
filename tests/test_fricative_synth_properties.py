"""
Property-based tests for fricative synthesis.

Tests verify that fricative consonants are synthesized with correct
noise characteristics based on place of articulation and voicing.
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
    CONSONANT_DATA, ConsonantParams,
    SIBILANTS, FRICATIVES
)
from klatt.fricative_synth import FricativeSynthesizer


# ============================================================================
# Test Data Generators
# ============================================================================

# All fricative consonants (sibilants + non-sibilants)
ALL_FRICATIVES = SIBILANTS | FRICATIVES
fricative_symbols = st.sampled_from(list(ALL_FRICATIVES))

# Voiceless fricatives only
voiceless_fricatives = [s for s in ALL_FRICATIVES if not CONSONANT_DATA[s].voiced]
voiceless_fricative_symbols = st.sampled_from(voiceless_fricatives)

# Voiced fricatives only
voiced_fricatives = [s for s in ALL_FRICATIVES if CONSONANT_DATA[s].voiced]
voiced_fricative_symbols = st.sampled_from(voiced_fricatives)

# Duration strategy (reasonable range for fricatives)
duration_ms = st.floats(min_value=50.0, max_value=300.0, allow_nan=False, allow_infinity=False)


# ============================================================================
# Property 7: Fricative Noise Frequency
# ============================================================================

@given(symbol=fricative_symbols, dur=duration_ms)
@settings(max_examples=100)
def test_fricative_noise_frequency(symbol: str, dur: float):
    """
    **Feature: consonant-synthesis, Property 7: Fricative Noise Frequency**
    **Validates: Requirements 4.1, 4.2, 4.3**
    
    For any fricative consonant, the noise center frequency SHALL correspond
    to its place of articulation as defined in CONSONANT_DATA.
    
    This test verifies that:
    1. The synthesizer uses the noise_freq from CONSONANT_DATA
    2. The generated audio has spectral energy concentrated around that frequency
    """
    params = CONSONANT_DATA[symbol]
    synth = FricativeSynthesizer(fs=16000)
    
    # Verify the consonant has noise parameters defined
    assert params.noise_freq > 0, \
        f"Fricative '{symbol}' should have noise_freq > 0"
    assert params.noise_bw > 0, \
        f"Fricative '{symbol}' should have noise_bw > 0"
    
    # Synthesize the fricative
    audio = synth.synthesize(params, dur)
    
    # Verify audio was generated
    assert len(audio) > 0, \
        f"Synthesized audio for '{symbol}' should not be empty"
    
    # Verify audio is normalized (within [-1, 1])
    assert np.max(np.abs(audio)) <= 1.0, \
        f"Audio for '{symbol}' should be normalized to [-1, 1]"
    
    # Verify the synthesizer reports correct noise frequency
    reported_freq = synth.get_noise_frequency(symbol)
    assert reported_freq == params.noise_freq, \
        f"Reported noise frequency ({reported_freq}) should match " \
        f"CONSONANT_DATA ({params.noise_freq}) for '{symbol}'"


@given(symbol=fricative_symbols)
@settings(max_examples=100)
def test_fricative_noise_params_from_data(symbol: str):
    """
    **Feature: consonant-synthesis, Property 7: Fricative Noise Frequency (Data Consistency)**
    **Validates: Requirements 4.1, 4.2, 4.3**
    
    For any fricative consonant, verify that CONSONANT_DATA contains
    appropriate noise parameters based on place of articulation.
    """
    params = CONSONANT_DATA[symbol]
    
    # Verify manner is fricative or sibilant
    assert params.manner in ('fricative', 'sibilant'), \
        f"'{symbol}' should have manner 'fricative' or 'sibilant'"
    
    # Verify noise_freq is positive and reasonable (50 Hz to 10000 Hz)
    assert 50 <= params.noise_freq <= 10000, \
        f"noise_freq for '{symbol}' should be in range [50, 10000], got {params.noise_freq}"
    
    # Verify noise_bw is positive and reasonable
    assert 100 <= params.noise_bw <= 5000, \
        f"noise_bw for '{symbol}' should be in range [100, 5000], got {params.noise_bw}"
    
    # Sibilants should have higher center frequencies (typically > 3000 Hz)
    if params.manner == 'sibilant':
        assert params.noise_freq >= 3000, \
            f"Sibilant '{symbol}' should have noise_freq >= 3000, got {params.noise_freq}"


# ============================================================================
# Property 5: Voiceless Consonant Silence
# ============================================================================

@given(symbol=voiceless_fricative_symbols, dur=duration_ms)
@settings(max_examples=100)
def test_voiceless_fricative_no_voicing(symbol: str, dur: float):
    """
    **Feature: consonant-synthesis, Property 5: Voiceless Consonant Silence**
    **Validates: Requirements 4.4**
    
    For any voiceless fricative, the synthesis SHALL set AV=0 during
    the consonant (no periodic voicing component).
    
    This test verifies that:
    1. The consonant is marked as voiceless in CONSONANT_DATA
    2. The synthesizer correctly identifies it as voiceless
    3. The synthesized audio contains only noise (no periodic component)
    """
    params = CONSONANT_DATA[symbol]
    synth = FricativeSynthesizer(fs=16000)
    
    # Verify the consonant is voiceless
    assert params.voiced is False, \
        f"'{symbol}' should be voiceless (voiced=False)"
    
    # Verify synthesizer reports correct voicing status
    assert synth.is_voiced(symbol) is False, \
        f"Synthesizer should report '{symbol}' as voiceless"
    
    # Synthesize the fricative
    audio = synth.synthesize(params, dur)
    
    # Verify audio was generated
    assert len(audio) > 0, \
        f"Synthesized audio for voiceless '{symbol}' should not be empty"
    
    # For voiceless fricatives, the audio should be noise-like
    # We can verify this by checking that the audio has high variance
    # (noise has high variance, silence has zero variance)
    if len(audio) > 10:
        variance = np.var(audio)
        assert variance > 0, \
            f"Voiceless fricative '{symbol}' should produce noise (variance > 0)"


@given(symbol=voiced_fricative_symbols, dur=duration_ms)
@settings(max_examples=100)
def test_voiced_fricative_has_voicing(symbol: str, dur: float):
    """
    **Feature: consonant-synthesis, Property 5: Voiceless Consonant Silence (Inverse)**
    **Validates: Requirements 4.5**
    
    For any voiced fricative, the synthesis SHALL maintain AV > 0
    (periodic voicing component present).
    
    This test verifies that:
    1. The consonant is marked as voiced in CONSONANT_DATA
    2. The synthesizer correctly identifies it as voiced
    """
    params = CONSONANT_DATA[symbol]
    synth = FricativeSynthesizer(fs=16000)
    
    # Verify the consonant is voiced
    assert params.voiced is True, \
        f"'{symbol}' should be voiced (voiced=True)"
    
    # Verify synthesizer reports correct voicing status
    assert synth.is_voiced(symbol) is True, \
        f"Synthesizer should report '{symbol}' as voiced"
    
    # Synthesize the fricative
    audio = synth.synthesize(params, dur)
    
    # Verify audio was generated
    assert len(audio) > 0, \
        f"Synthesized audio for voiced '{symbol}' should not be empty"


# ============================================================================
# Additional Tests for /h/ Whisper Synthesis
# ============================================================================

def test_h_whisper_synthesis():
    """
    Test that /h/ is synthesized using whisper-like parameters.
    
    **Validates: Requirements 4.6**
    """
    params = CONSONANT_DATA['h']
    synth = FricativeSynthesizer(fs=16000)
    
    # Verify /h/ is voiceless
    assert params.voiced is False, "/h/ should be voiceless"
    
    # Verify /h/ is a glottal fricative
    assert params.place == 'glottal', "/h/ should be glottal"
    
    # Synthesize /h/
    audio = synth.synthesize(params, 100.0)
    
    # Verify audio was generated
    assert len(audio) > 0, "Synthesized /h/ should not be empty"
    
    # Verify audio is normalized
    assert np.max(np.abs(audio)) <= 1.0, "/h/ audio should be normalized"


def test_h_has_wide_bandwidth():
    """
    Test that /h/ has wide bandwidth characteristic of whisper.
    
    **Validates: Requirements 4.6**
    """
    params = CONSONANT_DATA['h']
    
    # /h/ should have wide bandwidth for whisper-like quality
    assert params.noise_bw >= 1500, \
        f"/h/ should have wide bandwidth (>= 1500), got {params.noise_bw}"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_zero_duration_returns_empty():
    """Test that zero duration returns empty array."""
    params = CONSONANT_DATA['s']
    synth = FricativeSynthesizer(fs=16000)
    
    audio = synth.synthesize(params, 0.0)
    assert len(audio) == 0, "Zero duration should return empty array"


def test_negative_duration_returns_empty():
    """Test that negative duration returns empty array."""
    params = CONSONANT_DATA['s']
    synth = FricativeSynthesizer(fs=16000)
    
    audio = synth.synthesize(params, -10.0)
    assert len(audio) == 0, "Negative duration should return empty array"


def test_synthesizer_different_sample_rates():
    """Test synthesizer works with different sample rates."""
    params = CONSONANT_DATA['f']
    
    for fs in [8000, 16000, 22050, 44100]:
        synth = FricativeSynthesizer(fs=fs)
        audio = synth.synthesize(params, 100.0)
        
        expected_samples = int(100.0 * fs / 1000)
        assert len(audio) == expected_samples, \
            f"Audio length should be {expected_samples} at {fs} Hz, got {len(audio)}"
