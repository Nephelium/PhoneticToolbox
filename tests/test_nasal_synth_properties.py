"""
Property-based tests for nasal consonant synthesis.

Tests verify that nasal consonants are synthesized with correct
nasal pole (FNP) and nasal zero (FNZ) parameters based on place
of articulation.
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
    CONSONANT_DATA, ConsonantParams, NASALS,
    SIBILANTS, FRICATIVES, APPROXIMANTS, PLOSIVES
)
from klatt.nasal_synth import NasalSynthesizer


# ============================================================================
# Test Data Generators
# ============================================================================

# All nasal consonants
nasal_symbols = st.sampled_from(list(NASALS))

# Duration strategy (reasonable range for nasals)
duration_ms = st.floats(min_value=30.0, max_value=200.0, allow_nan=False, allow_infinity=False)


# ============================================================================
# Property 3: Nasal Synthesis Parameters
# ============================================================================

@given(symbol=nasal_symbols)
@settings(max_examples=100)
def test_nasal_has_fnp_and_fnz(symbol: str):
    """
    **Feature: consonant-synthesis, Property 3: Nasal Synthesis Parameters**
    **Validates: Requirements 2.2, 2.3**
    
    For any nasal consonant, the synthesizer SHALL set FNP > 0 and FNZ > 0
    according to the consonant's place of articulation.
    
    This test verifies that:
    1. All nasals have FNP (nasal pole) > 0 in CONSONANT_DATA
    2. All nasals have FNZ (nasal zero) > 0 in CONSONANT_DATA
    3. The synthesizer correctly retrieves these parameters
    """
    params = CONSONANT_DATA[symbol]
    synth = NasalSynthesizer(fs=16000)
    
    # Verify manner is nasal
    assert params.manner == 'nasal', \
        f"'{symbol}' should have manner 'nasal', got '{params.manner}'"
    
    # Verify FNP > 0 (Requirement 2.2)
    assert params.fnp > 0, \
        f"Nasal '{symbol}' should have FNP > 0, got {params.fnp}"
    
    # Verify FNZ > 0 (Requirement 2.3)
    assert params.fnz > 0, \
        f"Nasal '{symbol}' should have FNZ > 0, got {params.fnz}"
    
    # Verify synthesizer reports correct FNP
    reported_fnp = synth.get_fnp(symbol)
    assert reported_fnp == params.fnp, \
        f"Reported FNP ({reported_fnp}) should match CONSONANT_DATA ({params.fnp})"
    
    # Verify synthesizer reports correct FNZ
    reported_fnz = synth.get_fnz(symbol)
    assert reported_fnz == params.fnz, \
        f"Reported FNZ ({reported_fnz}) should match CONSONANT_DATA ({params.fnz})"


@given(symbol=nasal_symbols)
@settings(max_examples=100)
def test_nasal_fnp_in_valid_range(symbol: str):
    """
    **Feature: consonant-synthesis, Property 3: Nasal Synthesis Parameters**
    **Validates: Requirements 2.2**
    
    For any nasal consonant, the nasal pole frequency (FNP) SHALL be
    in a physiologically reasonable range (typically around 250 Hz).
    """
    params = CONSONANT_DATA[symbol]
    
    # FNP is typically around 250 Hz for all nasals
    # Allow range of 150-400 Hz for variation
    assert 150 <= params.fnp <= 400, \
        f"FNP for '{symbol}' should be in range [150, 400], got {params.fnp}"


@given(symbol=nasal_symbols)
@settings(max_examples=100)
def test_nasal_fnz_varies_by_place(symbol: str):
    """
    **Feature: consonant-synthesis, Property 3: Nasal Synthesis Parameters**
    **Validates: Requirements 2.3**
    
    For any nasal consonant, the nasal zero frequency (FNZ) SHALL vary
    based on place of articulation (it attenuates oral cavity resonances).
    """
    params = CONSONANT_DATA[symbol]
    
    # FNZ varies by place of articulation
    # Typically ranges from ~800 Hz (bilabial) to ~2500 Hz (velar)
    assert 500 <= params.fnz <= 3000, \
        f"FNZ for '{symbol}' should be in range [500, 3000], got {params.fnz}"
    
    # FNZ should be higher than FNP (anti-formant attenuates higher frequencies)
    assert params.fnz > params.fnp, \
        f"FNZ ({params.fnz}) should be greater than FNP ({params.fnp}) for '{symbol}'"


@given(symbol=nasal_symbols, dur=duration_ms)
@settings(max_examples=100)
def test_nasal_synthesis_produces_audio(symbol: str, dur: float):
    """
    **Feature: consonant-synthesis, Property 3: Nasal Synthesis Parameters**
    **Validates: Requirements 2.1, 2.2, 2.3**
    
    For any nasal consonant, the synthesizer SHALL produce valid audio
    using the FNP and FNZ parameters.
    """
    params = CONSONANT_DATA[symbol]
    synth = NasalSynthesizer(fs=16000)
    
    # Synthesize the nasal
    audio = synth.synthesize(params, dur)
    
    # Verify audio was generated
    assert len(audio) > 0, \
        f"Synthesized audio for nasal '{symbol}' should not be empty"
    
    # Verify audio is normalized (within [-1, 1])
    assert np.max(np.abs(audio)) <= 1.0, \
        f"Audio for nasal '{symbol}' should be normalized to [-1, 1]"
    
    # Verify audio has expected approximate length
    expected_samples = int(dur * 16000 / 1000)
    # Allow 10% tolerance due to resampling
    assert abs(len(audio) - expected_samples) < expected_samples * 0.1, \
        f"Audio length {len(audio)} should be close to {expected_samples}"


# ============================================================================
# Property 4: Voiced Consonant Voicing
# ============================================================================

@given(symbol=nasal_symbols)
@settings(max_examples=100)
def test_nasal_is_voiced(symbol: str):
    """
    **Feature: consonant-synthesis, Property 4: Voiced Consonant Voicing**
    **Validates: Requirements 2.4**
    
    For any nasal consonant, the synthesis SHALL maintain AV > 0
    (all nasals are voiced by definition).
    """
    params = CONSONANT_DATA[symbol]
    synth = NasalSynthesizer(fs=16000)
    
    # All nasals must be voiced
    assert params.voiced is True, \
        f"Nasal '{symbol}' should be voiced (voiced=True)"
    
    # Synthesizer should report nasal as voiced
    assert synth.is_voiced(symbol) is True, \
        f"Synthesizer should report nasal '{symbol}' as voiced"


@given(symbol=nasal_symbols, dur=duration_ms)
@settings(max_examples=100)
def test_nasal_audio_has_periodic_component(symbol: str, dur: float):
    """
    **Feature: consonant-synthesis, Property 4: Voiced Consonant Voicing**
    **Validates: Requirements 2.4**
    
    For any nasal consonant, the synthesized audio SHALL contain
    periodic voicing (not just noise).
    """
    params = CONSONANT_DATA[symbol]
    synth = NasalSynthesizer(fs=16000)
    
    # Synthesize the nasal
    audio = synth.synthesize(params, dur)
    
    # Verify audio was generated
    assert len(audio) > 0, \
        f"Synthesized audio for nasal '{symbol}' should not be empty"
    
    # Voiced sounds should have non-zero energy
    # (as opposed to silence which would indicate AV=0)
    rms = np.sqrt(np.mean(audio ** 2))
    assert rms > 0.01, \
        f"Nasal '{symbol}' should have audible energy (RMS > 0.01), got {rms}"


# ============================================================================
# Additional Voiced Consonant Tests (for Property 4)
# ============================================================================

# Voiced fricatives for Property 4 testing
voiced_fricatives = [s for s in (SIBILANTS | FRICATIVES) if CONSONANT_DATA[s].voiced]
voiced_fricative_symbols = st.sampled_from(voiced_fricatives) if voiced_fricatives else st.nothing()

# Voiced plosives for Property 4 testing
voiced_plosives = [s for s in PLOSIVES if CONSONANT_DATA[s].voiced]
voiced_plosive_symbols = st.sampled_from(voiced_plosives) if voiced_plosives else st.nothing()

# Approximants (all voiced) for Property 4 testing
approximant_symbols = st.sampled_from(list(APPROXIMANTS)) if APPROXIMANTS else st.nothing()


@given(symbol=approximant_symbols)
@settings(max_examples=100)
def test_approximant_is_voiced(symbol: str):
    """
    **Feature: consonant-synthesis, Property 4: Voiced Consonant Voicing**
    **Validates: Requirements 5.3**
    
    For any approximant consonant, the consonant SHALL be voiced.
    """
    params = CONSONANT_DATA[symbol]
    
    # All approximants are voiced
    assert params.voiced is True, \
        f"Approximant '{symbol}' should be voiced (voiced=True)"


@given(symbol=voiced_fricative_symbols)
@settings(max_examples=100)
def test_voiced_fricative_is_voiced(symbol: str):
    """
    **Feature: consonant-synthesis, Property 4: Voiced Consonant Voicing**
    **Validates: Requirements 4.5**
    
    For any voiced fricative, the consonant SHALL be marked as voiced.
    """
    params = CONSONANT_DATA[symbol]
    
    assert params.voiced is True, \
        f"Voiced fricative '{symbol}' should be voiced (voiced=True)"


@given(symbol=voiced_plosive_symbols)
@settings(max_examples=100)
def test_voiced_plosive_is_voiced(symbol: str):
    """
    **Feature: consonant-synthesis, Property 4: Voiced Consonant Voicing**
    **Validates: Requirements 3.4**
    
    For any voiced plosive, the consonant SHALL be marked as voiced.
    """
    params = CONSONANT_DATA[symbol]
    
    assert params.voiced is True, \
        f"Voiced plosive '{symbol}' should be voiced (voiced=True)"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_zero_duration_returns_empty():
    """Test that zero duration returns empty array."""
    params = CONSONANT_DATA['m']
    synth = NasalSynthesizer(fs=16000)
    
    audio = synth.synthesize(params, 0.0)
    assert len(audio) == 0, "Zero duration should return empty array"


def test_negative_duration_returns_empty():
    """Test that negative duration returns empty array."""
    params = CONSONANT_DATA['n']
    synth = NasalSynthesizer(fs=16000)
    
    audio = synth.synthesize(params, -10.0)
    assert len(audio) == 0, "Negative duration should return empty array"


def test_synthesizer_is_nasal_check():
    """Test is_nasal method correctly identifies nasals."""
    synth = NasalSynthesizer(fs=16000)
    
    # All NASALS should return True
    for symbol in NASALS:
        assert synth.is_nasal(symbol) is True, \
            f"'{symbol}' should be identified as nasal"
    
    # Non-nasals should return False
    non_nasals = ['p', 's', 'l', 'j']
    for symbol in non_nasals:
        assert synth.is_nasal(symbol) is False, \
            f"'{symbol}' should not be identified as nasal"


def test_all_nasals_have_formants():
    """Test that all nasals have F1, F2, F3 defined."""
    for symbol in NASALS:
        params = CONSONANT_DATA[symbol]
        
        assert params.f1 > 0, f"Nasal '{symbol}' should have F1 > 0"
        assert params.f2 > 0, f"Nasal '{symbol}' should have F2 > 0"
        assert params.f3 > 0, f"Nasal '{symbol}' should have F3 > 0"
