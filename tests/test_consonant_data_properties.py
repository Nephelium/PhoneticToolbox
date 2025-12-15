"""
Property-based tests for consonant data completeness.

Tests verify that all consonants in CONSONANT_DATA have the required
acoustic parameters as specified in the design document.
"""
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from klatt.consonant_data import (
    CONSONANT_DATA, ConsonantParams,
    NASALS, PLOSIVES, SIBILANTS, FRICATIVES,
    APPROXIMANTS, TAPS, TRILLS,
    LATERAL_FRICATIVES, LATERAL_APPROXIMANTS, LATERAL_FLAPS,
    ALL_CONSONANTS, DURATION_ADJUSTABLE, DURATION_FIXED
)


# ============================================================================
# Test Data Generators
# ============================================================================

# Strategy for sampling any consonant from CONSONANT_DATA
consonant_symbols = st.sampled_from(list(CONSONANT_DATA.keys()))

# Strategy for sampling nasal consonants
nasal_symbols = st.sampled_from(list(NASALS))

# Strategy for sampling fricative consonants (sibilants + non-sibilants)
all_fricatives = SIBILANTS | FRICATIVES
fricative_symbols = st.sampled_from(list(all_fricatives))

# Strategy for sampling resonant consonants (those with formant structure)
resonant_consonants = NASALS | APPROXIMANTS | TAPS | TRILLS | LATERAL_APPROXIMANTS | LATERAL_FLAPS
resonant_symbols = st.sampled_from(list(resonant_consonants))


# ============================================================================
# Property 11: Consonant Data Completeness
# ============================================================================

@given(symbol=consonant_symbols)
@settings(max_examples=100)
def test_consonant_data_completeness(symbol: str):
    """
    **Feature: consonant-synthesis, Property 11: Consonant Data Completeness**
    **Validates: Requirements 11.1, 11.4, 11.5**
    
    For any consonant in CONSONANT_DATA, the entry SHALL include:
    - F1-F3 (for resonant consonants)
    - voiced attribute
    - default_duration
    - duration_adjustable flag
    """
    params = CONSONANT_DATA[symbol]
    
    # Verify it's a ConsonantParams instance
    assert isinstance(params, ConsonantParams), \
        f"Entry for '{symbol}' should be a ConsonantParams instance"
    
    # Verify symbol matches
    assert params.symbol == symbol, \
        f"Symbol mismatch: expected '{symbol}', got '{params.symbol}'"
    
    # Verify voiced attribute is a boolean
    assert isinstance(params.voiced, bool), \
        f"voiced attribute for '{symbol}' should be a boolean"
    
    # Verify default_duration is positive
    assert params.default_duration > 0, \
        f"default_duration for '{symbol}' should be positive, got {params.default_duration}"
    
    # Verify duration_adjustable is a boolean
    assert isinstance(params.duration_adjustable, bool), \
        f"duration_adjustable for '{symbol}' should be a boolean"
    
    # Verify manner is set
    assert params.manner, \
        f"manner for '{symbol}' should not be empty"
    
    # Verify place is set
    assert params.place, \
        f"place for '{symbol}' should not be empty"


@given(symbol=resonant_symbols)
@settings(max_examples=100)
def test_resonant_consonant_formants(symbol: str):
    """
    **Feature: consonant-synthesis, Property 11: Consonant Data Completeness (Formants)**
    **Validates: Requirements 11.1**
    
    For any resonant consonant (nasal, approximant, tap, trill, lateral),
    the entry SHALL include F1, F2, F3 formant frequencies.
    """
    params = CONSONANT_DATA[symbol]
    
    # Resonant consonants should have formant values
    assert params.f1 > 0, \
        f"F1 for resonant consonant '{symbol}' should be positive, got {params.f1}"
    assert params.f2 > 0, \
        f"F2 for resonant consonant '{symbol}' should be positive, got {params.f2}"
    assert params.f3 > 0, \
        f"F3 for resonant consonant '{symbol}' should be positive, got {params.f3}"
    
    # Verify formant ordering (F1 < F2 < F3)
    assert params.f1 < params.f2, \
        f"F1 ({params.f1}) should be less than F2 ({params.f2}) for '{symbol}'"
    assert params.f2 < params.f3, \
        f"F2 ({params.f2}) should be less than F3 ({params.f3}) for '{symbol}'"



# ============================================================================
# Property 12: Nasal Data Completeness
# ============================================================================

@given(symbol=nasal_symbols)
@settings(max_examples=100)
def test_nasal_data_completeness(symbol: str):
    """
    **Feature: consonant-synthesis, Property 12: Nasal Data Completeness**
    **Validates: Requirements 11.2**
    
    For any nasal consonant in CONSONANT_DATA, the entry SHALL include
    FNP (nasal pole) and FNZ (nasal zero/anti-formant) parameters.
    """
    params = CONSONANT_DATA[symbol]
    
    # Verify manner is nasal
    assert params.manner == 'nasal', \
        f"Consonant '{symbol}' should have manner 'nasal', got '{params.manner}'"
    
    # Verify FNP (nasal pole) is set and positive
    assert params.fnp > 0, \
        f"FNP for nasal '{symbol}' should be positive, got {params.fnp}"
    
    # Verify FNZ (nasal zero) is set and positive
    assert params.fnz > 0, \
        f"FNZ for nasal '{symbol}' should be positive, got {params.fnz}"
    
    # Verify nasals are voiced
    assert params.voiced is True, \
        f"Nasal '{symbol}' should be voiced"


# ============================================================================
# Property 13: Fricative Data Completeness
# ============================================================================

@given(symbol=fricative_symbols)
@settings(max_examples=100)
def test_fricative_data_completeness(symbol: str):
    """
    **Feature: consonant-synthesis, Property 13: Fricative Data Completeness**
    **Validates: Requirements 11.3**
    
    For any fricative consonant in CONSONANT_DATA, the entry SHALL include
    noise_freq (center frequency) and noise_bw (bandwidth) parameters.
    """
    params = CONSONANT_DATA[symbol]
    
    # Verify manner is fricative or sibilant
    assert params.manner in ('fricative', 'sibilant'), \
        f"Consonant '{symbol}' should have manner 'fricative' or 'sibilant', got '{params.manner}'"
    
    # Verify noise_freq is set and positive
    assert params.noise_freq > 0, \
        f"noise_freq for fricative '{symbol}' should be positive, got {params.noise_freq}"
    
    # Verify noise_bw is set and positive
    assert params.noise_bw > 0, \
        f"noise_bw for fricative '{symbol}' should be positive, got {params.noise_bw}"


# ============================================================================
# Additional Consistency Tests
# ============================================================================

def test_all_consonants_in_data():
    """
    Verify that all consonants defined in category sets are present in CONSONANT_DATA.
    """
    all_categories = (
        NASALS | PLOSIVES | SIBILANTS | FRICATIVES |
        APPROXIMANTS | TAPS | TRILLS |
        LATERAL_FRICATIVES | LATERAL_APPROXIMANTS | LATERAL_FLAPS
    )
    
    for symbol in all_categories:
        assert symbol in CONSONANT_DATA, \
            f"Consonant '{symbol}' from category sets should be in CONSONANT_DATA"


def test_consonant_data_keys_match_all_consonants():
    """
    Verify that CONSONANT_DATA keys match ALL_CONSONANTS set.
    """
    data_keys = set(CONSONANT_DATA.keys())
    assert data_keys == ALL_CONSONANTS, \
        f"CONSONANT_DATA keys should match ALL_CONSONANTS. " \
        f"Missing: {ALL_CONSONANTS - data_keys}, Extra: {data_keys - ALL_CONSONANTS}"


def test_duration_adjustable_consistency():
    """
    Verify that duration_adjustable flag is consistent with category sets.
    """
    for symbol, params in CONSONANT_DATA.items():
        if symbol in DURATION_ADJUSTABLE:
            assert params.duration_adjustable is True, \
                f"'{symbol}' is in DURATION_ADJUSTABLE but has duration_adjustable=False"
        if symbol in DURATION_FIXED:
            assert params.duration_adjustable is False, \
                f"'{symbol}' is in DURATION_FIXED but has duration_adjustable=True"
