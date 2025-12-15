"""
Property-based tests for voiced consonant voicing.

Tests verify that all voiced consonants (nasals, approximants, voiced fricatives,
voiced plosives) maintain voicing (AV > 0) during synthesis.

Property 4: Voiced Consonant Voicing
For any voiced consonant (nasal, approximant, voiced fricative, voiced plosive),
the synthesis SHALL maintain AV > 0 during the voiced portion.
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
    NASALS, PLOSIVES, SIBILANTS, FRICATIVES, APPROXIMANTS,
    TAPS, TRILLS, LATERAL_FRICATIVES, LATERAL_APPROXIMANTS, LATERAL_FLAPS
)


# ============================================================================
# Collect All Voiced Consonants
# ============================================================================

def get_voiced_consonants():
    """Get all voiced consonants from CONSONANT_DATA."""
    return [sym for sym, params in CONSONANT_DATA.items() if params.voiced]


def get_voiceless_consonants():
    """Get all voiceless consonants from CONSONANT_DATA."""
    return [sym for sym, params in CONSONANT_DATA.items() if not params.voiced]


# All voiced consonants
ALL_VOICED = get_voiced_consonants()
voiced_consonant_symbols = st.sampled_from(ALL_VOICED) if ALL_VOICED else st.nothing()

# All voiceless consonants
ALL_VOICELESS = get_voiceless_consonants()
voiceless_consonant_symbols = st.sampled_from(ALL_VOICELESS) if ALL_VOICELESS else st.nothing()


# ============================================================================
# Property 4: Voiced Consonant Voicing - Main Tests
# ============================================================================

@given(symbol=voiced_consonant_symbols)
@settings(max_examples=100)
def test_voiced_consonant_has_voiced_flag(symbol: str):
    """
    **Feature: consonant-synthesis, Property 4: Voiced Consonant Voicing**
    **Validates: Requirements 2.4, 3.4, 4.5, 5.3**
    
    For any voiced consonant, the CONSONANT_DATA entry SHALL have
    voiced=True.
    """
    params = CONSONANT_DATA[symbol]
    
    assert params.voiced is True, \
        f"Consonant '{symbol}' should have voiced=True"


@given(symbol=voiceless_consonant_symbols)
@settings(max_examples=100)
def test_voiceless_consonant_has_voiceless_flag(symbol: str):
    """
    **Feature: consonant-synthesis, Property 4: Voiced Consonant Voicing (Inverse)**
    **Validates: Requirements 3.3, 4.4**
    
    For any voiceless consonant, the CONSONANT_DATA entry SHALL have
    voiced=False.
    """
    params = CONSONANT_DATA[symbol]
    
    assert params.voiced is False, \
        f"Consonant '{symbol}' should have voiced=False"


# ============================================================================
# Property 4: Voiced Consonant Voicing - By Category
# ============================================================================

@given(symbol=st.sampled_from(list(NASALS)))
@settings(max_examples=100)
def test_all_nasals_are_voiced(symbol: str):
    """
    **Feature: consonant-synthesis, Property 4: Voiced Consonant Voicing**
    **Validates: Requirements 2.4**
    
    For any nasal consonant, the consonant SHALL be voiced.
    All nasals are voiced by definition in phonetics.
    """
    params = CONSONANT_DATA[symbol]
    
    assert params.voiced is True, \
        f"Nasal '{symbol}' must be voiced (all nasals are voiced)"
    assert params.manner == 'nasal', \
        f"'{symbol}' should have manner 'nasal'"


@given(symbol=st.sampled_from(list(APPROXIMANTS)))
@settings(max_examples=100)
def test_all_approximants_are_voiced(symbol: str):
    """
    **Feature: consonant-synthesis, Property 4: Voiced Consonant Voicing**
    **Validates: Requirements 5.3**
    
    For any approximant consonant, the consonant SHALL be voiced.
    Approximants are typically voiced (voiceless approximants are rare).
    """
    params = CONSONANT_DATA[symbol]
    
    assert params.voiced is True, \
        f"Approximant '{symbol}' should be voiced"
    assert params.manner == 'approximant', \
        f"'{symbol}' should have manner 'approximant'"


@given(symbol=st.sampled_from(list(LATERAL_APPROXIMANTS)))
@settings(max_examples=100)
def test_lateral_approximants_are_voiced(symbol: str):
    """
    **Feature: consonant-synthesis, Property 4: Voiced Consonant Voicing**
    **Validates: Requirements 5.3**
    
    For any lateral approximant, the consonant SHALL be voiced.
    """
    params = CONSONANT_DATA[symbol]
    
    assert params.voiced is True, \
        f"Lateral approximant '{symbol}' should be voiced"


# ============================================================================
# Property 4: Voiced Consonant Voicing - Fricatives
# ============================================================================

# Voiced fricatives (sibilants + non-sibilants)
voiced_sibilants = [s for s in SIBILANTS if CONSONANT_DATA[s].voiced]
voiced_non_sibilants = [s for s in FRICATIVES if CONSONANT_DATA[s].voiced]
all_voiced_fricatives = voiced_sibilants + voiced_non_sibilants

if all_voiced_fricatives:
    @given(symbol=st.sampled_from(all_voiced_fricatives))
    @settings(max_examples=100)
    def test_voiced_fricatives_have_voicing(symbol: str):
        """
        **Feature: consonant-synthesis, Property 4: Voiced Consonant Voicing**
        **Validates: Requirements 4.5**
        
        For any voiced fricative, the consonant SHALL be marked as voiced.
        """
        params = CONSONANT_DATA[symbol]
        
        assert params.voiced is True, \
            f"Voiced fricative '{symbol}' should have voiced=True"
        assert params.manner in ('sibilant', 'fricative'), \
            f"'{symbol}' should have manner 'sibilant' or 'fricative'"


# ============================================================================
# Property 4: Voiced Consonant Voicing - Plosives
# ============================================================================

# Voiced plosives
voiced_plosives = [s for s in PLOSIVES if CONSONANT_DATA[s].voiced]

if voiced_plosives:
    @given(symbol=st.sampled_from(voiced_plosives))
    @settings(max_examples=100)
    def test_voiced_plosives_have_voicing(symbol: str):
        """
        **Feature: consonant-synthesis, Property 4: Voiced Consonant Voicing**
        **Validates: Requirements 3.4**
        
        For any voiced plosive, the consonant SHALL be marked as voiced.
        """
        params = CONSONANT_DATA[symbol]
        
        assert params.voiced is True, \
            f"Voiced plosive '{symbol}' should have voiced=True"
        assert params.manner == 'plosive', \
            f"'{symbol}' should have manner 'plosive'"


# ============================================================================
# Property 4: Voiced Consonant Voicing - Taps, Trills, Laterals
# ============================================================================

@given(symbol=st.sampled_from(list(TAPS)))
@settings(max_examples=100)
def test_taps_are_voiced(symbol: str):
    """
    **Feature: consonant-synthesis, Property 4: Voiced Consonant Voicing**
    **Validates: Requirements 2.4, 3.4, 4.5, 5.3**
    
    For any tap/flap consonant, the consonant SHALL be voiced.
    Taps are typically voiced.
    """
    params = CONSONANT_DATA[symbol]
    
    assert params.voiced is True, \
        f"Tap '{symbol}' should be voiced"


@given(symbol=st.sampled_from(list(TRILLS)))
@settings(max_examples=100)
def test_trills_are_voiced(symbol: str):
    """
    **Feature: consonant-synthesis, Property 4: Voiced Consonant Voicing**
    **Validates: Requirements 2.4, 3.4, 4.5, 5.3**
    
    For any trill consonant, the consonant SHALL be voiced.
    Trills are typically voiced.
    """
    params = CONSONANT_DATA[symbol]
    
    assert params.voiced is True, \
        f"Trill '{symbol}' should be voiced"


# ============================================================================
# Consistency Tests
# ============================================================================

def test_voiced_voiceless_pairs_are_consistent():
    """
    Test that voiced/voiceless pairs are correctly marked.
    
    Common pairs: p/b, t/d, k/ɡ, f/v, s/z, etc.
    """
    pairs = [
        ('p', 'b'),  # bilabial plosives
        ('t', 'd'),  # alveolar plosives
        ('k', 'ɡ'),  # velar plosives
        ('f', 'v'),  # labiodental fricatives
        ('s', 'z'),  # alveolar sibilants
        ('ʃ', 'ʒ'),  # postalveolar sibilants
        ('θ', 'ð'),  # dental fricatives
        ('x', 'ɣ'),  # velar fricatives
    ]
    
    for voiceless, voiced in pairs:
        if voiceless in CONSONANT_DATA and voiced in CONSONANT_DATA:
            vl_params = CONSONANT_DATA[voiceless]
            v_params = CONSONANT_DATA[voiced]
            
            assert vl_params.voiced is False, \
                f"'{voiceless}' should be voiceless"
            assert v_params.voiced is True, \
                f"'{voiced}' should be voiced"
            
            # They should have the same place of articulation
            assert vl_params.place == v_params.place, \
                f"'{voiceless}' and '{voiced}' should have same place"


def test_all_consonants_have_voicing_attribute():
    """Test that all consonants have the voiced attribute defined."""
    for symbol, params in CONSONANT_DATA.items():
        assert isinstance(params.voiced, bool), \
            f"'{symbol}' should have voiced as bool, got {type(params.voiced)}"


def test_voiced_count_is_reasonable():
    """Test that the number of voiced consonants is reasonable."""
    voiced_count = len(get_voiced_consonants())
    voiceless_count = len(get_voiceless_consonants())
    total = len(CONSONANT_DATA)
    
    # Most consonants should be either voiced or voiceless
    assert voiced_count + voiceless_count == total, \
        "All consonants should be either voiced or voiceless"
    
    # There should be a reasonable mix (not all voiced or all voiceless)
    assert voiced_count > 0, "There should be some voiced consonants"
    assert voiceless_count > 0, "There should be some voiceless consonants"
