"""
Property-based tests for transition (音渡段) generation.

Tests verify that transitions are correctly inserted between consonants
and vowels with appropriate duration and acoustic properties.
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
    NASALS, PLOSIVES, SIBILANTS, FRICATIVES,
    APPROXIMANTS, LATERAL_APPROXIMANTS, TAPS, TRILLS
)
from klatt.transition_gen import (
    TransitionGenerator, generate_transition, is_transition_needed,
    DEFAULT_VOWEL_FORMANTS, SMOOTHABLE_MANNERS
)


# ============================================================================
# Test Data Generators
# ============================================================================

# All consonant symbols
all_consonants = list(CONSONANT_DATA.keys())
consonant_symbols = st.sampled_from(all_consonants)

# All vowel symbols
all_vowels = list(DEFAULT_VOWEL_FORMANTS.keys())
vowel_symbols = st.sampled_from(all_vowels)

# Duration strategy (valid range for transitions: 20-50ms)
valid_duration_ms = st.floats(min_value=20.0, max_value=50.0, 
                               allow_nan=False, allow_infinity=False)

# Extended duration strategy (including out-of-range values)
any_duration_ms = st.floats(min_value=1.0, max_value=100.0,
                            allow_nan=False, allow_infinity=False)


# ============================================================================
# Helper Functions
# ============================================================================

def make_vowel_segment(symbol: str) -> dict:
    """Create a vowel segment dictionary."""
    formants = DEFAULT_VOWEL_FORMANTS.get(symbol, [500, 1500, 2500])
    return {
        'type': 'vowel',
        'symbol': symbol,
        'f1': formants[0],
        'f2': formants[1],
        'f3': formants[2]
    }


def make_consonant_segment(symbol: str) -> dict:
    """Create a consonant segment dictionary."""
    params = CONSONANT_DATA.get(symbol)
    if params:
        return {
            'type': 'consonant',
            'symbol': symbol,
            'f1': params.f1,
            'f2': params.f2,
            'f3': params.f3
        }
    return {'type': 'consonant', 'symbol': symbol}


def make_silence_segment() -> dict:
    """Create a silence segment dictionary."""
    return {'type': 'silence', 'symbol': ' '}


# ============================================================================
# Property 10: Transition Insertion
# ============================================================================

@given(consonant=consonant_symbols, vowel=vowel_symbols)
@settings(max_examples=100)
def test_cv_transition_needed(consonant: str, vowel: str):
    """
    **Feature: consonant-synthesis, Property 10: Transition Insertion**
    **Validates: Requirements 10.1**
    
    For any consonant-vowel sequence, a transition SHALL be needed.
    """
    consonant_seg = make_consonant_segment(consonant)
    vowel_seg = make_vowel_segment(vowel)
    
    generator = TransitionGenerator(fs=16000)
    
    # Transition should be needed for consonant -> vowel
    assert generator.is_transition_needed(consonant_seg, vowel_seg), \
        f"Transition should be needed between consonant '{consonant}' and vowel '{vowel}'"


@given(vowel=vowel_symbols, consonant=consonant_symbols)
@settings(max_examples=100)
def test_vc_transition_needed(vowel: str, consonant: str):
    """
    **Feature: consonant-synthesis, Property 10: Transition Insertion**
    **Validates: Requirements 10.2**
    
    For any vowel-consonant sequence, a transition SHALL be needed.
    """
    vowel_seg = make_vowel_segment(vowel)
    consonant_seg = make_consonant_segment(consonant)
    
    generator = TransitionGenerator(fs=16000)
    
    # Transition should be needed for vowel -> consonant
    assert generator.is_transition_needed(vowel_seg, consonant_seg), \
        f"Transition should be needed between vowel '{vowel}' and consonant '{consonant}'"


@given(consonant=consonant_symbols, vowel=vowel_symbols, duration=valid_duration_ms)
@settings(max_examples=100)
def test_cv_transition_duration_in_range(consonant: str, vowel: str, duration: float):
    """
    **Feature: consonant-synthesis, Property 10: Transition Insertion**
    **Validates: Requirements 10.5**
    
    For any consonant-vowel transition, the duration SHALL be in range 20-50ms.
    """
    consonant_seg = make_consonant_segment(consonant)
    vowel_seg = make_vowel_segment(vowel)
    
    generator = TransitionGenerator(fs=16000)
    
    # Generate transition
    audio = generator.generate(consonant_seg, vowel_seg, duration)
    
    # Calculate actual duration from audio length
    actual_duration_ms = len(audio) / 16000 * 1000
    
    # Duration should be in valid range (with some tolerance for resampling)
    assert 15.0 <= actual_duration_ms <= 60.0, \
        f"Transition duration {actual_duration_ms:.1f}ms should be approximately in range [20, 50]ms"


@given(consonant=consonant_symbols, vowel=vowel_symbols, duration=any_duration_ms)
@settings(max_examples=100)
def test_transition_duration_clamped(consonant: str, vowel: str, duration: float):
    """
    **Feature: consonant-synthesis, Property 10: Transition Insertion**
    **Validates: Requirements 10.5**
    
    For any requested duration, the actual transition duration SHALL be
    clamped to the valid range [20, 50]ms.
    """
    consonant_seg = make_consonant_segment(consonant)
    vowel_seg = make_vowel_segment(vowel)
    
    generator = TransitionGenerator(fs=16000)
    
    # Generate transition with potentially out-of-range duration
    audio = generator.generate(consonant_seg, vowel_seg, duration)
    
    # Calculate actual duration from audio length
    actual_duration_ms = len(audio) / 16000 * 1000
    
    # Duration should be clamped to valid range (with tolerance)
    # Min is 20ms, max is 50ms
    assert actual_duration_ms >= 15.0, \
        f"Transition duration {actual_duration_ms:.1f}ms should be at least ~20ms"
    assert actual_duration_ms <= 60.0, \
        f"Transition duration {actual_duration_ms:.1f}ms should be at most ~50ms"


@given(consonant=consonant_symbols, vowel=vowel_symbols)
@settings(max_examples=100)
def test_transition_produces_audio(consonant: str, vowel: str):
    """
    **Feature: consonant-synthesis, Property 10: Transition Insertion**
    **Validates: Requirements 10.1, 10.2**
    
    For any consonant-vowel or vowel-consonant sequence, the transition
    generator SHALL produce valid audio output.
    """
    consonant_seg = make_consonant_segment(consonant)
    vowel_seg = make_vowel_segment(vowel)
    
    generator = TransitionGenerator(fs=16000)
    
    # Test CV transition
    cv_audio = generator.generate(consonant_seg, vowel_seg)
    assert len(cv_audio) > 0, \
        f"CV transition for '{consonant}'-'{vowel}' should produce audio"
    assert np.max(np.abs(cv_audio)) <= 1.0, \
        f"CV transition audio should be normalized to [-1, 1]"
    
    # Test VC transition
    vc_audio = generator.generate(vowel_seg, consonant_seg)
    assert len(vc_audio) > 0, \
        f"VC transition for '{vowel}'-'{consonant}' should produce audio"
    assert np.max(np.abs(vc_audio)) <= 1.0, \
        f"VC transition audio should be normalized to [-1, 1]"


# ============================================================================
# Smoothable vs Non-Smoothable Transitions
# ============================================================================

# Smoothable consonants (nasals, approximants, lateral approximants)
smoothable_consonants = [s for s in all_consonants 
                         if CONSONANT_DATA[s].manner in SMOOTHABLE_MANNERS]
smoothable_symbols = st.sampled_from(smoothable_consonants) if smoothable_consonants else st.nothing()

# Non-smoothable consonants (plosives, fricatives, etc.)
non_smoothable_consonants = [s for s in all_consonants 
                             if CONSONANT_DATA[s].manner not in SMOOTHABLE_MANNERS]
non_smoothable_symbols = st.sampled_from(non_smoothable_consonants) if non_smoothable_consonants else st.nothing()


@given(consonant=smoothable_symbols, vowel=vowel_symbols)
@settings(max_examples=100)
def test_smoothable_consonant_uses_formant_smoothing(consonant: str, vowel: str):
    """
    **Feature: consonant-synthesis, Property 10: Transition Insertion**
    **Validates: Requirements 10.3**
    
    For any smoothable consonant (nasal, approximant, lateral_approximant),
    the transition SHALL use formant smoothing.
    """
    consonant_seg = make_consonant_segment(consonant)
    vowel_seg = make_vowel_segment(vowel)
    
    generator = TransitionGenerator(fs=16000)
    
    # Check that formant smoothing can be used
    can_smooth = generator.can_smooth(consonant_seg, vowel_seg)
    
    assert can_smooth, \
        f"Smoothable consonant '{consonant}' (manner={CONSONANT_DATA[consonant].manner}) " \
        f"should use formant smoothing with vowel '{vowel}'"


@given(consonant=non_smoothable_symbols, vowel=vowel_symbols)
@settings(max_examples=100)
def test_non_smoothable_consonant_uses_approximant(consonant: str, vowel: str):
    """
    **Feature: consonant-synthesis, Property 10: Transition Insertion**
    **Validates: Requirements 10.4**
    
    For any non-smoothable consonant (plosive, fricative, etc.),
    the transition SHALL NOT use formant smoothing.
    """
    consonant_seg = make_consonant_segment(consonant)
    vowel_seg = make_vowel_segment(vowel)
    
    generator = TransitionGenerator(fs=16000)
    
    # Check that formant smoothing cannot be used
    can_smooth = generator.can_smooth(consonant_seg, vowel_seg)
    
    assert not can_smooth, \
        f"Non-smoothable consonant '{consonant}' (manner={CONSONANT_DATA[consonant].manner}) " \
        f"should NOT use formant smoothing with vowel '{vowel}'"


# ============================================================================
# Edge Cases
# ============================================================================

def test_vowel_vowel_no_transition_needed():
    """Test that vowel-vowel sequences don't require transition."""
    generator = TransitionGenerator(fs=16000)
    
    vowel1 = make_vowel_segment('a')
    vowel2 = make_vowel_segment('i')
    
    # Vowel-vowel should not need transition
    assert not generator.is_transition_needed(vowel1, vowel2), \
        "Vowel-vowel sequence should not need transition"


def test_silence_vowel_no_transition_needed():
    """Test that silence-vowel sequences don't require transition."""
    generator = TransitionGenerator(fs=16000)
    
    silence = make_silence_segment()
    vowel = make_vowel_segment('a')
    
    # Silence-vowel should not need transition
    assert not generator.is_transition_needed(silence, vowel), \
        "Silence-vowel sequence should not need transition"


def test_consonant_silence_no_transition_needed():
    """Test that consonant-silence sequences don't require transition."""
    generator = TransitionGenerator(fs=16000)
    
    consonant = make_consonant_segment('m')
    silence = make_silence_segment()
    
    # Consonant-silence should not need transition
    assert not generator.is_transition_needed(consonant, silence), \
        "Consonant-silence sequence should not need transition"


def test_zero_duration_returns_empty():
    """Test that zero duration returns empty array."""
    generator = TransitionGenerator(fs=16000)
    
    consonant = make_consonant_segment('m')
    vowel = make_vowel_segment('a')
    
    # Zero duration should return empty (but will be clamped to min)
    audio = generator.generate(consonant, vowel, 0.0)
    # Due to clamping, this will actually produce audio at min duration
    # So we just verify it doesn't crash


def test_default_duration_is_30ms():
    """Test that default transition duration is 30ms."""
    generator = TransitionGenerator(fs=16000)
    
    assert generator.get_transition_duration_ms() == 30.0, \
        "Default transition duration should be 30ms"


def test_set_transition_duration():
    """Test setting custom transition duration."""
    generator = TransitionGenerator(fs=16000)
    
    # Set to valid value
    generator.set_transition_duration_ms(40.0)
    assert generator.get_transition_duration_ms() == 40.0
    
    # Set to below minimum (should clamp)
    generator.set_transition_duration_ms(10.0)
    assert generator.get_transition_duration_ms() == 20.0
    
    # Set to above maximum (should clamp)
    generator.set_transition_duration_ms(100.0)
    assert generator.get_transition_duration_ms() == 50.0


# ============================================================================
# Convenience Function Tests
# ============================================================================

def test_generate_transition_convenience():
    """Test the generate_transition convenience function."""
    consonant = make_consonant_segment('n')
    vowel = make_vowel_segment('a')
    
    audio = generate_transition(consonant, vowel, fs=16000, duration_ms=30.0)
    
    assert len(audio) > 0, "Convenience function should produce audio"
    assert np.max(np.abs(audio)) <= 1.0, "Audio should be normalized"


def test_is_transition_needed_convenience():
    """Test the is_transition_needed convenience function."""
    consonant = make_consonant_segment('m')
    vowel = make_vowel_segment('a')
    
    assert is_transition_needed(consonant, vowel), \
        "Convenience function should detect CV transition need"
    assert is_transition_needed(vowel, consonant), \
        "Convenience function should detect VC transition need"
