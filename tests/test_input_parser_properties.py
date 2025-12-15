"""
Property-based tests for the input parser.

Tests verify that the input parser correctly handles duration modifiers
for both adjustable and non-adjustable consonants.
"""
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from klatt.input_parser import (
    InputParser, ParsedSegment,
    calculate_duration_modifier, apply_duration_modifier
)
from klatt.consonant_data import (
    CONSONANT_DATA, DURATION_ADJUSTABLE, DURATION_FIXED,
    PLOSIVES, TAPS, TRILLS, LATERAL_FLAPS
)


# ============================================================================
# Test Data Generators
# ============================================================================

# Strategy for duration-adjustable consonants
adjustable_consonants = st.sampled_from(list(DURATION_ADJUSTABLE))

# Strategy for non-adjustable consonants (plosives, taps, trills, lateral flaps)
non_adjustable_consonants = st.sampled_from(list(DURATION_FIXED))

# Strategy for single duration modifiers
single_modifier = st.sampled_from(['+', '-', '*', '/'])

# Strategy for modifier strings (0-4 modifiers)
modifier_string = st.text(alphabet='+-*/', min_size=0, max_size=4)


# ============================================================================
# Property 8: Duration Modifier Effect
# ============================================================================

@given(symbol=adjustable_consonants)
@settings(max_examples=100)
def test_duration_modifier_plus(symbol: str):
    """
    **Feature: consonant-synthesis, Property 8: Duration Modifier Effect (+)**
    **Validates: Requirements 8.3**
    
    For any duration-adjustable consonant with modifier "+",
    the duration SHALL be 1.1x the default.
    """
    parser = InputParser()
    
    # Parse consonant with + modifier
    segments = parser.parse(f"{symbol}+")
    
    assert len(segments) == 1, f"Expected 1 segment, got {len(segments)}"
    seg = segments[0]
    
    assert seg['type'] == 'consonant', f"Expected consonant, got {seg['type']}"
    assert seg['symbol'] == symbol, f"Expected symbol '{symbol}', got '{seg['symbol']}'"
    
    # Verify modifier is 1.1
    expected_modifier = 1.1
    assert abs(seg['duration_modifier'] - expected_modifier) < 0.001, \
        f"Expected modifier {expected_modifier}, got {seg['duration_modifier']}"


@given(symbol=adjustable_consonants)
@settings(max_examples=100)
def test_duration_modifier_minus(symbol: str):
    """
    **Feature: consonant-synthesis, Property 8: Duration Modifier Effect (-)**
    **Validates: Requirements 8.4**
    
    For any duration-adjustable consonant with modifier "-",
    the duration SHALL be 0.9x the default.
    """
    parser = InputParser()
    
    # Parse consonant with - modifier
    segments = parser.parse(f"{symbol}-")
    
    assert len(segments) == 1, f"Expected 1 segment, got {len(segments)}"
    seg = segments[0]
    
    assert seg['type'] == 'consonant', f"Expected consonant, got {seg['type']}"
    
    # Verify modifier is 0.9
    expected_modifier = 0.9
    assert abs(seg['duration_modifier'] - expected_modifier) < 0.001, \
        f"Expected modifier {expected_modifier}, got {seg['duration_modifier']}"


@given(symbol=adjustable_consonants)
@settings(max_examples=100)
def test_duration_modifier_multiply(symbol: str):
    """
    **Feature: consonant-synthesis, Property 8: Duration Modifier Effect (*)**
    **Validates: Requirements 8.5**
    
    For any duration-adjustable consonant with modifier "*",
    the duration SHALL be 2x the default.
    """
    parser = InputParser()
    
    # Parse consonant with * modifier
    segments = parser.parse(f"{symbol}*")
    
    assert len(segments) == 1, f"Expected 1 segment, got {len(segments)}"
    seg = segments[0]
    
    assert seg['type'] == 'consonant', f"Expected consonant, got {seg['type']}"
    
    # Verify modifier is 2.0
    expected_modifier = 2.0
    assert abs(seg['duration_modifier'] - expected_modifier) < 0.001, \
        f"Expected modifier {expected_modifier}, got {seg['duration_modifier']}"


@given(symbol=adjustable_consonants)
@settings(max_examples=100)
def test_duration_modifier_divide(symbol: str):
    """
    **Feature: consonant-synthesis, Property 8: Duration Modifier Effect (/)**
    **Validates: Requirements 8.6**
    
    For any duration-adjustable consonant with modifier "/",
    the duration SHALL be 0.5x the default.
    """
    parser = InputParser()
    
    # Parse consonant with / modifier
    segments = parser.parse(f"{symbol}/")
    
    assert len(segments) == 1, f"Expected 1 segment, got {len(segments)}"
    seg = segments[0]
    
    assert seg['type'] == 'consonant', f"Expected consonant, got {seg['type']}"
    
    # Verify modifier is 0.5
    expected_modifier = 0.5
    assert abs(seg['duration_modifier'] - expected_modifier) < 0.001, \
        f"Expected modifier {expected_modifier}, got {seg['duration_modifier']}"


@given(symbol=adjustable_consonants, modifiers=modifier_string)
@settings(max_examples=100)
def test_duration_modifier_cumulative(symbol: str, modifiers: str):
    """
    **Feature: consonant-synthesis, Property 8: Duration Modifier Effect (Cumulative)**
    **Validates: Requirements 8.3, 8.4, 8.5, 8.6**
    
    For any duration-adjustable consonant with multiple modifiers,
    the modifiers SHALL be applied cumulatively.
    """
    parser = InputParser()
    
    # Calculate expected modifier
    expected_modifier = calculate_duration_modifier(modifiers)
    
    # Parse consonant with modifiers
    segments = parser.parse(f"{symbol}{modifiers}")
    
    assert len(segments) == 1, f"Expected 1 segment, got {len(segments)}"
    seg = segments[0]
    
    assert seg['type'] == 'consonant', f"Expected consonant, got {seg['type']}"
    
    # Verify modifier matches expected
    assert abs(seg['duration_modifier'] - expected_modifier) < 0.001, \
        f"Expected modifier {expected_modifier}, got {seg['duration_modifier']}"


# ============================================================================
# Property 9: Non-Adjustable Duration Invariance
# ============================================================================

@given(symbol=non_adjustable_consonants, modifiers=modifier_string)
@settings(max_examples=100)
def test_non_adjustable_duration_invariance(symbol: str, modifiers: str):
    """
    **Feature: consonant-synthesis, Property 9: Non-Adjustable Duration Invariance**
    **Validates: Requirements 8.2**
    
    For any non-adjustable consonant (plosive, tap, trill) with any duration modifier,
    the duration SHALL remain at the default value (modifier = 1.0).
    """
    parser = InputParser()
    
    # Parse consonant with modifiers
    segments = parser.parse(f"{symbol}{modifiers}")
    
    assert len(segments) == 1, f"Expected 1 segment, got {len(segments)}"
    seg = segments[0]
    
    assert seg['type'] == 'consonant', f"Expected consonant, got {seg['type']}"
    assert seg['symbol'] == symbol, f"Expected symbol '{symbol}', got '{seg['symbol']}'"
    
    # Verify modifier is always 1.0 for non-adjustable consonants
    assert seg['duration_modifier'] == 1.0, \
        f"Non-adjustable consonant '{symbol}' should have modifier 1.0, got {seg['duration_modifier']}"
    
    # Verify duration_adjustable flag is False
    assert seg['duration_adjustable'] is False, \
        f"Non-adjustable consonant '{symbol}' should have duration_adjustable=False"


@given(symbol=non_adjustable_consonants)
@settings(max_examples=100)
def test_non_adjustable_with_plus_modifier(symbol: str):
    """
    **Feature: consonant-synthesis, Property 9: Non-Adjustable Duration Invariance (+)**
    **Validates: Requirements 8.2**
    
    For any non-adjustable consonant with "+" modifier,
    the duration SHALL remain at the default value.
    """
    parser = InputParser()
    
    # Parse with + modifier
    segments = parser.parse(f"{symbol}+")
    
    assert len(segments) == 1
    seg = segments[0]
    
    # Modifier should be 1.0 (ignored)
    assert seg['duration_modifier'] == 1.0, \
        f"Non-adjustable consonant '{symbol}' with '+' should have modifier 1.0"


@given(symbol=non_adjustable_consonants)
@settings(max_examples=100)
def test_non_adjustable_with_multiply_modifier(symbol: str):
    """
    **Feature: consonant-synthesis, Property 9: Non-Adjustable Duration Invariance (*)**
    **Validates: Requirements 8.2**
    
    For any non-adjustable consonant with "*" modifier,
    the duration SHALL remain at the default value.
    """
    parser = InputParser()
    
    # Parse with * modifier
    segments = parser.parse(f"{symbol}*")
    
    assert len(segments) == 1
    seg = segments[0]
    
    # Modifier should be 1.0 (ignored)
    assert seg['duration_modifier'] == 1.0, \
        f"Non-adjustable consonant '{symbol}' with '*' should have modifier 1.0"


# ============================================================================
# Additional Parser Tests
# ============================================================================

def test_parser_empty_input():
    """Test that empty input returns empty list."""
    parser = InputParser()
    segments = parser.parse("")
    assert segments == []


def test_parser_space_as_silence():
    """Test that space is parsed as silence."""
    parser = InputParser()
    segments = parser.parse(" ")
    
    assert len(segments) == 1
    assert segments[0]['type'] == 'silence'
    assert segments[0]['symbol'] == ' '


def test_parser_vowel_recognition():
    """Test that vowels are correctly recognized."""
    parser = InputParser()
    segments = parser.parse("a")
    
    assert len(segments) == 1
    assert segments[0]['type'] == 'vowel'
    assert segments[0]['symbol'] == 'a'


def test_parser_mixed_input():
    """Test parsing mixed vowel and consonant input."""
    parser = InputParser()
    segments = parser.parse("ama")
    
    assert len(segments) == 3
    assert segments[0]['type'] == 'vowel'
    assert segments[0]['symbol'] == 'a'
    assert segments[1]['type'] == 'consonant'
    assert segments[1]['symbol'] == 'm'
    assert segments[2]['type'] == 'vowel'
    assert segments[2]['symbol'] == 'a'


def test_calculate_duration_modifier():
    """Test the calculate_duration_modifier helper function."""
    assert calculate_duration_modifier("") == 1.0
    assert abs(calculate_duration_modifier("+") - 1.1) < 0.001
    assert abs(calculate_duration_modifier("-") - 0.9) < 0.001
    assert abs(calculate_duration_modifier("*") - 2.0) < 0.001
    assert abs(calculate_duration_modifier("/") - 0.5) < 0.001
    assert abs(calculate_duration_modifier("++") - 1.21) < 0.001
    assert abs(calculate_duration_modifier("*/") - 1.0) < 0.001


def test_apply_duration_modifier():
    """Test the apply_duration_modifier helper function."""
    # Adjustable
    assert abs(apply_duration_modifier(100.0, 1.1, True) - 110.0) < 0.001
    assert abs(apply_duration_modifier(100.0, 2.0, True) - 200.0) < 0.001
    
    # Non-adjustable (modifier ignored)
    assert abs(apply_duration_modifier(100.0, 1.1, False) - 100.0) < 0.001
    assert abs(apply_duration_modifier(100.0, 2.0, False) - 100.0) < 0.001
