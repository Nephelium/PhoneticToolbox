"""
Property-based tests for consonant menu completeness and clipboard functionality.

Tests verify that the consonant menu contains all consonants organized by
manner of articulation, and that clipboard copy functionality works correctly.
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
    CONSONANT_DATA,
    NASALS, PLOSIVES, SIBILANTS, FRICATIVES,
    APPROXIMANTS, TAPS, TRILLS,
    LATERAL_FRICATIVES, LATERAL_APPROXIMANTS, LATERAL_FLAPS
)


# ============================================================================
# Menu Category Definitions (must match klatt_gui.py)
# ============================================================================

# This mirrors the category structure in klatt_gui.py._create_consonant_menu
MENU_CATEGORIES = {
    "鼻音": NASALS,
    "塞音": PLOSIVES,
    "有咝擦音": SIBILANTS,
    "无咝擦音": FRICATIVES,
    "近音": APPROXIMANTS,
    "闪音": TAPS,
    "颤音": TRILLS,
    "边擦音": LATERAL_FRICATIVES,
    "边近音": LATERAL_APPROXIMANTS,
    "边闪音": LATERAL_FLAPS,
}


# ============================================================================
# Test Data Generators
# ============================================================================

# Strategy for sampling any category name
category_names = st.sampled_from(list(MENU_CATEGORIES.keys()))

# Strategy for sampling any consonant from CONSONANT_DATA
consonant_symbols = st.sampled_from(list(CONSONANT_DATA.keys()))


# ============================================================================
# Property 1: Consonant Menu Completeness
# ============================================================================

@given(category_name=category_names)
@settings(max_examples=100)
def test_consonant_menu_completeness(category_name: str):
    """
    **Feature: consonant-synthesis, Property 1: Consonant Menu Completeness**
    **Validates: Requirements 1.3**
    
    For any manner of articulation category in the consonant menu,
    the submenu SHALL contain exactly all consonants defined for that
    category in CONSONANT_DATA.
    """
    category_consonants = MENU_CATEGORIES[category_name]
    
    # Verify all consonants in the category are in CONSONANT_DATA
    for consonant in category_consonants:
        assert consonant in CONSONANT_DATA, \
            f"Consonant '{consonant}' in category '{category_name}' should be in CONSONANT_DATA"
        
        # Verify the consonant's manner matches the expected category
        params = CONSONANT_DATA[consonant]
        expected_manners = _get_expected_manners(category_name)
        assert params.manner in expected_manners, \
            f"Consonant '{consonant}' has manner '{params.manner}', " \
            f"expected one of {expected_manners} for category '{category_name}'"


def _get_expected_manners(category_name: str) -> set:
    """Map category names to expected manner values in CONSONANT_DATA."""
    manner_map = {
        "鼻音": {"nasal"},
        "塞音": {"plosive"},
        "有咝擦音": {"sibilant"},
        "无咝擦音": {"fricative"},
        "近音": {"approximant"},
        "闪音": {"tap"},
        "颤音": {"trill"},
        "边擦音": {"lateral_fricative"},
        "边近音": {"lateral_approximant"},
        "边闪音": {"lateral_flap"},
    }
    return manner_map.get(category_name, set())


def test_all_consonants_covered_by_menu():
    """
    Verify that all consonants in CONSONANT_DATA are covered by exactly one menu category.
    """
    all_menu_consonants = set()
    for category_consonants in MENU_CATEGORIES.values():
        all_menu_consonants.update(category_consonants)
    
    consonant_data_keys = set(CONSONANT_DATA.keys())
    
    # All consonants in data should be in menu
    missing_from_menu = consonant_data_keys - all_menu_consonants
    assert not missing_from_menu, \
        f"Consonants missing from menu categories: {missing_from_menu}"
    
    # All consonants in menu should be in data
    missing_from_data = all_menu_consonants - consonant_data_keys
    assert not missing_from_data, \
        f"Menu consonants missing from CONSONANT_DATA: {missing_from_data}"


def test_no_duplicate_consonants_across_categories():
    """
    Verify that no consonant appears in multiple menu categories.
    """
    seen = {}
    for category_name, consonants in MENU_CATEGORIES.items():
        for consonant in consonants:
            if consonant in seen:
                pytest.fail(
                    f"Consonant '{consonant}' appears in both "
                    f"'{seen[consonant]}' and '{category_name}'"
                )
            seen[consonant] = category_name


# ============================================================================
# Property 2: Clipboard Copy Correctness
# ============================================================================

@given(consonant=consonant_symbols)
@settings(max_examples=100)
def test_clipboard_copy_correctness(consonant: str):
    """
    **Feature: consonant-synthesis, Property 2: Clipboard Copy Correctness**
    **Validates: Requirements 1.4**
    
    For any consonant in the menu, clicking it SHALL result in that exact
    IPA symbol being copied to the system clipboard.
    
    Note: This test verifies the data integrity that would be copied.
    Actual clipboard testing requires GUI interaction which is tested separately.
    """
    # Verify the consonant exists in CONSONANT_DATA
    assert consonant in CONSONANT_DATA, \
        f"Consonant '{consonant}' should be in CONSONANT_DATA"
    
    # Verify the consonant symbol is a valid string
    assert isinstance(consonant, str), \
        f"Consonant symbol should be a string, got {type(consonant)}"
    
    # Verify the symbol is not empty
    assert len(consonant) > 0, \
        "Consonant symbol should not be empty"
    
    # Verify the symbol matches the params.symbol
    params = CONSONANT_DATA[consonant]
    assert params.symbol == consonant, \
        f"Symbol mismatch: key '{consonant}' != params.symbol '{params.symbol}'"


def test_consonant_symbols_are_valid_unicode():
    """
    Verify that all consonant symbols are valid Unicode characters
    suitable for clipboard operations.
    """
    for consonant in CONSONANT_DATA.keys():
        # Should be encodable as UTF-8
        try:
            encoded = consonant.encode('utf-8')
            decoded = encoded.decode('utf-8')
            assert decoded == consonant, \
                f"Consonant '{consonant}' should round-trip through UTF-8 encoding"
        except UnicodeError as e:
            pytest.fail(f"Consonant '{consonant}' has Unicode encoding issue: {e}")
