"""
Property-based tests for TextGrid parser.

**Feature: phonetic-toolbox, Property 2: TextGrid Round-Trip Consistency**
**Validates: Requirements 2.6, 13.1**
"""
import sys
import tempfile
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from hypothesis import given, settings

from utils.textgrid_parser import TextGrid, Tier, Interval, parse_textgrid, write_textgrid
from tests.generators import textgrid_obj


def textgrids_equal(tg1: TextGrid, tg2: TextGrid, tolerance: float = 1e-9) -> bool:
    """
    Compare two TextGrid objects for equality within floating-point tolerance.
    """
    if abs(tg1.xmin - tg2.xmin) > tolerance:
        return False
    if abs(tg1.xmax - tg2.xmax) > tolerance:
        return False
    if len(tg1.tiers) != len(tg2.tiers):
        return False
    
    for tier1, tier2 in zip(tg1.tiers, tg2.tiers):
        if tier1.name != tier2.name:
            return False
        if abs(tier1.xmin - tier2.xmin) > tolerance:
            return False
        if abs(tier1.xmax - tier2.xmax) > tolerance:
            return False
        if len(tier1.intervals) != len(tier2.intervals):
            return False
        
        for int1, int2 in zip(tier1.intervals, tier2.intervals):
            if abs(int1.xmin - int2.xmin) > tolerance:
                return False
            if abs(int1.xmax - int2.xmax) > tolerance:
                return False
            if int1.text != int2.text:
                return False
    
    return True


@given(tg=textgrid_obj())
@settings(max_examples=100)
def test_textgrid_round_trip_consistency(tg: TextGrid):
    """
    **Feature: phonetic-toolbox, Property 2: TextGrid Round-Trip Consistency**
    **Validates: Requirements 2.6, 13.1**
    
    For any valid TextGrid data structure, writing it to a file and then 
    parsing it back SHALL produce an equivalent data structure with all 
    tiers, intervals, and text preserved.
    """
    # Use tempfile context manager for each test iteration
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test.TextGrid"
        
        # Write the TextGrid to a temporary file
        write_textgrid(tg, output_path)
        
        # Parse it back
        parsed_tg = parse_textgrid(output_path)
        
        # Verify parsing succeeded
        assert parsed_tg is not None, "Parsing should succeed for valid TextGrid"
        
        # Verify round-trip consistency
        assert textgrids_equal(tg, parsed_tg), (
            f"Round-trip should preserve TextGrid data.\n"
            f"Original: {tg}\n"
            f"Parsed: {parsed_tg}"
        )
