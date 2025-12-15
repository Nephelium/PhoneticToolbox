"""
Test to verify hypothesis is properly configured.
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

from tests.generators import (
    f0_values,
    safe_filenames,
    chinese_chars,
    acoustic_params_dict,
    textgrid_tier,
)


@given(f0=f0_values)
@settings(max_examples=100)
def test_f0_generator_produces_valid_range(f0: float):
    """Verify F0 generator produces values in valid range."""
    assert 40.0 <= f0 <= 500.0


@given(filename=safe_filenames)
@settings(max_examples=100)
def test_safe_filename_generator(filename: str):
    """Verify safe filename generator produces non-empty strings."""
    assert len(filename) > 0
    assert len(filename.strip()) > 0


@given(chinese=chinese_chars)
@settings(max_examples=100)
def test_chinese_chars_generator(chinese: str):
    """Verify Chinese character generator produces valid Chinese text."""
    assert len(chinese) > 0
    # Check all characters are in Chinese Unicode range
    for char in chinese:
        assert 0x4e00 <= ord(char) <= 0x9fff


@given(params=acoustic_params_dict())
@settings(max_examples=50)
def test_acoustic_params_dict_structure(params: dict):
    """Verify acoustic params dict has expected structure."""
    assert "frameshift" in params
    assert "pF0" in params
    assert "pF1" in params
    assert "Energy" in params
    assert isinstance(params["frameshift"], float)
    assert hasattr(params["pF0"], "__len__")


@given(tier=textgrid_tier())
@settings(max_examples=50)
def test_textgrid_tier_structure(tier: dict):
    """Verify TextGrid tier has expected structure."""
    assert "name" in tier
    assert "intervals" in tier
    assert "xmin" in tier
    assert "xmax" in tier
    assert isinstance(tier["intervals"], list)
    assert len(tier["intervals"]) >= 1
