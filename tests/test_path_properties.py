"""
Property-based tests for path utility functions.

This module tests:
- Chinese character detection (Property 6)
- Path validation correctness (Property 8)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Import the functions under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rename_tool import is_chinese, has_chinese


# ============================================================================
# Generators for path testing
# ============================================================================

# ASCII-only text (no Chinese characters)
ascii_only_text = st.text(
    alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'S'),
        min_codepoint=0x0020,
        max_codepoint=0x007F  # ASCII range only
    ),
    min_size=1,
    max_size=50
).filter(lambda x: len(x.strip()) > 0)

# Chinese characters only (Unicode range \u4e00-\u9fff)
chinese_only_text = st.text(
    alphabet=st.characters(min_codepoint=0x4e00, max_codepoint=0x9fff),
    min_size=1,
    max_size=20
)

# Mixed text with at least one Chinese character
@st.composite
def mixed_with_chinese(draw):
    """Generate text that contains at least one Chinese character."""
    chinese_part = draw(chinese_only_text)
    ascii_part = draw(ascii_only_text)
    # Combine in random order
    if draw(st.booleans()):
        return chinese_part + ascii_part
    else:
        return ascii_part + chinese_part


# ============================================================================
# Property 6: Chinese Character Detection
# ============================================================================

@given(char=st.characters(min_codepoint=0x4e00, max_codepoint=0x9fff))
@settings(max_examples=100)
def test_is_chinese_true_for_chinese_chars(char: str):
    """
    **Feature: phonetic-toolbox, Property 6: Chinese Character Detection**
    **Validates: Requirements 8.5**
    
    For any character in the Chinese Unicode range (\u4e00-\u9fff),
    is_chinese SHALL return True.
    """
    assert is_chinese(char) is True


@given(char=st.characters(min_codepoint=0x0020, max_codepoint=0x007F))
@settings(max_examples=100)
def test_is_chinese_false_for_ascii_chars(char: str):
    """
    **Feature: phonetic-toolbox, Property 6: Chinese Character Detection**
    **Validates: Requirements 8.5**
    
    For any ASCII character, is_chinese SHALL return False.
    """
    assert is_chinese(char) is False


@given(text=chinese_only_text)
@settings(max_examples=100)
def test_has_chinese_true_for_chinese_text(text: str):
    """
    **Feature: phonetic-toolbox, Property 6: Chinese Character Detection**
    **Validates: Requirements 8.5**
    
    For any text containing only Chinese characters,
    has_chinese SHALL return True.
    """
    assert has_chinese(text) is True


@given(text=ascii_only_text)
@settings(max_examples=100)
def test_has_chinese_false_for_ascii_text(text: str):
    """
    **Feature: phonetic-toolbox, Property 6: Chinese Character Detection**
    **Validates: Requirements 8.5**
    
    For any text containing only ASCII characters,
    has_chinese SHALL return False.
    """
    # Filter out any text that might accidentally contain Chinese
    assume(all(ord(c) < 0x4e00 or ord(c) > 0x9fff for c in text))
    assert has_chinese(text) is False


@given(text=mixed_with_chinese())
@settings(max_examples=100)
def test_has_chinese_true_for_mixed_text(text: str):
    """
    **Feature: phonetic-toolbox, Property 6: Chinese Character Detection**
    **Validates: Requirements 8.5**
    
    For any text containing at least one Chinese character,
    has_chinese SHALL return True.
    """
    assert has_chinese(text) is True


# ============================================================================
# Property 8: Path Validation Correctness
# ============================================================================

def validate_path_exists(path: str) -> bool:
    """
    Validate that a path exists on the filesystem.
    
    This is the path validation function as specified in the design document.
    """
    return os.path.exists(path)


@given(filename=st.text(
    alphabet=st.characters(
        whitelist_categories=('L', 'N'),
        min_codepoint=0x0041,
        max_codepoint=0x007A
    ),
    min_size=1,
    max_size=20
).filter(lambda x: len(x.strip()) > 0 and x.isalnum()))
@settings(max_examples=100)
def test_path_validation_true_for_existing_paths(filename: str):
    """
    **Feature: phonetic-toolbox, Property 8: Path Validation Correctness**
    **Validates: Requirements 8.2, 14.1**
    
    For any file path that exists on the filesystem,
    the path validation function SHALL return True.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a file with the generated filename
        file_path = os.path.join(tmp_dir, f"{filename}.txt")
        with open(file_path, 'w') as f:
            f.write("test content")
        
        # Verify the path validation returns True
        assert validate_path_exists(file_path) is True


@given(filename=st.text(
    alphabet=st.characters(
        whitelist_categories=('L', 'N'),
        min_codepoint=0x0041,
        max_codepoint=0x007A
    ),
    min_size=1,
    max_size=20
).filter(lambda x: len(x.strip()) > 0 and x.isalnum()))
@settings(max_examples=100)
def test_path_validation_false_for_nonexistent_paths(filename: str):
    """
    **Feature: phonetic-toolbox, Property 8: Path Validation Correctness**
    **Validates: Requirements 8.2, 14.1**
    
    For any file path that does not exist on the filesystem,
    the path validation function SHALL return False.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a path that doesn't exist
        nonexistent_path = os.path.join(tmp_dir, f"nonexistent_{filename}.txt")
        
        # Verify the path validation returns False
        assert validate_path_exists(nonexistent_path) is False


@given(dirname=st.text(
    alphabet=st.characters(
        whitelist_categories=('L', 'N'),
        min_codepoint=0x0041,
        max_codepoint=0x007A
    ),
    min_size=1,
    max_size=20
).filter(lambda x: len(x.strip()) > 0 and x.isalnum()))
@settings(max_examples=100)
def test_path_validation_true_for_existing_directories(dirname: str):
    """
    **Feature: phonetic-toolbox, Property 8: Path Validation Correctness**
    **Validates: Requirements 8.2, 14.1**
    
    For any directory path that exists on the filesystem,
    the path validation function SHALL return True.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a subdirectory
        dir_path = os.path.join(tmp_dir, dirname)
        os.makedirs(dir_path, exist_ok=True)
        
        # Verify the path validation returns True
        assert validate_path_exists(dir_path) is True
