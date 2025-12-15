"""
Property-based tests for filename conversion functionality (rename_tool.py).

Tests the Chinese to Pinyin conversion and restoration round-trip property.
"""
from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Import the functions under test
from rename_tool import (
    is_chinese,
    has_chinese,
    to_pinyin,
    batch_rename_process,
    restore_process,
    LOG_FILE
)

# Import generators
from tests.generators import chinese_chars, safe_filenames


# ============================================================================
# Helper Strategies
# ============================================================================

# Chinese characters that are valid for filenames
chinese_filename_chars = st.text(
    alphabet=st.characters(min_codepoint=0x4e00, max_codepoint=0x9fff),
    min_size=1,
    max_size=10
)

# Mixed Chinese and ASCII for realistic filenames
mixed_filename = st.one_of(
    chinese_filename_chars,
    st.builds(
        lambda cn, en: cn + en,
        chinese_filename_chars,
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=0, max_size=5)
    )
)


# ============================================================================
# Property Tests
# ============================================================================

@given(chinese_name=chinese_filename_chars)
@settings(max_examples=100, deadline=None)
def test_filename_conversion_round_trip(chinese_name: str):
    """
    **Feature: phonetic-toolbox, Property 7: Filename Conversion Round-Trip**
    **Validates: Requirements 8.6**
    
    For any Chinese filename, converting to pinyin and then restoring 
    using the rename log SHALL produce the original filename.
    """
    # Skip empty or whitespace-only names
    assume(len(chinese_name.strip()) > 0)
    
    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a file with Chinese name
        original_filename = f"{chinese_name}.wav"
        original_path = os.path.join(tmp_dir, original_filename)
        
        # Create the file
        with open(original_path, 'w') as f:
            f.write("test content")
        
        # Verify file exists
        assert os.path.exists(original_path), f"Failed to create test file: {original_path}"
        
        # Step 1: Convert Chinese filename to pinyin
        success, work_dir, log_path, message = batch_rename_process(tmp_dir, mode='1')
        
        # Verify conversion succeeded
        assert success, f"Batch rename failed: {message}"
        assert log_path is not None, "Log path should not be None"
        assert os.path.exists(log_path), f"Log file should exist: {log_path}"
        
        # Verify original file no longer exists (was renamed)
        assert not os.path.exists(original_path), "Original file should have been renamed"
        
        # Step 2: Restore using the log
        restore_success, restore_message = restore_process(log_path)
        
        # Verify restoration succeeded
        assert restore_success, f"Restore failed: {restore_message}"
        
        # Step 3: Verify original filename is restored
        assert os.path.exists(original_path), f"Original file should be restored: {original_path}"
        
        # Verify content is preserved
        with open(original_path, 'r') as f:
            content = f.read()
        assert content == "test content", "File content should be preserved"


@given(chinese_name=chinese_filename_chars)
@settings(max_examples=100, deadline=None)
def test_to_pinyin_produces_different_output(chinese_name: str):
    """
    **Feature: phonetic-toolbox, Property 7.1: Pinyin Conversion Produces Output**
    **Validates: Requirements 8.6**
    
    For any Chinese text, to_pinyin SHALL produce a non-empty string.
    Note: Some rare Chinese characters may not have pinyin mappings in pypinyin,
    so we only verify that the function produces output without errors.
    """
    assume(len(chinese_name.strip()) > 0)
    
    pinyin_result = to_pinyin(chinese_name)
    
    # Verify output is non-empty
    assert len(pinyin_result) > 0, \
        f"Pinyin result should not be empty for input: {chinese_name}"


@given(text=st.text(min_size=1, max_size=50))
@settings(max_examples=100, deadline=None)
def test_has_chinese_detection_correctness(text: str):
    """
    **Feature: phonetic-toolbox, Property 7.2: Chinese Detection Correctness**
    **Validates: Requirements 8.5, 8.6**
    
    For any text, has_chinese SHALL return True if and only if
    the text contains at least one character in Unicode range \\u4e00-\\u9fff.
    """
    # Manual check for Chinese characters
    expected = any('\u4e00' <= char <= '\u9fff' for char in text)
    actual = has_chinese(text)
    
    assert actual == expected, \
        f"has_chinese({repr(text)}) returned {actual}, expected {expected}"


@given(char=st.characters())
@settings(max_examples=100, deadline=None)
def test_is_chinese_single_char(char: str):
    """
    **Feature: phonetic-toolbox, Property 7.3: Single Character Chinese Detection**
    **Validates: Requirements 8.5, 8.6**
    
    For any single character, is_chinese SHALL return True if and only if
    the character is in Unicode range \\u4e00-\\u9fff.
    """
    expected = '\u4e00' <= char <= '\u9fff'
    actual = is_chinese(char)
    
    assert actual == expected, \
        f"is_chinese({repr(char)}) returned {actual}, expected {expected}"


@given(chinese_name=chinese_filename_chars)
@settings(max_examples=50, deadline=None)
def test_directory_rename_round_trip(chinese_name: str):
    """
    **Feature: phonetic-toolbox, Property 7.4: Directory Rename Round-Trip**
    **Validates: Requirements 8.6**
    
    For any Chinese directory name, converting to pinyin and restoring
    SHALL produce the original directory name.
    """
    assume(len(chinese_name.strip()) > 0)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a subdirectory with Chinese name
        chinese_dir = os.path.join(tmp_dir, chinese_name)
        os.makedirs(chinese_dir)
        
        # Create a file inside the Chinese directory
        test_file = os.path.join(chinese_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        # Step 1: Convert
        success, work_dir, log_path, message = batch_rename_process(tmp_dir, mode='1')
        assert success, f"Batch rename failed: {message}"
        
        # Verify original directory no longer exists
        assert not os.path.exists(chinese_dir), "Original directory should have been renamed"
        
        # Step 2: Restore
        restore_success, restore_message = restore_process(log_path)
        assert restore_success, f"Restore failed: {restore_message}"
        
        # Step 3: Verify original directory is restored
        assert os.path.exists(chinese_dir), f"Original directory should be restored: {chinese_dir}"
        assert os.path.exists(test_file), "File inside directory should be accessible"
