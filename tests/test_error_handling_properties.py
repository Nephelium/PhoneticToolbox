"""
Property-based tests for error handling mechanisms.

This module tests:
- Property 12: Error Detection for Invalid Paths
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Import the project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.textgrid_parser import parse_textgrid
from utils.csv_io import load_csv_any
from rename_tool import batch_rename_process, restore_process


# ============================================================================
# Generators for error handling testing
# ============================================================================

# Generate safe filenames that won't exist
safe_filename_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
nonexistent_filenames = st.text(
    alphabet=safe_filename_chars,
    min_size=5,
    max_size=30
).filter(lambda x: len(x.strip()) > 0)


# ============================================================================
# Property 12: Error Detection for Invalid Paths
# ============================================================================

@given(filename=nonexistent_filenames)
@settings(max_examples=100)
def test_textgrid_parser_returns_none_for_nonexistent_path(filename: str):
    """
    **Feature: phonetic-toolbox, Property 12: Error Detection for Invalid Paths**
    **Validates: Requirements 14.1, 14.2**
    
    For any non-existent file path, parse_textgrid SHALL return None
    rather than raising an unhandled exception.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a path that doesn't exist
        nonexistent_path = Path(tmp_dir) / f"{filename}.TextGrid"
        
        # Ensure the path doesn't exist
        assume(not nonexistent_path.exists())
        
        # parse_textgrid should return None for non-existent paths
        result = parse_textgrid(nonexistent_path)
        assert result is None


@given(filename=nonexistent_filenames)
@settings(max_examples=100)
def test_csv_loader_returns_empty_dict_for_nonexistent_path(filename: str):
    """
    **Feature: phonetic-toolbox, Property 12: Error Detection for Invalid Paths**
    **Validates: Requirements 14.1, 14.2**
    
    For any non-existent file path, load_csv_any SHALL return an empty
    dictionary rather than raising an unhandled exception.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a path that doesn't exist
        nonexistent_path = Path(tmp_dir) / f"{filename}.csv"
        
        # Ensure the path doesn't exist
        assume(not nonexistent_path.exists())
        
        # load_csv_any should return empty dict for non-existent paths
        result = load_csv_any(nonexistent_path)
        assert result == {}


@given(dirname=nonexistent_filenames)
@settings(max_examples=100)
def test_batch_rename_returns_error_for_nonexistent_directory(dirname: str):
    """
    **Feature: phonetic-toolbox, Property 12: Error Detection for Invalid Paths**
    **Validates: Requirements 14.1, 14.2**
    
    For any non-existent directory path, batch_rename_process SHALL return
    a failure status (False) with an appropriate error message.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a path that doesn't exist
        nonexistent_dir = os.path.join(tmp_dir, dirname)
        
        # Ensure the path doesn't exist
        assume(not os.path.exists(nonexistent_dir))
        
        # batch_rename_process should return failure for non-existent paths
        success, work_dir, log_path, message = batch_rename_process(nonexistent_dir)
        
        assert success is False
        assert work_dir is None
        assert log_path is None
        assert "不存在" in message or "not found" in message.lower()


@given(filename=nonexistent_filenames)
@settings(max_examples=100)
def test_restore_process_returns_error_for_nonexistent_log(filename: str):
    """
    **Feature: phonetic-toolbox, Property 12: Error Detection for Invalid Paths**
    **Validates: Requirements 14.1, 14.2**
    
    For any non-existent log file path, restore_process SHALL return
    a failure status (False) with an appropriate error message.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a path that doesn't exist
        nonexistent_log = os.path.join(tmp_dir, f"{filename}.json")
        
        # Ensure the path doesn't exist
        assume(not os.path.exists(nonexistent_log))
        
        # restore_process should return failure for non-existent paths
        success, message = restore_process(nonexistent_log)
        
        assert success is False
        assert "找不到" in message or "not found" in message.lower()


@given(filename=nonexistent_filenames)
@settings(max_examples=100)
def test_textgrid_parser_handles_invalid_content_gracefully(filename: str):
    """
    **Feature: phonetic-toolbox, Property 12: Error Detection for Invalid Paths**
    **Validates: Requirements 14.1, 14.2**
    
    For any file with invalid TextGrid content, parse_textgrid SHALL return
    None rather than raising an unhandled exception.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a file with invalid content
        invalid_file = Path(tmp_dir) / f"{filename}.TextGrid"
        invalid_file.write_text("This is not a valid TextGrid file content")
        
        # parse_textgrid should return None for invalid content
        result = parse_textgrid(invalid_file)
        assert result is None


@given(content=st.text(
    alphabet=st.characters(min_codepoint=0x0020, max_codepoint=0x007F),  # ASCII only
    min_size=0,
    max_size=100
))
@settings(max_examples=100)
def test_csv_loader_handles_invalid_content_gracefully(content: str):
    """
    **Feature: phonetic-toolbox, Property 12: Error Detection for Invalid Paths**
    **Validates: Requirements 14.1, 14.2**
    
    For any file with arbitrary ASCII content, load_csv_any SHALL not raise
    an unhandled exception (it may return empty dict or partial data).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a file with arbitrary content
        test_file = Path(tmp_dir) / "test.csv"
        test_file.write_text(content, encoding='utf-8')
        
        # load_csv_any should not raise an exception
        try:
            result = load_csv_any(test_file)
            # Result should be a dictionary (possibly empty)
            assert isinstance(result, dict)
        except Exception as e:
            # If an exception is raised, it should be a known, handled type
            # For now, we fail the test if any unhandled exception occurs
            pytest.fail(f"load_csv_any raised unhandled exception: {type(e).__name__}: {e}")


# ============================================================================
# Additional error handling tests for path validation
# ============================================================================

@given(filename=nonexistent_filenames)
@settings(max_examples=100)
def test_path_exists_returns_false_for_nonexistent_paths(filename: str):
    """
    **Feature: phonetic-toolbox, Property 12: Error Detection for Invalid Paths**
    **Validates: Requirements 14.1, 14.2**
    
    For any non-existent path, os.path.exists SHALL return False.
    This is a fundamental property that other error handling relies on.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a path that doesn't exist
        nonexistent_path = os.path.join(tmp_dir, f"nonexistent_{filename}.txt")
        
        # Ensure the path doesn't exist
        assume(not os.path.exists(nonexistent_path))
        
        # os.path.exists should return False
        assert os.path.exists(nonexistent_path) is False


@given(filename=nonexistent_filenames)
@settings(max_examples=100)
def test_path_exists_returns_true_for_existing_paths(filename: str):
    """
    **Feature: phonetic-toolbox, Property 12: Error Detection for Invalid Paths**
    **Validates: Requirements 14.1, 14.2**
    
    For any existing path, os.path.exists SHALL return True.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a file
        existing_path = os.path.join(tmp_dir, f"{filename}.txt")
        with open(existing_path, 'w') as f:
            f.write("test content")
        
        # os.path.exists should return True
        assert os.path.exists(existing_path) is True
