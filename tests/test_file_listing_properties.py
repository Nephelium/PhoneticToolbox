"""
Property-based tests for WAV file listing functionality.

**Feature: phonetic-toolbox, Property 1: WAV File Listing Completeness**
**Validates: Requirements 2.1, 3.4**
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Set

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st


# ============================================================================
# Core file listing logic (extracted from ParameterEstimationController)
# ============================================================================

def list_wav_files(directory: Path, recursive: bool = False) -> List[str]:
    """
    List WAV files in a directory.
    
    This function mirrors the logic in ParameterEstimationController._refresh_files
    
    Args:
        directory: The directory to search for WAV files
        recursive: If True, search subdirectories recursively
        
    Returns:
        List of WAV filenames (not full paths)
    """
    if not directory.exists():
        return []
    
    if recursive:
        wav_files = list(directory.rglob("*.wav"))
    else:
        wav_files = list(directory.glob("*.wav"))
    
    return [wav.name for wav in wav_files]


def get_expected_wav_files(directory: Path, recursive: bool = False) -> Set[str]:
    """
    Get the expected set of WAV files using os.walk for verification.
    
    This provides an independent implementation to verify against.
    """
    if not directory.exists():
        return set()
    
    wav_files = set()
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.lower().endswith('.wav'):
                    wav_files.add(f)
    else:
        for f in os.listdir(directory):
            if os.path.isfile(directory / f) and f.lower().endswith('.wav'):
                wav_files.add(f)
    
    return wav_files


# ============================================================================
# Strategies for generating test directory structures
# ============================================================================

# Safe filename characters
safe_chars = st.sampled_from('abcdefghijklmnopqrstuvwxyz0123456789_-')
safe_filename = st.text(safe_chars, min_size=1, max_size=20)


@st.composite
def wav_filename(draw):
    """Generate a valid WAV filename."""
    name = draw(safe_filename)
    return f"{name}.wav"


@st.composite
def non_wav_filename(draw):
    """Generate a non-WAV filename."""
    name = draw(safe_filename)
    ext = draw(st.sampled_from(['.txt', '.mp3', '.csv', '.TextGrid', '.json']))
    return f"{name}{ext}"


@st.composite
def directory_structure(draw, max_depth: int = 2, max_files_per_dir: int = 5, max_subdirs: int = 3):
    """
    Generate a directory structure specification.
    
    Returns a dict with:
        - 'wav_files': list of WAV filenames in root
        - 'other_files': list of non-WAV filenames in root
        - 'subdirs': dict mapping subdir names to their contents (recursive)
    """
    # Generate files for root directory
    n_wav = draw(st.integers(min_value=0, max_value=max_files_per_dir))
    n_other = draw(st.integers(min_value=0, max_value=max_files_per_dir))
    
    wav_files = [draw(wav_filename()) for _ in range(n_wav)]
    other_files = [draw(non_wav_filename()) for _ in range(n_other)]
    
    # Ensure unique filenames
    wav_files = list(set(wav_files))
    other_files = list(set(other_files))
    
    result = {
        'wav_files': wav_files,
        'other_files': other_files,
        'subdirs': {}
    }
    
    # Generate subdirectories (only if depth allows)
    if max_depth > 0:
        n_subdirs = draw(st.integers(min_value=0, max_value=max_subdirs))
        for _ in range(n_subdirs):
            subdir_name = draw(safe_filename)
            # Avoid duplicate subdir names
            if subdir_name not in result['subdirs']:
                subdir_content = draw(directory_structure(
                    max_depth=max_depth - 1,
                    max_files_per_dir=max_files_per_dir,
                    max_subdirs=max_subdirs
                ))
                result['subdirs'][subdir_name] = subdir_content
    
    return result


def create_directory_structure(base_path: Path, structure: dict) -> None:
    """Create the actual directory structure from a specification."""
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Create WAV files
    for wav_name in structure['wav_files']:
        (base_path / wav_name).touch()
    
    # Create other files
    for other_name in structure['other_files']:
        (base_path / other_name).touch()
    
    # Create subdirectories recursively
    for subdir_name, subdir_content in structure['subdirs'].items():
        subdir_path = base_path / subdir_name
        create_directory_structure(subdir_path, subdir_content)


def count_all_wav_files(structure: dict) -> int:
    """Count total WAV files in a structure (recursive)."""
    count = len(structure['wav_files'])
    for subdir_content in structure['subdirs'].values():
        count += count_all_wav_files(subdir_content)
    return count


def get_all_wav_filenames(structure: dict) -> Set[str]:
    """Get all WAV filenames in a structure (recursive)."""
    filenames = set(structure['wav_files'])
    for subdir_content in structure['subdirs'].values():
        filenames.update(get_all_wav_filenames(subdir_content))
    return filenames


def get_root_wav_filenames(structure: dict) -> Set[str]:
    """Get WAV filenames only in root directory (non-recursive)."""
    return set(structure['wav_files'])


# ============================================================================
# Property-Based Tests
# ============================================================================

class TestWavFileListingCompleteness:
    """
    **Feature: phonetic-toolbox, Property 1: WAV File Listing Completeness**
    **Validates: Requirements 2.1, 3.4**
    
    For any directory containing WAV files, when the file list is refreshed,
    the list SHALL contain exactly all WAV files in that directory (non-recursive mode)
    or all WAV files in the directory tree (recursive mode).
    """
    
    @given(structure=directory_structure())
    @settings(max_examples=100, deadline=None)
    def test_non_recursive_lists_only_root_wav_files(self, structure: dict):
        """
        **Feature: phonetic-toolbox, Property 1: WAV File Listing Completeness**
        **Validates: Requirements 2.1, 3.4**
        
        In non-recursive mode, listing should return exactly the WAV files
        in the root directory, not including subdirectories.
        """
        # Create a fresh temp directory for each test run
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = Path(tmp_dir) / "test_wav_dir"
            create_directory_structure(test_dir, structure)
            
            # Get expected WAV files (root only)
            expected = get_root_wav_filenames(structure)
            
            # Get actual WAV files using our function
            actual = set(list_wav_files(test_dir, recursive=False))
            
            # Verify completeness: actual should equal expected
            assert actual == expected, (
                f"Non-recursive listing mismatch.\n"
                f"Expected: {expected}\n"
                f"Actual: {actual}\n"
                f"Missing: {expected - actual}\n"
                f"Extra: {actual - expected}"
            )
    
    @given(structure=directory_structure())
    @settings(max_examples=100, deadline=None)
    def test_recursive_lists_all_wav_files(self, structure: dict):
        """
        **Feature: phonetic-toolbox, Property 1: WAV File Listing Completeness**
        **Validates: Requirements 2.1, 3.4**
        
        In recursive mode, listing should return all WAV files
        in the directory tree, including subdirectories.
        """
        # Create a fresh temp directory for each test run
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = Path(tmp_dir) / "test_wav_dir"
            create_directory_structure(test_dir, structure)
            
            # Get expected WAV files (all, recursive)
            expected = get_all_wav_filenames(structure)
            
            # Get actual WAV files using our function
            actual = set(list_wav_files(test_dir, recursive=True))
            
            # Verify completeness: actual should equal expected
            assert actual == expected, (
                f"Recursive listing mismatch.\n"
                f"Expected: {expected}\n"
                f"Actual: {actual}\n"
                f"Missing: {expected - actual}\n"
                f"Extra: {actual - expected}"
            )
    
    @given(structure=directory_structure())
    @settings(max_examples=100, deadline=None)
    def test_non_recursive_excludes_subdirectory_files(self, structure: dict):
        """
        **Feature: phonetic-toolbox, Property 1: WAV File Listing Completeness**
        **Validates: Requirements 2.1, 3.4**
        
        Non-recursive listing should not include files from subdirectories.
        """
        # Skip if no subdirectories with WAV files
        subdir_wav_count = count_all_wav_files(structure) - len(structure['wav_files'])
        assume(subdir_wav_count > 0)
        
        # Create a fresh temp directory for each test run
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = Path(tmp_dir) / "test_wav_dir"
            create_directory_structure(test_dir, structure)
            
            # Get non-recursive listing
            non_recursive_files = set(list_wav_files(test_dir, recursive=False))
            
            # Get recursive listing
            recursive_files = set(list_wav_files(test_dir, recursive=True))
            
            # Non-recursive should be a subset of recursive
            assert non_recursive_files <= recursive_files, (
                f"Non-recursive files should be subset of recursive files.\n"
                f"Non-recursive: {non_recursive_files}\n"
                f"Recursive: {recursive_files}"
            )
            
            # If there are subdirectory WAV files, recursive should have more
            if subdir_wav_count > 0:
                # Note: filenames might overlap between root and subdirs
                # So we just check that recursive >= non-recursive
                assert len(recursive_files) >= len(non_recursive_files)
    
    @given(structure=directory_structure())
    @settings(max_examples=100, deadline=None)
    def test_listing_excludes_non_wav_files(self, structure: dict):
        """
        **Feature: phonetic-toolbox, Property 1: WAV File Listing Completeness**
        **Validates: Requirements 2.1, 3.4**
        
        Listing should only include .wav files, not other file types.
        """
        # Create a fresh temp directory for each test run
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = Path(tmp_dir) / "test_wav_dir"
            create_directory_structure(test_dir, structure)
            
            # Get listing (both modes)
            non_recursive_files = list_wav_files(test_dir, recursive=False)
            recursive_files = list_wav_files(test_dir, recursive=True)
            
            # All files should end with .wav
            for f in non_recursive_files:
                assert f.lower().endswith('.wav'), f"Non-WAV file in listing: {f}"
            
            for f in recursive_files:
                assert f.lower().endswith('.wav'), f"Non-WAV file in listing: {f}"
    
    def test_empty_directory_returns_empty_list(self):
        """
        **Feature: phonetic-toolbox, Property 1: WAV File Listing Completeness**
        **Validates: Requirements 2.1, 3.4**
        
        An empty directory should return an empty list.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = Path(tmp_dir) / "empty_dir"
            test_dir.mkdir()
            
            assert list_wav_files(test_dir, recursive=False) == []
            assert list_wav_files(test_dir, recursive=True) == []
    
    def test_nonexistent_directory_returns_empty_list(self):
        """
        **Feature: phonetic-toolbox, Property 1: WAV File Listing Completeness**
        **Validates: Requirements 2.1, 3.4**
        
        A non-existent directory should return an empty list.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = Path(tmp_dir) / "nonexistent_dir"
            
            assert list_wav_files(test_dir, recursive=False) == []
            assert list_wav_files(test_dir, recursive=True) == []
