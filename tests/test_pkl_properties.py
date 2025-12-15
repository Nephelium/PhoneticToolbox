"""
Property-based tests for PKL data loading functionality.

**Feature: phonetic-toolbox, Property 11: PKL Data Loading Consistency**
**Validates: Requirements 13.3**
"""
from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from contextlib import contextmanager

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from utils.lip_reader import read_lip_data


# ============================================================================
# Helper context manager for temporary directories
# ============================================================================

@contextmanager
def temp_directory():
    """Create a temporary directory that is cleaned up after use."""
    tmp_dir = tempfile.mkdtemp()
    try:
        yield Path(tmp_dir)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================================
# Strategies for generating valid lip tracking PKL data
# ============================================================================

@st.composite
def lip_tracking_data(draw, min_frames: int = 5, max_frames: int = 100):
    """
    Generate valid lip tracking data structure matching the expected format.
    
    The PKL file should contain:
    - absolute_timestamps or relative_times: array of timestamps
    - area: lip area ratio values
    - outer_width: normalized outer lip width values
    - open: normalized lip openness values
    - circularity: lip circularity values
    """
    n_frames = draw(st.integers(min_value=min_frames, max_value=max_frames))
    
    # Generate timestamps (monotonically increasing)
    start_time = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    frame_interval = draw(st.floats(min_value=0.01, max_value=0.1, allow_nan=False, allow_infinity=False))
    
    relative_times = [i * frame_interval for i in range(n_frames)]
    absolute_timestamps = [start_time + t for t in relative_times]
    
    # Generate lip metric values (normalized 0-1 range)
    area = draw(arrays(
        dtype=np.float64,
        shape=n_frames,
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))
    
    outer_width = draw(arrays(
        dtype=np.float64,
        shape=n_frames,
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))
    
    open_vals = draw(arrays(
        dtype=np.float64,
        shape=n_frames,
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))
    
    circularity = draw(arrays(
        dtype=np.float64,
        shape=n_frames,
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))
    
    return {
        'absolute_timestamps': absolute_timestamps,
        'relative_times': relative_times,
        'area': area.tolist(),
        'outer_width': outer_width.tolist(),
        'open': open_vals.tolist(),
        'circularity': circularity.tolist(),
    }


@st.composite
def target_times_array(draw, max_duration: float = 10.0, min_points: int = 5, max_points: int = 100):
    """Generate an array of target timestamps for interpolation."""
    n_points = draw(st.integers(min_value=min_points, max_value=max_points))
    duration = draw(st.floats(min_value=0.1, max_value=max_duration, allow_nan=False, allow_infinity=False))
    return np.linspace(0, duration, n_points)


# ============================================================================
# Property Tests
# ============================================================================

class TestPKLDataLoadingProperties:
    """
    Property-based tests for PKL data loading consistency.
    
    **Feature: phonetic-toolbox, Property 11: PKL Data Loading Consistency**
    **Validates: Requirements 13.3**
    """
    
    @given(data=lip_tracking_data())
    @settings(max_examples=100, deadline=None)
    def test_pkl_loading_produces_expected_keys(self, data: Dict[str, Any]):
        """
        **Feature: phonetic-toolbox, Property 11: PKL Data Loading Consistency**
        **Validates: Requirements 13.3**
        
        For any valid PKL file containing lip tracking data, loading SHALL produce
        a data structure with the expected keys (LipArea, LipWidth, LipOpen, LipCirc).
        """
        with temp_directory() as tmp_path:
            # Create temporary PKL file
            pkl_path = tmp_path / "test_lip_data.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Create target times within the data's time range
            relative_times = data['relative_times']
            if len(relative_times) < 2:
                return  # Skip if not enough data points
            
            max_time = max(relative_times)
            target_times = np.linspace(0, max_time * 0.9, 20)  # Stay within bounds
            
            # Load the data
            result = read_lip_data(str(pkl_path), target_times)
            
            # Verify expected keys are present
            expected_keys = {'LipArea', 'LipWidth', 'LipOpen', 'LipCirc'}
            assert set(result.keys()) == expected_keys, \
                f"Expected keys {expected_keys}, got {set(result.keys())}"
    
    @given(data=lip_tracking_data())
    @settings(max_examples=100, deadline=None)
    def test_pkl_loading_produces_correct_array_shapes(self, data: Dict[str, Any]):
        """
        **Feature: phonetic-toolbox, Property 11: PKL Data Loading Consistency**
        **Validates: Requirements 13.3**
        
        For any valid PKL file, the loaded arrays SHALL have shapes matching
        the target_times array length.
        """
        with temp_directory() as tmp_path:
            # Create temporary PKL file
            pkl_path = tmp_path / "test_lip_data.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Create target times
            relative_times = data['relative_times']
            if len(relative_times) < 2:
                return
            
            max_time = max(relative_times)
            n_target_points = 25
            target_times = np.linspace(0, max_time * 0.9, n_target_points)
            
            # Load the data
            result = read_lip_data(str(pkl_path), target_times)
            
            # Verify all arrays have the correct shape
            for key, arr in result.items():
                assert isinstance(arr, np.ndarray), f"{key} should be numpy array"
                assert arr.shape == (n_target_points,), \
                    f"{key} shape {arr.shape} != expected ({n_target_points},)"
    
    @given(data=lip_tracking_data(), smooth_win=st.integers(min_value=0, max_value=5))
    @settings(max_examples=100, deadline=None)
    def test_pkl_loading_with_smoothing_preserves_structure(
        self, data: Dict[str, Any], smooth_win: int
    ):
        """
        **Feature: phonetic-toolbox, Property 11: PKL Data Loading Consistency**
        **Validates: Requirements 13.3**
        
        For any valid PKL file and smoothing window, loading SHALL produce
        arrays with the same structure regardless of smoothing parameter.
        """
        with temp_directory() as tmp_path:
            # Create temporary PKL file
            pkl_path = tmp_path / "test_lip_data.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Create target times
            relative_times = data['relative_times']
            if len(relative_times) < 2:
                return
            
            max_time = max(relative_times)
            n_target_points = 30
            target_times = np.linspace(0, max_time * 0.9, n_target_points)
            
            # Load with smoothing
            result = read_lip_data(str(pkl_path), target_times, smooth_win=smooth_win)
            
            # Verify structure is preserved
            expected_keys = {'LipArea', 'LipWidth', 'LipOpen', 'LipCirc'}
            assert set(result.keys()) == expected_keys
            
            for key, arr in result.items():
                assert arr.shape == (n_target_points,)
                # Values should be finite (no inf)
                assert np.all(np.isfinite(arr) | np.isnan(arr)), \
                    f"{key} contains infinite values"
    
    @given(data=lip_tracking_data())
    @settings(max_examples=100, deadline=None)
    def test_pkl_loading_values_in_valid_range(self, data: Dict[str, Any]):
        """
        **Feature: phonetic-toolbox, Property 11: PKL Data Loading Consistency**
        **Validates: Requirements 13.3**
        
        For any valid PKL file with normalized values (0-1), loaded values
        SHALL remain within a reasonable range after interpolation.
        """
        with temp_directory() as tmp_path:
            # Create temporary PKL file
            pkl_path = tmp_path / "test_lip_data.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Create target times within data range
            relative_times = data['relative_times']
            if len(relative_times) < 2:
                return
            
            min_time = min(relative_times)
            max_time = max(relative_times)
            # Stay strictly within bounds to avoid extrapolation NaNs
            target_times = np.linspace(min_time + 0.001, max_time - 0.001, 20)
            
            # Load the data
            result = read_lip_data(str(pkl_path), target_times)
            
            # Verify values are in valid range (allowing for interpolation)
            for key, arr in result.items():
                valid_mask = ~np.isnan(arr)
                if np.any(valid_mask):
                    valid_values = arr[valid_mask]
                    # Interpolated values should be within [0, 1] for normalized data
                    assert np.all(valid_values >= -0.01), \
                        f"{key} has values below 0: {valid_values.min()}"
                    assert np.all(valid_values <= 1.01), \
                        f"{key} has values above 1: {valid_values.max()}"


class TestPKLDataLoadingEdgeCases:
    """Edge case tests for PKL data loading."""
    
    def test_nonexistent_file_returns_empty_dict(self, tmp_path: Path):
        """Loading a non-existent file should return an empty dictionary."""
        fake_path = tmp_path / "nonexistent.pkl"
        target_times = np.linspace(0, 1, 10)
        
        result = read_lip_data(str(fake_path), target_times)
        
        assert result == {}
    
    def test_empty_pkl_returns_empty_dict(self, tmp_path: Path):
        """Loading a PKL file with no valid timestamps should return empty dict."""
        pkl_path = tmp_path / "empty.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump({}, f)
        
        target_times = np.linspace(0, 1, 10)
        result = read_lip_data(str(pkl_path), target_times)
        
        assert result == {}
    
    def test_pkl_with_only_relative_times(self, tmp_path: Path):
        """Loading PKL with only relative_times (no absolute) should work."""
        data = {
            'relative_times': [0.0, 0.1, 0.2, 0.3, 0.4],
            'area': [0.1, 0.2, 0.3, 0.4, 0.5],
            'outer_width': [0.2, 0.3, 0.4, 0.5, 0.6],
            'open': [0.3, 0.4, 0.5, 0.6, 0.7],
            'circularity': [0.4, 0.5, 0.6, 0.7, 0.8],
        }
        
        pkl_path = tmp_path / "relative_only.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        
        target_times = np.linspace(0.05, 0.35, 10)
        result = read_lip_data(str(pkl_path), target_times)
        
        assert set(result.keys()) == {'LipArea', 'LipWidth', 'LipOpen', 'LipCirc'}
        for arr in result.values():
            assert arr.shape == (10,)
