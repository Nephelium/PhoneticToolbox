"""
Property-based tests for CSV IO functionality.

Tests the round-trip consistency of saving and loading acoustic parameter data.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.csv_io import save_csv, load_csv_any
from tests.generators import (
    acoustic_params_dict,
    f0_values,
    f1_values,
    f2_values,
    energy_values,
    frameshift_values,
)


# ============================================================================
# Property 3: CSV Round-Trip Consistency
# ============================================================================

@st.composite
def csv_param_dict(draw):
    """
    Generate a parameter dictionary suitable for CSV round-trip testing.
    
    Constraints:
    - All arrays must have the same length
    - No NaN or Inf values (CSV round-trip may not preserve these exactly)
    - Scalar values should be numeric
    """
    n_frames = draw(st.integers(min_value=5, max_value=100))
    
    # Generate arrays with finite values only
    float_elements = st.floats(
        min_value=-1e6, 
        max_value=1e6, 
        allow_nan=False, 
        allow_infinity=False
    )
    
    params = {
        "frameshift": float(draw(frameshift_values)),
        "pF0": draw(arrays(np.float64, n_frames, elements=f0_values)),
        "pF1": draw(arrays(np.float64, n_frames, elements=f1_values)),
        "pF2": draw(arrays(np.float64, n_frames, elements=f2_values)),
        "Energy": draw(arrays(np.float64, n_frames, elements=energy_values)),
    }
    
    return params


@given(params=csv_param_dict())
@settings(max_examples=100, deadline=None)
def test_csv_round_trip_consistency(params):
    """
    **Feature: phonetic-toolbox, Property 3: CSV Round-Trip Consistency**
    **Validates: Requirements 2.3, 13.2**
    
    For any valid acoustic parameter dictionary, saving it to CSV and loading
    it back SHALL produce an equivalent dictionary with all parameter arrays
    preserved within floating-point precision.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_params.csv"
    
        # Save the parameters to CSV
        save_csv(csv_path, params)
        
        # Verify file was created
        assert csv_path.exists(), "CSV file should be created"
        
        # Load the parameters back
        loaded = load_csv_any(csv_path)
        
        # Verify all keys are present
        for key in params:
            assert key in loaded, f"Key '{key}' should be present in loaded data"
        
        # Verify values match within floating-point precision
        for key, original_value in params.items():
            loaded_value = loaded[key]
            
            if isinstance(original_value, np.ndarray):
                # For arrays, compare element-wise with tolerance
                if isinstance(loaded_value, np.ndarray):
                    assert loaded_value.shape == original_value.shape, \
                        f"Array shape mismatch for '{key}': {loaded_value.shape} vs {original_value.shape}"
                    np.testing.assert_allclose(
                        loaded_value, 
                        original_value, 
                        rtol=1e-6, 
                        atol=1e-10,
                        err_msg=f"Array values mismatch for '{key}'"
                    )
                else:
                    # Loaded as scalar but original was array - this shouldn't happen
                    # unless all values were identical
                    if len(original_value) == 1 or np.all(original_value == original_value[0]):
                        # Constant array loaded as scalar is acceptable
                        np.testing.assert_allclose(
                            float(loaded_value),
                            original_value[0],
                            rtol=1e-6,
                            atol=1e-10,
                            err_msg=f"Scalar/array mismatch for '{key}'"
                        )
                    else:
                        pytest.fail(f"Expected array for '{key}', got scalar: {loaded_value}")
            else:
                # For scalars, compare directly with tolerance
                if isinstance(loaded_value, np.ndarray):
                    # Scalar saved as constant column, loaded as array
                    # All values should be the same as the original scalar
                    assert np.allclose(loaded_value, original_value, rtol=1e-6, atol=1e-10), \
                        f"Scalar '{key}' loaded as array with different values"
                else:
                    np.testing.assert_allclose(
                        float(loaded_value),
                        float(original_value),
                        rtol=1e-6,
                        atol=1e-10,
                        err_msg=f"Scalar value mismatch for '{key}'"
                    )


@given(params=csv_param_dict())
@settings(max_examples=100, deadline=None)
def test_csv_array_length_preserved(params):
    """
    **Feature: phonetic-toolbox, Property 3: CSV Round-Trip Consistency (Array Length)**
    **Validates: Requirements 2.3, 13.2**
    
    For any parameter dictionary with arrays, the array lengths SHALL be
    preserved after CSV round-trip.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_length.csv"
        
        # Get original array lengths
        original_lengths = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                original_lengths[key] = len(value)
        
        # Save and load
        save_csv(csv_path, params)
        loaded = load_csv_any(csv_path)
        
        # Verify array lengths
        for key, expected_len in original_lengths.items():
            loaded_value = loaded.get(key)
            if isinstance(loaded_value, np.ndarray):
                assert len(loaded_value) == expected_len, \
                    f"Array length mismatch for '{key}': {len(loaded_value)} vs {expected_len}"


@given(
    n_frames=st.integers(min_value=1, max_value=50),
    n_params=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100, deadline=None)
def test_csv_multiple_arrays_same_length(n_frames, n_params):
    """
    **Feature: phonetic-toolbox, Property 3: CSV Round-Trip Consistency (Multiple Arrays)**
    **Validates: Requirements 2.3, 13.2**
    
    For any dictionary with multiple arrays of the same length, all arrays
    SHALL maintain their length after CSV round-trip.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generate multiple parameter arrays
        params = {}
        for i in range(n_params):
            params[f"param_{i}"] = np.random.uniform(-100, 100, n_frames)
        
        csv_path = Path(tmp_dir) / "test_multi.csv"
        
        # Save and load
        save_csv(csv_path, params)
        loaded = load_csv_any(csv_path)
        
        # Verify all arrays have correct length
        for key in params:
            loaded_value = loaded.get(key)
            assert loaded_value is not None, f"Key '{key}' should be present"
            if isinstance(loaded_value, np.ndarray):
                assert len(loaded_value) == n_frames, \
                    f"Array '{key}' length should be {n_frames}, got {len(loaded_value)}"
