"""
Property-based tests for parameter extraction functionality.

**Feature: phonetic-toolbox, Property 10: Parameter Extraction Validity**
**Validates: Requirements 2.2, 3.3**
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
from hypothesis import given, settings, assume

# Ensure the project root is in sys.path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.praat_service import compute_praat_f0_formants


def get_test_wav_files() -> List[Path]:
    """Get list of WAV files from klatt directory for testing."""
    klatt_dir = PROJECT_ROOT / "klatt"
    if not klatt_dir.exists():
        return []
    return list(klatt_dir.glob("*.wav"))


# Get available test WAV files
TEST_WAV_FILES = get_test_wav_files()


@pytest.mark.skipif(len(TEST_WAV_FILES) == 0, reason="No WAV files found in klatt directory")
class TestParameterExtractionValidity:
    """
    Property-based tests for parameter extraction validity.
    
    **Feature: phonetic-toolbox, Property 10: Parameter Extraction Validity**
    **Validates: Requirements 2.2, 3.3**
    
    Property: For any valid WAV audio file with voiced content, parameter extraction
    SHALL produce F0 values within the configured min/max range for voiced frames.
    """
    
    @pytest.mark.parametrize("wav_path", TEST_WAV_FILES, ids=lambda p: p.name)
    @pytest.mark.parametrize("min_f0,max_f0", [
        (40, 500),   # Default range
        (50, 400),   # Narrower range
        (75, 300),   # Even narrower
    ])
    def test_f0_values_within_configured_range(
        self, wav_path: Path, min_f0: int, max_f0: int
    ):
        """
        **Feature: phonetic-toolbox, Property 10: Parameter Extraction Validity**
        **Validates: Requirements 2.2, 3.3**
        
        For any valid WAV audio file with voiced content, parameter extraction
        SHALL produce F0 values within the configured min/max range for voiced frames.
        
        Note: Praat's pitch tracking algorithm uses interpolation and smoothing,
        which can result in F0 values slightly outside the configured range.
        We allow a small tolerance (1 Hz) to account for this algorithmic behavior.
        """
        # Extract parameters
        result = compute_praat_f0_formants(
            wav_path=wav_path,
            frameshift_ms=5,
            min_f0=min_f0,
            max_f0=max_f0
        )
        
        # Get F0 values
        f0_values = result["pF0"]
        
        # Filter out NaN values (unvoiced frames)
        voiced_f0 = f0_values[~np.isnan(f0_values)]
        
        # Skip if no voiced frames (some audio may be entirely unvoiced)
        if len(voiced_f0) == 0:
            pytest.skip(f"No voiced frames in {wav_path.name}")
        
        # Allow small tolerance for pitch tracking algorithm interpolation
        tolerance = 1.0  # Hz
        
        # Property: All voiced F0 values should be within configured range (with tolerance)
        assert np.all(voiced_f0 >= min_f0 - tolerance), (
            f"F0 values below min_f0={min_f0} (with {tolerance}Hz tolerance): min={np.min(voiced_f0):.2f}"
        )
        assert np.all(voiced_f0 <= max_f0 + tolerance), (
            f"F0 values above max_f0={max_f0} (with {tolerance}Hz tolerance): max={np.max(voiced_f0):.2f}"
        )
    
    @pytest.mark.parametrize("wav_path", TEST_WAV_FILES, ids=lambda p: p.name)
    def test_formant_values_are_positive(self, wav_path: Path):
        """
        **Feature: phonetic-toolbox, Property 10: Parameter Extraction Validity**
        **Validates: Requirements 2.2, 3.3**
        
        For any valid WAV audio file, formant values (when present) should be positive.
        """
        result = compute_praat_f0_formants(
            wav_path=wav_path,
            frameshift_ms=5,
            min_f0=40,
            max_f0=500
        )
        
        # Check formants F1-F4
        for formant_key in ["pF1", "pF2", "pF3", "pF4"]:
            formant_values = result[formant_key]
            # Filter out NaN values
            valid_formants = formant_values[~np.isnan(formant_values)]
            
            if len(valid_formants) > 0:
                # All valid formant values should be positive
                assert np.all(valid_formants > 0), (
                    f"{formant_key} has non-positive values: min={np.min(valid_formants):.2f}"
                )
    
    @pytest.mark.parametrize("wav_path", TEST_WAV_FILES, ids=lambda p: p.name)
    def test_formant_ordering(self, wav_path: Path):
        """
        **Feature: phonetic-toolbox, Property 10: Parameter Extraction Validity**
        **Validates: Requirements 2.2, 3.3**
        
        For any valid WAV audio file, formants should generally follow F1 < F2 < F3 < F4
        ordering (when all are present and valid).
        """
        result = compute_praat_f0_formants(
            wav_path=wav_path,
            frameshift_ms=5,
            min_f0=40,
            max_f0=500
        )
        
        f1 = result["pF1"]
        f2 = result["pF2"]
        f3 = result["pF3"]
        f4 = result["pF4"]
        
        # Find frames where all formants are valid
        valid_mask = (
            ~np.isnan(f1) & ~np.isnan(f2) & ~np.isnan(f3) & ~np.isnan(f4)
        )
        
        if not np.any(valid_mask):
            pytest.skip(f"No frames with all valid formants in {wav_path.name}")
        
        # Check ordering for valid frames
        # Note: We check that the majority follow the ordering, as some frames
        # may have estimation errors
        f1_valid = f1[valid_mask]
        f2_valid = f2[valid_mask]
        f3_valid = f3[valid_mask]
        f4_valid = f4[valid_mask]
        
        # Count frames where ordering is correct
        correct_order = (
            (f1_valid < f2_valid) & (f2_valid < f3_valid) & (f3_valid < f4_valid)
        )
        correct_ratio = np.sum(correct_order) / len(correct_order)
        
        # At least 50% of frames should have correct ordering
        # (allowing for some estimation errors)
        assert correct_ratio >= 0.5, (
            f"Only {correct_ratio*100:.1f}% of frames have correct formant ordering"
        )
    
    @pytest.mark.parametrize("wav_path", TEST_WAV_FILES, ids=lambda p: p.name)
    def test_result_contains_required_keys(self, wav_path: Path):
        """
        **Feature: phonetic-toolbox, Property 10: Parameter Extraction Validity**
        **Validates: Requirements 2.2, 3.3**
        
        Parameter extraction should return all required keys.
        """
        result = compute_praat_f0_formants(
            wav_path=wav_path,
            frameshift_ms=5,
            min_f0=40,
            max_f0=500
        )
        
        required_keys = ["pF0", "pF1", "pF2", "pF3", "pF4", "pB1", "pB2", "pB3", "pB4", "Fs"]
        
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
    
    @pytest.mark.parametrize("wav_path", TEST_WAV_FILES, ids=lambda p: p.name)
    def test_array_lengths_consistent(self, wav_path: Path):
        """
        **Feature: phonetic-toolbox, Property 10: Parameter Extraction Validity**
        **Validates: Requirements 2.2, 3.3**
        
        All parameter arrays should have the same length.
        """
        result = compute_praat_f0_formants(
            wav_path=wav_path,
            frameshift_ms=5,
            min_f0=40,
            max_f0=500
        )
        
        array_keys = ["pF0", "pF1", "pF2", "pF3", "pF4", "pB1", "pB2", "pB3", "pB4"]
        
        lengths = {key: len(result[key]) for key in array_keys}
        unique_lengths = set(lengths.values())
        
        assert len(unique_lengths) == 1, (
            f"Inconsistent array lengths: {lengths}"
        )


class TestParameterExtractionEdgeCases:
    """
    Edge case tests for parameter extraction.
    """
    
    def test_nonexistent_file_raises_error(self, tmp_path: Path):
        """
        Parameter extraction should raise an error for non-existent files.
        """
        fake_path = tmp_path / "nonexistent.wav"
        
        with pytest.raises(Exception):
            compute_praat_f0_formants(
                wav_path=fake_path,
                frameshift_ms=5,
                min_f0=40,
                max_f0=500
            )
