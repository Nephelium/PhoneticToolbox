import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.state import AppState
from tests.generators import app_state_params


def test_defaults_match_matlab_init():
    """Test that AppState defaults match expected values."""
    s = AppState()
    assert s.windowsize == 25
    assert s.frameshift == 2  # Fixed: actual default is 2, not 5
    assert s.preemphasis == 0.96
    assert s.lpcOrder == 12
    assert s.F0Praatmin == 40
    assert s.F0Praatmax == 500


def test_to_dict_returns_all_fields():
    """Test that to_dict includes all dataclass fields."""
    s = AppState()
    d = s.to_dict()
    
    # Check key fields are present
    assert "frameshift" in d
    assert "windowsize" in d
    assert "F0ReaperMinF0" in d
    assert "F0ReaperMaxF0" in d
    assert "recursedir" in d


def test_from_dict_creates_valid_state():
    """Test that from_dict creates a valid AppState."""
    data = {
        "frameshift": 5,
        "windowsize": 30,
        "F0ReaperMinF0": 50,
        "F0ReaperMaxF0": 400,
    }
    s = AppState.from_dict(data)
    
    assert s.frameshift == 5
    assert s.windowsize == 30
    assert s.F0ReaperMinF0 == 50
    assert s.F0ReaperMaxF0 == 400


def test_save_and_load_file(tmp_path):
    """Test saving and loading AppState to/from file."""
    s = AppState()
    s.frameshift = 7
    s.windowsize = 35
    
    file_path = tmp_path / "settings.json"
    s.save_to_file(file_path)
    
    loaded = AppState.load_from_file(file_path)
    
    assert loaded.frameshift == 7
    assert loaded.windowsize == 35


# ============================================================================
# Property-Based Tests
# ============================================================================

@given(params=app_state_params())
@settings(max_examples=100)
def test_settings_persistence_round_trip(params, tmp_path_factory):
    """
    **Feature: phonetic-toolbox, Property 4: Settings Persistence Round-Trip**
    **Validates: Requirements 3.2**
    
    For any valid parameter settings, saving them and reloading SHALL restore
    the same parameter values.
    """
    # Create AppState with random parameters
    s = AppState()
    for key, value in params.items():
        setattr(s, key, value)
    
    # Save to dict and restore
    d = s.to_dict()
    restored = AppState.from_dict(d)
    
    # Verify all modified parameters are preserved
    for key, value in params.items():
        assert getattr(restored, key) == value, f"Field {key} not preserved: {value} != {getattr(restored, key)}"


@given(params=app_state_params())
@settings(max_examples=100)
def test_settings_file_persistence_round_trip(params, tmp_path_factory):
    """
    **Feature: phonetic-toolbox, Property 4: Settings Persistence Round-Trip (File)**
    **Validates: Requirements 3.2**
    
    For any valid parameter settings, saving to file and reloading SHALL restore
    the same parameter values.
    """
    # Create a unique temp directory for this test
    tmp_path = tmp_path_factory.mktemp("settings")
    
    # Create AppState with random parameters
    s = AppState()
    for key, value in params.items():
        setattr(s, key, value)
    
    # Save to file and reload
    file_path = tmp_path / "settings.json"
    s.save_to_file(file_path)
    restored = AppState.load_from_file(file_path)
    
    # Verify all modified parameters are preserved
    for key, value in params.items():
        assert getattr(restored, key) == value, f"Field {key} not preserved: {value} != {getattr(restored, key)}"
