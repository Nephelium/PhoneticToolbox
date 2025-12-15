"""
Shared pytest fixtures and configuration for PhoneticToolbox tests.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np
from hypothesis import settings, Verbosity

# Ensure the project root is in sys.path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Hypothesis Configuration
# ============================================================================

# Register hypothesis profiles
settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose, deadline=None)

# Load profile from environment or use default
settings.load_profile("dev")


# ============================================================================
# Fixtures for file paths and directories
# ============================================================================

@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def test_audio_dir(project_root: Path) -> Path:
    """Return the klatt directory containing test audio files."""
    return project_root / "klatt"


@pytest.fixture
def sample_wav_path(test_audio_dir: Path) -> Path:
    """Return path to a sample WAV file for testing."""
    # Use one of the existing klatt audio files
    wav_files = list(test_audio_dir.glob("*.wav"))
    if wav_files:
        return wav_files[0]
    pytest.skip("No WAV files found in klatt directory")


# ============================================================================
# Fixtures for temporary directories
# ============================================================================

@pytest.fixture
def temp_wav_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample WAV files."""
    wav_dir = tmp_path / "wav_files"
    wav_dir.mkdir()
    return wav_dir


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return out_dir


# ============================================================================
# Fixtures for sample data generation
# ============================================================================

@pytest.fixture
def sample_acoustic_params() -> Dict[str, Any]:
    """Generate sample acoustic parameter data for testing."""
    n_frames = 50
    return {
        "frameshift": 2.0,
        "pF0": np.random.uniform(80, 300, n_frames).astype(float),
        "pF1": np.random.uniform(200, 800, n_frames).astype(float),
        "pF2": np.random.uniform(800, 2500, n_frames).astype(float),
        "pF3": np.random.uniform(2000, 3500, n_frames).astype(float),
        "pF4": np.random.uniform(3000, 4500, n_frames).astype(float),
        "Energy": np.random.uniform(0, 1, n_frames).astype(float),
    }


@pytest.fixture
def sample_textgrid_data() -> Dict[str, Any]:
    """Generate sample TextGrid-like data for testing."""
    return {
        "xmin": 0.0,
        "xmax": 1.0,
        "tiers": [
            {
                "name": "words",
                "intervals": [
                    {"xmin": 0.0, "xmax": 0.3, "text": "hello"},
                    {"xmin": 0.3, "xmax": 0.7, "text": "world"},
                    {"xmin": 0.7, "xmax": 1.0, "text": ""},
                ]
            }
        ]
    }


# ============================================================================
# Fixtures for AppState
# ============================================================================

@pytest.fixture
def app_state():
    """Create a fresh AppState instance for testing."""
    from models.state import AppState
    return AppState()
