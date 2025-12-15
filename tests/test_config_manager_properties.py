"""
Property-based tests for ConfigManager in Whisper Transcription Tool.

Tests configuration persistence and round-trip properties.
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_transcription.config_manager import ConfigManager


# ============================================================================
# Strategies for generating test data
# ============================================================================

# Strategy for valid file path strings (non-empty, no null bytes)
valid_path_strings = st.text(
    alphabet=st.characters(
        blacklist_categories=('Cs',),  # Exclude surrogates
        blacklist_characters=('\x00',)  # Exclude null bytes
    ),
    min_size=0,
    max_size=200
)


# ============================================================================
# Property-Based Tests
# ============================================================================

@given(model_path=valid_path_strings)
@settings(max_examples=100)
def test_config_round_trip_model_path(model_path, tmp_path_factory):
    """
    **Feature: whisper-transcription, Property 1: Configuration Round Trip**
    **Validates: Requirements 1.2, 1.3**
    
    For any valid model path string, saving it to the configuration and then
    loading the configuration should return the same model path string.
    """
    # Create a unique temp directory for this test
    tmp_path = tmp_path_factory.mktemp("config")
    
    # Create ConfigManager with temp directory
    config = ConfigManager(config_dir=str(tmp_path))
    
    # Set the model path
    config.set_model_path(model_path)
    
    # Save configuration
    save_result = config.save_config()
    assert save_result, "Config save should succeed"
    
    # Create a new ConfigManager to load the saved config
    loaded_config = ConfigManager(config_dir=str(tmp_path))
    
    # Verify the model path is preserved
    assert loaded_config.get_model_path() == model_path, \
        f"Model path not preserved: expected '{model_path}', got '{loaded_config.get_model_path()}'"


@given(
    model_path=valid_path_strings,
    last_directory=valid_path_strings
)
@settings(max_examples=100)
def test_config_round_trip_all_fields(model_path, last_directory, tmp_path_factory):
    """
    **Feature: whisper-transcription, Property 1: Configuration Round Trip (All Fields)**
    **Validates: Requirements 1.2, 1.3**
    
    For any valid configuration values, saving and loading should preserve
    all configuration fields.
    """
    # Create a unique temp directory for this test
    tmp_path = tmp_path_factory.mktemp("config")
    
    # Create ConfigManager with temp directory
    config = ConfigManager(config_dir=str(tmp_path))
    
    # Set all configuration values
    config.set_model_path(model_path)
    config.set_last_directory(last_directory)
    
    # Save configuration
    save_result = config.save_config()
    assert save_result, "Config save should succeed"
    
    # Create a new ConfigManager to load the saved config
    loaded_config = ConfigManager(config_dir=str(tmp_path))
    
    # Verify all fields are preserved
    assert loaded_config.get_model_path() == model_path, \
        f"Model path not preserved: expected '{model_path}', got '{loaded_config.get_model_path()}'"
    assert loaded_config.get_last_directory() == last_directory, \
        f"Last directory not preserved: expected '{last_directory}', got '{loaded_config.get_last_directory()}'"


# ============================================================================
# Unit Tests
# ============================================================================

def test_default_config_values(tmp_path):
    """Test that default configuration values are correct."""
    config = ConfigManager(config_dir=str(tmp_path))
    
    assert config.get_model_path() == ""
    assert config.get_last_directory() == ""


def test_missing_config_file_creates_defaults(tmp_path):
    """Test that missing config file results in default values."""
    # Ensure no config file exists
    config_path = tmp_path / ConfigManager.CONFIG_FILE
    assert not config_path.exists()
    
    # Create ConfigManager
    config = ConfigManager(config_dir=str(tmp_path))
    
    # Should have default values
    assert config.get_model_path() == ""
    assert config.get_last_directory() == ""


def test_save_creates_config_file(tmp_path):
    """Test that save_config creates the config file."""
    config = ConfigManager(config_dir=str(tmp_path))
    config.set_model_path("/path/to/model")
    
    # Save config
    result = config.save_config()
    
    assert result is True
    assert (tmp_path / ConfigManager.CONFIG_FILE).exists()


def test_corrupted_config_file_returns_defaults(tmp_path):
    """Test that corrupted config file results in default values."""
    # Create a corrupted config file
    config_path = tmp_path / ConfigManager.CONFIG_FILE
    config_path.write_text("{ invalid json }", encoding='utf-8')
    
    # Create ConfigManager
    config = ConfigManager(config_dir=str(tmp_path))
    
    # Should have default values
    assert config.get_model_path() == ""
    assert config.get_last_directory() == ""
