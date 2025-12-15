"""
Property-based tests for FileService in Whisper Transcription Tool.

Tests audio file discovery and lab file naming properties.
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

from whisper_transcription.file_service import FileService


# ============================================================================
# Strategies for generating test data
# ============================================================================

# Safe filename characters (ASCII letters, digits, underscore, hyphen)
safe_filename_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"

# Strategy for valid base filenames (no extension)
valid_base_filenames = st.text(
    alphabet=safe_filename_chars,
    min_size=1,
    max_size=50
)

# Strategy for audio extensions
audio_extensions = st.sampled_from(['.wav', '.mp3', '.flac', '.ogg', '.m4a'])

# Strategy for non-audio extensions
non_audio_extensions = st.sampled_from(['.txt', '.pdf', '.doc', '.jpg', '.png', '.exe', '.py'])

# Strategy for optional suffix
optional_suffix = st.one_of(
    st.just(""),
    st.text(alphabet=safe_filename_chars, min_size=1, max_size=10).map(lambda s: f"_{s}")
)


# ============================================================================
# Property-Based Tests
# ============================================================================

@given(
    filenames=st.lists(valid_base_filenames, min_size=0, max_size=10, unique=True),
    extensions=st.lists(audio_extensions, min_size=0, max_size=10)
)
@settings(max_examples=100)
def test_audio_file_discovery_completeness(filenames, extensions, tmp_path_factory):
    """
    **Feature: whisper-transcription, Property 3: Audio File Discovery Completeness**
    **Validates: Requirements 2.3**
    
    For any directory containing audio files with supported extensions,
    the file discovery function should return all and only those audio files.
    """
    # Create a unique temp directory for this test
    tmp_path = tmp_path_factory.mktemp("audio")
    
    # Create audio files in temp directory
    created_audio_files = set()
    
    # Pair filenames with extensions (use min length to avoid index errors)
    num_files = min(len(filenames), len(extensions))
    
    for i in range(num_files):
        filename = filenames[i]
        ext = extensions[i]
        full_name = f"{filename}{ext}"
        file_path = tmp_path / full_name
        file_path.write_text("dummy audio content")
        created_audio_files.add(str(file_path.resolve()))
    
    # Discover audio files
    discovered = FileService.discover_audio_files(str(tmp_path))
    discovered_set = set(discovered)
    
    # Property: All created audio files should be discovered
    assert created_audio_files == discovered_set, \
        f"Mismatch: created={created_audio_files}, discovered={discovered_set}"


@given(
    audio_filenames=st.lists(valid_base_filenames, min_size=1, max_size=5, unique=True),
    non_audio_filenames=st.lists(valid_base_filenames, min_size=1, max_size=5, unique=True),
    audio_ext=audio_extensions,
    non_audio_ext=non_audio_extensions
)
@settings(max_examples=100)
def test_audio_file_discovery_excludes_non_audio(
    audio_filenames, non_audio_filenames, audio_ext, non_audio_ext, tmp_path_factory
):
    """
    **Feature: whisper-transcription, Property 3: Audio File Discovery Completeness**
    **Validates: Requirements 2.3**
    
    For any directory containing both audio and non-audio files,
    the file discovery function should return only audio files.
    """
    # Create a unique temp directory for this test
    tmp_path = tmp_path_factory.mktemp("audio")
    
    # Ensure no overlap between audio and non-audio filenames
    non_audio_filenames = [f for f in non_audio_filenames if f not in audio_filenames]
    assume(len(non_audio_filenames) > 0)
    
    # Create audio files
    audio_files = set()
    for filename in audio_filenames:
        file_path = tmp_path / f"{filename}{audio_ext}"
        file_path.write_text("dummy audio content")
        audio_files.add(str(file_path.resolve()))
    
    # Create non-audio files
    for filename in non_audio_filenames:
        file_path = tmp_path / f"{filename}{non_audio_ext}"
        file_path.write_text("dummy non-audio content")
    
    # Discover audio files
    discovered = FileService.discover_audio_files(str(tmp_path))
    discovered_set = set(discovered)
    
    # Property: Only audio files should be discovered
    assert discovered_set == audio_files, \
        f"Expected only audio files: {audio_files}, got: {discovered_set}"


@given(
    base_filename=valid_base_filenames,
    audio_ext=audio_extensions,
    suffix=optional_suffix
)
@settings(max_examples=100)
def test_lab_file_naming_convention(base_filename, audio_ext, suffix, tmp_path_factory):
    """
    **Feature: whisper-transcription, Property 7: Lab File Naming Convention**
    **Validates: Requirements 5.1, 5.4**
    
    For any audio file path, the generated .lab file should have the same
    base name and be located in the same directory.
    """
    # Create a unique temp directory for this test
    tmp_path = tmp_path_factory.mktemp("lab")
    
    # Create audio file path
    audio_path = tmp_path / f"{base_filename}{audio_ext}"
    
    # Get lab path
    lab_path = FileService.get_lab_path(str(audio_path), suffix)
    lab_path_obj = Path(lab_path)
    
    # Property 1: Lab file should be in the same directory
    assert lab_path_obj.parent == audio_path.parent, \
        f"Lab file not in same directory: audio={audio_path.parent}, lab={lab_path_obj.parent}"
    
    # Property 2: Lab file should have .lab extension
    assert lab_path_obj.suffix == ".lab", \
        f"Lab file should have .lab extension, got: {lab_path_obj.suffix}"
    
    # Property 3: Lab file base name should start with audio file base name
    expected_stem = base_filename + suffix if suffix else base_filename
    assert lab_path_obj.stem == expected_stem, \
        f"Lab file stem mismatch: expected={expected_stem}, got={lab_path_obj.stem}"


# Strategy for text content that avoids line ending normalization issues
# Python's text mode normalizes \r to \n on Windows, so we exclude bare \r
text_content = st.text(
    alphabet=st.characters(
        blacklist_characters=('\r',)  # Exclude carriage return to avoid line ending normalization
    ),
    min_size=0,
    max_size=1000
)


@given(
    base_filename=valid_base_filenames,
    audio_ext=audio_extensions,
    content=text_content,
    suffix=optional_suffix
)
@settings(max_examples=100)
def test_lab_file_save_and_read_roundtrip(base_filename, audio_ext, content, suffix, tmp_path_factory):
    """
    **Feature: whisper-transcription, Property 7: Lab File Naming Convention**
    **Validates: Requirements 5.1, 5.4**
    
    For any audio file and content, saving a lab file and reading it back
    should return the same content.
    
    Note: We exclude bare \\r characters because Python's text mode file handling
    normalizes line endings, which is expected behavior for text files.
    """
    # Create a unique temp directory for this test
    tmp_path = tmp_path_factory.mktemp("lab")
    
    # Create audio file path (doesn't need to exist for lab file saving)
    audio_path = tmp_path / f"{base_filename}{audio_ext}"
    
    # Save lab file
    result = FileService.save_lab_file(str(audio_path), content, suffix)
    assert result is True, "Lab file save should succeed"
    
    # Read lab file back
    lab_path = FileService.get_lab_path(str(audio_path), suffix)
    with open(lab_path, 'r', encoding='utf-8') as f:
        read_content = f.read()
    
    # Property: Content should be preserved
    assert read_content == content, \
        f"Content not preserved: expected={content!r}, got={read_content!r}"


# ============================================================================
# Unit Tests
# ============================================================================

def test_discover_audio_files_empty_directory(tmp_path):
    """Test that empty directory returns empty list."""
    result = FileService.discover_audio_files(str(tmp_path))
    assert result == []


def test_discover_audio_files_nonexistent_path():
    """Test that nonexistent path returns empty list."""
    result = FileService.discover_audio_files("/nonexistent/path/12345")
    assert result == []


def test_discover_audio_files_single_file(tmp_path):
    """Test discovery of a single audio file."""
    audio_file = tmp_path / "test.wav"
    audio_file.write_text("dummy")
    
    result = FileService.discover_audio_files(str(audio_file))
    assert len(result) == 1
    assert result[0] == str(audio_file.resolve())


def test_discover_audio_files_single_non_audio_file(tmp_path):
    """Test that single non-audio file returns empty list."""
    non_audio_file = tmp_path / "test.txt"
    non_audio_file.write_text("dummy")
    
    result = FileService.discover_audio_files(str(non_audio_file))
    assert result == []


def test_audio_extensions_constant():
    """Test that AUDIO_EXTENSIONS contains expected formats."""
    expected = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    assert FileService.AUDIO_EXTENSIONS == expected


def test_get_lab_path_basic(tmp_path):
    """Test basic lab path generation."""
    audio_path = tmp_path / "test.wav"
    lab_path = FileService.get_lab_path(str(audio_path))
    expected = tmp_path / "test.lab"
    assert Path(lab_path) == expected


def test_get_lab_path_with_suffix(tmp_path):
    """Test lab path generation with suffix."""
    audio_path = tmp_path / "test.wav"
    lab_path = FileService.get_lab_path(str(audio_path), "_pinyin")
    expected = tmp_path / "test_pinyin.lab"
    assert Path(lab_path) == expected
