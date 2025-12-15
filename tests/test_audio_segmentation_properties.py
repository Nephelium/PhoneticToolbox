"""
Property-based tests for audio segmentation functionality.

**Feature: phonetic-toolbox, Property 9: Audio Segmentation Consistency**
**Validates: Requirements 13.4**
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from scipy.io import wavfile

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.textgrid_parser import TextGrid, Tier, Interval


# ============================================================================
# Audio Segmentation Core Logic (extracted for testability)
# ============================================================================

def segment_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    intervals: List[Interval],
    skip_labels: Tuple[str, ...] = ("sil", "eps", "<sil>", "<eps>", "")
) -> List[Tuple[np.ndarray, Interval]]:
    """
    Segment audio data based on TextGrid intervals.
    
    This function extracts the core segmentation logic from 
    ParameterEstimationController._save_segmented_audio() for testability.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        intervals: List of TextGrid Interval objects
        skip_labels: Labels to skip (silence markers)
    
    Returns:
        List of (audio_segment, interval) tuples
    """
    segments = []
    
    for interval in intervals:
        txt = interval.text.strip()
        if not txt or txt.lower() in [s.lower() for s in skip_labels]:
            continue
        
        start_sample = int(interval.xmin * sample_rate)
        end_sample = int(interval.xmax * sample_rate)
        
        # Boundary checks
        if start_sample >= len(audio_data) or end_sample <= start_sample:
            continue
        
        # Clamp end_sample to audio length
        end_sample = min(end_sample, len(audio_data))
        
        sliced_audio = audio_data[start_sample:end_sample]
        segments.append((sliced_audio, interval))
    
    return segments


def calculate_segment_duration_samples(segment: np.ndarray) -> int:
    """Calculate the duration of a segment in samples."""
    return len(segment)


def calculate_interval_duration_samples(interval: Interval, sample_rate: int) -> int:
    """Calculate the expected duration of an interval in samples."""
    return int((interval.xmax - interval.xmin) * sample_rate)


# ============================================================================
# Hypothesis Strategies for Audio Segmentation
# ============================================================================

@st.composite
def audio_with_textgrid(draw, min_duration_sec: float = 0.5, max_duration_sec: float = 5.0):
    """
    Generate audio data with matching TextGrid intervals.
    
    Returns a tuple of (audio_data, sample_rate, intervals, duration_sec)
    """
    # Generate audio parameters
    sample_rate = draw(st.sampled_from([8000, 16000, 22050, 44100]))
    duration_sec = draw(st.floats(
        min_value=min_duration_sec, 
        max_value=max_duration_sec,
        allow_nan=False,
        allow_infinity=False
    ))
    
    n_samples = int(duration_sec * sample_rate)
    assume(n_samples > 0)
    
    # Generate audio data (simple sine wave or random noise)
    audio_data = draw(st.sampled_from([
        np.sin(2 * np.pi * 440 * np.arange(n_samples) / sample_rate).astype(np.float32),
        np.random.randn(n_samples).astype(np.float32) * 0.5
    ]))
    
    # Generate non-overlapping intervals that cover part of the audio
    n_intervals = draw(st.integers(min_value=1, max_value=5))
    
    # Generate interval boundaries
    intervals = []
    current_time = 0.0
    segment_duration = duration_sec / (n_intervals + 1)  # Leave some gaps
    
    for i in range(n_intervals):
        xmin = current_time
        xmax = min(current_time + segment_duration * 0.8, duration_sec)  # 80% of segment
        
        if xmax <= xmin:
            break
            
        # Generate non-silence label
        label = draw(st.sampled_from(["a", "e", "i", "o", "u", "word1", "word2", "phone"]))
        
        intervals.append(Interval(xmin=xmin, xmax=xmax, text=label))
        current_time = xmax + segment_duration * 0.2  # 20% gap
    
    assume(len(intervals) > 0)
    
    return audio_data, sample_rate, intervals, duration_sec


# ============================================================================
# Property Tests
# ============================================================================

@given(data=audio_with_textgrid())
@settings(max_examples=100, deadline=None)
def test_audio_segmentation_consistency(data):
    """
    **Feature: phonetic-toolbox, Property 9: Audio Segmentation Consistency**
    **Validates: Requirements 13.4**
    
    For any audio file and corresponding TextGrid with non-empty intervals,
    segmenting the audio SHALL produce segments whose total duration equals
    the sum of interval durations (within sample precision).
    """
    audio_data, sample_rate, intervals, duration_sec = data
    
    # Perform segmentation
    segments = segment_audio(audio_data, sample_rate, intervals)
    
    # Calculate total duration of segments in samples
    total_segment_samples = sum(
        calculate_segment_duration_samples(seg) 
        for seg, _ in segments
    )
    
    # Calculate expected total duration from intervals in samples
    expected_total_samples = sum(
        calculate_interval_duration_samples(interval, sample_rate)
        for _, interval in segments
    )
    
    # The actual segment samples should match expected within 1 sample tolerance
    # (due to integer rounding in sample calculation)
    assert abs(total_segment_samples - expected_total_samples) <= len(segments), (
        f"Segment duration mismatch.\n"
        f"Total segment samples: {total_segment_samples}\n"
        f"Expected from intervals: {expected_total_samples}\n"
        f"Difference: {abs(total_segment_samples - expected_total_samples)}\n"
        f"Number of segments: {len(segments)}"
    )


@given(data=audio_with_textgrid())
@settings(max_examples=100, deadline=None)
def test_segment_boundaries_within_audio(data):
    """
    **Feature: phonetic-toolbox, Property 9: Audio Segmentation Consistency**
    **Validates: Requirements 13.4**
    
    For any segmentation, all segment boundaries SHALL be within the 
    original audio bounds.
    """
    audio_data, sample_rate, intervals, duration_sec = data
    
    # Perform segmentation
    segments = segment_audio(audio_data, sample_rate, intervals)
    
    for segment, interval in segments:
        # Each segment should have valid length
        assert len(segment) > 0, f"Segment for interval {interval} should not be empty"
        
        # Segment length should not exceed audio length
        assert len(segment) <= len(audio_data), (
            f"Segment length {len(segment)} exceeds audio length {len(audio_data)}"
        )


@given(data=audio_with_textgrid())
@settings(max_examples=100, deadline=None)
def test_segment_count_matches_valid_intervals(data):
    """
    **Feature: phonetic-toolbox, Property 9: Audio Segmentation Consistency**
    **Validates: Requirements 13.4**
    
    The number of segments produced SHALL equal the number of valid 
    (non-silence) intervals that fall within audio bounds.
    """
    audio_data, sample_rate, intervals, duration_sec = data
    
    # Count valid intervals (non-silence, within bounds)
    skip_labels = ("sil", "eps", "<sil>", "<eps>", "")
    valid_intervals = [
        interval for interval in intervals
        if interval.text.strip() 
        and interval.text.strip().lower() not in [s.lower() for s in skip_labels]
        and int(interval.xmin * sample_rate) < len(audio_data)
        and int(interval.xmax * sample_rate) > int(interval.xmin * sample_rate)
    ]
    
    # Perform segmentation
    segments = segment_audio(audio_data, sample_rate, intervals)
    
    assert len(segments) == len(valid_intervals), (
        f"Segment count {len(segments)} should match valid interval count {len(valid_intervals)}"
    )


@given(data=audio_with_textgrid())
@settings(max_examples=100, deadline=None)
def test_silence_intervals_excluded(data):
    """
    **Feature: phonetic-toolbox, Property 9: Audio Segmentation Consistency**
    **Validates: Requirements 13.4**
    
    Intervals with silence labels SHALL be excluded from segmentation.
    """
    audio_data, sample_rate, intervals, duration_sec = data
    
    # Add some silence intervals
    silence_labels = ["sil", "SIL", "<sil>", "eps", "<eps>", "", "  "]
    silence_intervals = [
        Interval(xmin=0.0, xmax=0.1, text=label)
        for label in silence_labels
    ]
    
    # Combine with original intervals
    all_intervals = silence_intervals + list(intervals)
    
    # Perform segmentation
    segments = segment_audio(audio_data, sample_rate, all_intervals)
    
    # Verify no silence labels in segments
    for segment, interval in segments:
        assert interval.text.strip().lower() not in ["sil", "eps", "<sil>", "<eps>"], (
            f"Silence interval {interval.text} should be excluded"
        )
        assert interval.text.strip() != "", "Empty label should be excluded"
