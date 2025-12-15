"""
Hypothesis generators (strategies) for PhoneticToolbox property-based testing.

This module provides custom strategies for generating test data that matches
the domain constraints of phonetic analysis.
"""
from __future__ import annotations

import string
from typing import Dict, Any, List

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


# ============================================================================
# Basic Value Strategies
# ============================================================================

# F0 (Fundamental Frequency) range: typically 40-500 Hz for human speech
f0_values = st.floats(min_value=40.0, max_value=500.0, allow_nan=False, allow_infinity=False)

# Formant frequency ranges (Hz)
f1_values = st.floats(min_value=200.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
f2_values = st.floats(min_value=700.0, max_value=2800.0, allow_nan=False, allow_infinity=False)
f3_values = st.floats(min_value=1800.0, max_value=3800.0, allow_nan=False, allow_infinity=False)
f4_values = st.floats(min_value=2800.0, max_value=5000.0, allow_nan=False, allow_infinity=False)

# Time values (seconds)
time_values = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)

# Frame shift (ms): typically 1-10 ms
frameshift_values = st.integers(min_value=1, max_value=10)

# Window size (ms): typically 10-50 ms
windowsize_values = st.integers(min_value=10, max_value=50)

# Energy values (normalized 0-1)
energy_values = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ============================================================================
# String Strategies
# ============================================================================

# ASCII filenames (safe for filesystem)
safe_filename_chars = string.ascii_letters + string.digits + "_-"
safe_filenames = st.text(
    alphabet=safe_filename_chars,
    min_size=1,
    max_size=50
).filter(lambda x: len(x.strip()) > 0)

# Chinese characters (Unicode range \u4e00-\u9fff)
chinese_chars = st.text(
    alphabet=st.characters(min_codepoint=0x4e00, max_codepoint=0x9fff),
    min_size=1,
    max_size=20
)

# Mixed Chinese and ASCII text
mixed_text = st.text(
    alphabet=st.characters(
        whitelist_categories=('L', 'N'),  # Letters and Numbers
        min_codepoint=0x0020,
        max_codepoint=0x9fff
    ),
    min_size=1,
    max_size=50
).filter(lambda x: len(x.strip()) > 0)

# TextGrid interval text (phoneme labels)
phoneme_labels = st.sampled_from([
    "a", "e", "i", "o", "u", "ɑ", "ə", "ɛ", "ɪ", "ʊ",
    "p", "t", "k", "b", "d", "g", "m", "n", "ŋ",
    "f", "s", "ʃ", "h", "v", "z", "ʒ",
    "l", "r", "w", "j",
    "sil", "", " "
])


# ============================================================================
# Array Strategies
# ============================================================================

def acoustic_param_array(n_frames: int = 50) -> st.SearchStrategy[np.ndarray]:
    """Generate an array of acoustic parameter values."""
    return arrays(
        dtype=np.float64,
        shape=n_frames,
        elements=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    )


def f0_array(n_frames: int = 50) -> st.SearchStrategy[np.ndarray]:
    """Generate an array of F0 values with some NaN (unvoiced) frames."""
    return arrays(
        dtype=np.float64,
        shape=n_frames,
        elements=st.one_of(
            st.floats(min_value=40.0, max_value=500.0, allow_nan=False, allow_infinity=False),
            st.just(np.nan)
        )
    )


# ============================================================================
# TextGrid Strategies
# ============================================================================

# Import TextGrid data classes for generating proper objects
from utils.textgrid_parser import TextGrid, Tier, Interval


@st.composite
def textgrid_interval(draw, max_duration: float = 10.0):
    """Generate a single TextGrid interval."""
    xmin = draw(st.floats(min_value=0.0, max_value=max_duration - 0.01, allow_nan=False, allow_infinity=False))
    duration = draw(st.floats(min_value=0.01, max_value=min(1.0, max_duration - xmin), allow_nan=False, allow_infinity=False))
    xmax = xmin + duration
    text = draw(phoneme_labels)
    return {"xmin": xmin, "xmax": xmax, "text": text}


@st.composite
def textgrid_tier(draw, name: str = None, duration: float = 10.0, min_intervals: int = 1, max_intervals: int = 10):
    """Generate a TextGrid tier with non-overlapping intervals."""
    if name is None:
        name = draw(st.sampled_from(["words", "phones", "segments", "tones"]))
    
    n_intervals = draw(st.integers(min_value=min_intervals, max_value=max_intervals))
    
    # Generate non-overlapping intervals
    intervals = []
    current_time = 0.0
    segment_duration = duration / n_intervals
    
    for i in range(n_intervals):
        xmin = current_time
        # Add some variation but ensure non-overlapping
        xmax = min(current_time + segment_duration, duration)
        text = draw(phoneme_labels)
        intervals.append({"xmin": xmin, "xmax": xmax, "text": text})
        current_time = xmax
    
    return {
        "name": name,
        "xmin": 0.0,
        "xmax": duration,
        "intervals": intervals
    }


@st.composite
def textgrid_data(draw, duration: float = 10.0, min_tiers: int = 1, max_tiers: int = 3):
    """Generate a complete TextGrid data structure."""
    n_tiers = draw(st.integers(min_value=min_tiers, max_value=max_tiers))
    tier_names = ["words", "phones", "segments", "tones", "misc"][:n_tiers]
    
    tiers = []
    for name in tier_names:
        tier = draw(textgrid_tier(name=name, duration=duration))
        tiers.append(tier)
    
    return {
        "xmin": 0.0,
        "xmax": duration,
        "tiers": tiers
    }


@st.composite
def textgrid_interval_obj(draw, xmin: float = 0.0, max_duration: float = 10.0):
    """Generate a TextGrid Interval object."""
    duration = draw(st.floats(min_value=0.01, max_value=min(1.0, max_duration - xmin), allow_nan=False, allow_infinity=False))
    xmax = xmin + duration
    text = draw(phoneme_labels)
    return Interval(xmin=xmin, xmax=xmax, text=text)


@st.composite
def textgrid_tier_obj(draw, name: str = None, duration: float = 10.0, min_intervals: int = 1, max_intervals: int = 10):
    """Generate a TextGrid Tier object with non-overlapping intervals."""
    if name is None:
        name = draw(st.sampled_from(["words", "phones", "segments", "tones"]))
    
    n_intervals = draw(st.integers(min_value=min_intervals, max_value=max_intervals))
    
    # Generate non-overlapping intervals that cover the entire duration
    intervals = []
    current_time = 0.0
    segment_duration = duration / n_intervals
    
    for i in range(n_intervals):
        xmin = current_time
        xmax = min(current_time + segment_duration, duration)
        text = draw(phoneme_labels)
        intervals.append(Interval(xmin=xmin, xmax=xmax, text=text))
        current_time = xmax
    
    return Tier(name=name, xmin=0.0, xmax=duration, intervals=intervals)


@st.composite
def textgrid_obj(draw, duration: float = None, min_tiers: int = 1, max_tiers: int = 3):
    """Generate a complete TextGrid object."""
    if duration is None:
        duration = draw(st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False))
    
    n_tiers = draw(st.integers(min_value=min_tiers, max_value=max_tiers))
    tier_names = ["words", "phones", "segments", "tones", "misc"][:n_tiers]
    
    tiers = []
    for name in tier_names:
        tier = draw(textgrid_tier_obj(name=name, duration=duration))
        tiers.append(tier)
    
    return TextGrid(xmin=0.0, xmax=duration, tiers=tiers)


# ============================================================================
# CSV/Parameter Dictionary Strategies
# ============================================================================

@st.composite
def acoustic_params_dict(draw, n_frames: int = None):
    """Generate a dictionary of acoustic parameters like those saved to CSV."""
    if n_frames is None:
        n_frames = draw(st.integers(min_value=10, max_value=100))
    
    frameshift = draw(frameshift_values)
    
    # Generate parameter arrays
    params = {
        "frameshift": float(frameshift),
        "pF0": draw(arrays(np.float64, n_frames, elements=f0_values)),
        "pF1": draw(arrays(np.float64, n_frames, elements=f1_values)),
        "pF2": draw(arrays(np.float64, n_frames, elements=f2_values)),
        "pF3": draw(arrays(np.float64, n_frames, elements=f3_values)),
        "pF4": draw(arrays(np.float64, n_frames, elements=f4_values)),
        "Energy": draw(arrays(np.float64, n_frames, elements=energy_values)),
    }
    
    return params


# ============================================================================
# AppState Strategies
# ============================================================================

@st.composite
def app_state_params(draw):
    """Generate valid AppState parameter values."""
    return {
        "frameshift": draw(frameshift_values),
        "windowsize": draw(windowsize_values),
        "F0ReaperMinF0": draw(st.integers(min_value=40, max_value=100)),  # 最低 40Hz，与 Praat 一致
        "F0ReaperMaxF0": draw(st.integers(min_value=300, max_value=600)),
        "recursedir": draw(st.integers(min_value=0, max_value=1)),
        "linkmatdir": draw(st.integers(min_value=0, max_value=1)),
        "linkwavdir": draw(st.integers(min_value=0, max_value=1)),
    }


# ============================================================================
# File Path Strategies
# ============================================================================

@st.composite
def valid_wav_filename(draw):
    """Generate a valid WAV filename."""
    name = draw(safe_filenames)
    return f"{name}.wav"


@st.composite
def chinese_filename(draw):
    """Generate a filename containing Chinese characters."""
    chinese_part = draw(chinese_chars)
    return f"{chinese_part}.wav"
