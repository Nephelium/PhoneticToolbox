import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class LipTimeAxis:
    times: np.ndarray
    indices: np.ndarray
    sample_count: int
    manual_offset: float


def resolve_lip_time_axis(data: dict, pkl_path: Path) -> LipTimeAxis:
    """Resolve audio-relative times, excluding the separately stored offset.

    Prefer the first audio frame in metadata, then the timestamps companion.
    Preserve measured audio/lip delay even in legacy recordings. Only legacy
    relative-only recordings are rebased to zero when no audio anchor exists.
    Both the web editor and parameter export must use this same contract.
    """
    if not isinstance(data, dict):
        raise ValueError("Lip recording must be a dictionary")
    metadata = data.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    manual_offset = float(metadata.get("lip_manual_offset", 0.0) or 0.0)
    if not np.isfinite(manual_offset):
        raise ValueError("Lip offset must be finite")

    def finite_number(value):
        try:
            number = float(value)
            return number if np.isfinite(number) else None
        except (TypeError, ValueError):
            return None

    anchor = finite_number(metadata.get("audio_first_frame_time"))
    if anchor is None:
        path = Path(pkl_path)
        companion = path.with_name(f"{path.stem}_timestamps.pkl")
        try:
            with companion.open("rb") as handle:
                timestamps = pickle.load(handle)
            if isinstance(timestamps, dict):
                anchor = finite_number(timestamps.get("start_time"))
        except (OSError, EOFError, pickle.UnpicklingError):
            pass

    raw_absolute = data.get("absolute_timestamps")
    absolute = np.asarray(raw_absolute if raw_absolute is not None else [], dtype=float).reshape(-1)
    anchored = anchor is not None and absolute.size > 0
    if anchored:
        times = absolute - anchor
    else:
        raw_relative = data.get("relative_times")
        times = np.asarray(raw_relative if raw_relative is not None else [], dtype=float).reshape(-1)
    sample_count = len(times)
    indices = np.flatnonzero(np.isfinite(times))
    indices = indices[np.argsort(times[indices], kind="stable")]
    times = times[indices]
    unique = np.r_[True, np.diff(times) > 1e-9] if len(times) else np.array([], dtype=bool)
    times, indices = times[unique], indices[unique]
    if len(times) < 2:
        raise ValueError("Lip timestamps are not sufficient")
    if not anchored and metadata.get("time_alignment_mode") != "anchored_audio_start":
        times = times - times[0]
    return LipTimeAxis(times, indices, sample_count, manual_offset)

def read_lip_data(pkl_path: str, target_times: np.ndarray, smooth_win: int = 0) -> Dict[str, np.ndarray]:
    """
    Read lip feature data from a pickle file and interpolate to target timestamps.
    
    Args:
        pkl_path: Path to the .pkl file containing lip metrics.
        target_times: Array of timestamps (in seconds) to interpolate to.
        smooth_win: Window size for moving average smoothing (0 or 1 to disable).
        
    Returns:
        Dictionary containing interpolated and smoothed lip parameters:
        - LipArea: Lip Area Ratio (area)
        - LipWidth: Normalized Outer Lip Width (outer_width)
        - LipOpen: Normalized Lip Openness (open)
        - LipCirc: Lip Circularity (circularity)
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading lip data {pkl_path}: {e}")
        return {}

    try:
        axis = resolve_lip_time_axis(data, Path(pkl_path))
    except (TypeError, ValueError) as exc:
        print(f"Invalid lip time axis {pkl_path}: {exc}")
        return {}
    src_times = axis.times + axis.manual_offset

    # Mapping from internal key to output key
    key_map = {
        'area': 'LipArea',           # Lip Area Ratio
        'outer_width': 'LipWidth',   # Normalized Outer Lip Width
        'open': 'LipOpen',           # Normalized Lip Openness
        'circularity': 'LipCirc'     # Lip Circularity
    }
    
    result = {}
    
    for src_key, out_key in key_map.items():
        vals = data.get(src_key)
        if vals is None:
            continue
        vals = np.array(vals, dtype=float)
        if len(vals) != axis.sample_count:
            continue
        vals = vals[axis.indices]
        
        # Handle source NaNs: interpolate only using valid points
        valid_mask = ~np.isnan(vals)
        if np.count_nonzero(valid_mask) < 2:
            interp_vals = np.full_like(target_times, np.nan)
        else:
            interp_vals = np.interp(target_times, src_times[valid_mask], vals[valid_mask], left=np.nan, right=np.nan)
            
        # Smoothing
        if smooth_win > 1:
            interp_vals = _smooth_array(interp_vals, smooth_win)
            
        result[out_key] = interp_vals
        
    return result

def _smooth_array(arr: np.ndarray, win: int) -> np.ndarray:
    """Apply moving average smoothing, handling NaNs."""
    if win <= 1:
        return arr
        
    x = np.array(arr, dtype=float)
    m = ~np.isnan(x)
    if np.count_nonzero(m) == 0:
        return x
        
    out = x.copy()
    idx = np.where(m)[0]
    if idx.size == 0:
        return x
        
    # Find continuous segments of valid data
    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([0], cuts + 1))
    ends = np.concatenate((cuts, [idx.size - 1]))
    
    hl = win // 2
    hr = win - hl
    
    for si, ei in zip(starts, ends):
        s = int(idx[si]); e = int(idx[ei]); L = e - s + 1
        
        # If segment is too short, keep as is
        if L < win:
            continue
            
        seg = x[s:e+1]
        vals = np.empty_like(seg)
        
        for t in range(L):
            l = max(0, t - hl)
            r = min(L, t + hr)
            w = seg[l:r]
            vals[t] = np.mean(w)
            
        out[s:e+1] = vals
        
    return out
