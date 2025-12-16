import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

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

    # Attempt to load corresponding timestamps file for alignment
    p = Path(pkl_path)
    # The timestamps file is expected to be [stem]_timestamps.pkl
    # If the user renamed lip_metrics_data.pkl to [stem].pkl, then [stem]_timestamps.pkl should exist.
    ts_path = p.parent / f"{p.stem}_timestamps.pkl"
    
    audio_start_time = None
    if ts_path.exists():
        try:
            with open(ts_path, 'rb') as f:
                ts_data = pickle.load(f)
                audio_start_time = ts_data.get('start_time')
                print(f"Loaded audio timestamps from {ts_path}, start_time: {audio_start_time}")
        except Exception as e:
            print(f"Error loading timestamps {ts_path}: {e}")
    
    # Determine source times
    src_times = None
    
    # Priority 1: Use absolute timestamps aligned with audio start time
    if audio_start_time is not None:
        abs_times = data.get('absolute_timestamps')
        if abs_times and len(abs_times) > 0:
            abs_times = np.array(abs_times, dtype=float)
            src_times = abs_times - audio_start_time
            print("Using aligned absolute timestamps for lip data")

    # Priority 2: Use relative times from lip data (assume aligned start)
    if src_times is None:
        rel_times = data.get('relative_times')
        if rel_times and len(rel_times) > 0:
            src_times = np.array(rel_times, dtype=float)
            print("Using relative timestamps from lip data (no audio alignment)")
            
    if src_times is None or len(src_times) < 2:
        print("No valid timestamps found in lip data")
        return {}
    
    # Align first data point to time 0 (shift all times so first point starts at 0)
    first_time = src_times[0]
    if first_time != 0:
        src_times = src_times - first_time
        print(f"Shifted lip data times by {first_time:.4f}s to align first point to t=0")
    
    # Mapping from internal key to output key
    # Using the normalized values as requested
    key_map = {
        'area': 'LipArea',           # Lip Area Ratio
        'outer_width': 'LipWidth',   # Normalized Outer Lip Width
        'open': 'LipOpen',           # Normalized Lip Openness
        'circularity': 'LipCirc'     # Lip Circularity
    }
    
    result = {}
    
    for src_key, out_key in key_map.items():
        vals = data.get(src_key)
        if vals is None or len(vals) != len(src_times):
            continue
            
        vals = np.array(vals, dtype=float)
        
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
