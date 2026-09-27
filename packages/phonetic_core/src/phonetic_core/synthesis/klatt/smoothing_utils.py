"""
Smoothing utilities for parameter curves.

This module provides functions for smoothing parameter data while
excluding specific time regions (e.g., consonant segments).
"""

import numpy as np
from typing import List, Tuple


def smooth_excluding_regions(data: np.ndarray, t_grid: np.ndarray, 
                             exclude_regions: List[Tuple[float, float]], 
                             smooth_size: int,
                             mode: str = 'nearest') -> np.ndarray:
    """
    Apply smoothing to data while excluding specified time regions.
    
    This function applies uniform filtering (moving average) to the data,
    but preserves the original values in excluded regions (e.g., consonant segments).
    The smoothing is applied only to non-excluded regions.
    
    Args:
        data: 1D numpy array of values to smooth
        t_grid: 1D numpy array of time values corresponding to data
        exclude_regions: List of (start_time, end_time) tuples to exclude from smoothing
        smooth_size: Size of the smoothing window
        mode: Mode for uniform_filter1d ('nearest', 'constant', etc.)
        
    Returns:
        Smoothed data array with excluded regions preserved
    """
    from scipy.ndimage import uniform_filter1d
    
    if smooth_size <= 1 or len(data) == 0:
        return data.copy()
    
    if not exclude_regions:
        # No regions to exclude, apply normal smoothing
        return uniform_filter1d(data, size=smooth_size, mode=mode)
    
    # Create a mask for excluded regions
    exclude_mask = np.zeros(len(data), dtype=bool)
    for (t_start, t_end) in exclude_regions:
        region_mask = (t_grid >= t_start) & (t_grid <= t_end)
        exclude_mask |= region_mask
    
    # Store original values in excluded regions
    original_excluded = data[exclude_mask].copy()
    
    # Apply smoothing to the entire array
    smoothed = uniform_filter1d(data, size=smooth_size, mode=mode)
    
    # Restore original values in excluded regions
    smoothed[exclude_mask] = original_excluded
    
    return smoothed
