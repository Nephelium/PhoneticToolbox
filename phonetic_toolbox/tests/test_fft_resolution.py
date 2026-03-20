
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
import sys
import os
import time
import pandas as pd

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from phonetic_toolbox.models.config import AcousticConfig
from phonetic_toolbox.core.acoustic import (
    compute_praat_f0,
    compute_praat_formants,
    compute_spectral_features_batch,
    compute_H1H2_H2H4_corrected,
    compute_H1A1A2A3_corrected
)

def test_fft_resolutions():
    # 1. Setup Audio
    search_dir = Path("C:/Users/13680/Desktop/测试音频/textgrid测试")
    if not search_dir.exists():
        print(f"Directory not found: {search_dir}")
        return

    wav_files = list(search_dir.glob("*.wav"))
    if not wav_files:
        print("No wav files found.")
        return

    target_file = max(wav_files, key=lambda p: p.stat().st_size)
    print(f"Testing with file: {target_file}")
    
    # Read Audio
    fs, y_int = wavfile.read(str(target_file))
    if y_int.dtype == np.int16:
        y = y_int.astype(np.float64) / 32768.0
    elif y_int.dtype == np.int32:
        y = y_int.astype(np.float64) / 2147483648.0
    else:
        y = y_int.astype(np.float64)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Config & Prep
    config = AcousticConfig()
    frameshift_ms = config.frameshift_ms
    min_f0 = config.min_f0
    max_f0 = config.max_f0
    n_periods = config.n_periods
    
    # Get F0 and Formants
    print("Computing F0 and Formants...")
    f0 = compute_praat_f0(target_file, frameshift_ms, min_f0, max_f0, "cc")
    voiced_mask = (f0 > 0)
    
    formant_res = compute_praat_formants(target_file, frameshift_ms, max_formant=config.max_formant, num_formants=config.num_formants)
    pF1 = formant_res.get("pF1")
    pF2 = formant_res.get("pF2")
    pF3 = formant_res.get("pF3")
    
    # Align
    target_len = len(pF1)
    if len(f0) > target_len: f0 = f0[:target_len]
    
    # Resolutions to test
    resolutions = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    results = {}
    
    print("-" * 80)
    print(f"{'Resolution':<10} | {'Time (ms)':<10} | {'Mean H1':<10} | {'Mean H1-H2':<12} | {'Mean H1-A1':<12}")
    print("-" * 80)
    
    baseline_res = None # Store 0.5Hz result as baseline
    
    for res in resolutions:
        t0 = time.perf_counter()
        spec_res = compute_spectral_features_batch(
            y, fs, frameshift_ms, f0, pF1, pF2, pF3, n_periods, voiced_mask,
            target_resolution=res
        )
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000
        
        # Calculate derived params
        h1 = spec_res["H1"]
        h2 = spec_res["H2"]
        a1 = spec_res["A1"]
        
        h1h2 = h1 - h2
        h1a1 = h1 - a1
        
        mean_h1 = np.nanmean(h1)
        mean_h1h2 = np.nanmean(h1h2)
        mean_h1a1 = np.nanmean(h1a1)
        
        print(f"{res:<10.1f} | {elapsed:<10.2f} | {mean_h1:<10.2f} | {mean_h1h2:<12.2f} | {mean_h1a1:<12.2f}")
        
        results[res] = {
            "H1": h1,
            "H2": h2,
            "A1": a1,
            "H1-H2": h1h2,
            "H1-A1": h1a1
        }
        
        if res == 0.5:
            baseline_res = results[res]

    print("-" * 80)
    print("\nComparison with Baseline (0.5 Hz):")
    print(f"{'Resolution':<10} | {'Max Diff H1':<12} | {'Max Diff H1-H2':<15} | {'Max Diff H1-A1':<15}")
    print("-" * 80)
    
    for res in resolutions[1:]:
        curr = results[res]
        base = baseline_res
        
        diff_h1 = np.nanmax(np.abs(curr["H1"] - base["H1"]))
        diff_h1h2 = np.nanmax(np.abs(curr["H1-H2"] - base["H1-H2"]))
        diff_h1a1 = np.nanmax(np.abs(curr["H1-A1"] - base["H1-A1"]))
        
        print(f"{res:<10.1f} | {diff_h1:<12.4f} | {diff_h1h2:<15.4f} | {diff_h1a1:<15.4f}")

if __name__ == "__main__":
    test_fft_resolutions()
