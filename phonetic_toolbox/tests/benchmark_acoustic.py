
import time
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonetic_toolbox.models.config import AcousticConfig
from phonetic_toolbox.core.acoustic import (
    compute_praat_f0,
    compute_reaper_f0,
    compute_praat_formants,
    compute_harmonics_H1H2H4,
    compute_amplitudes_A1A2A3,
    compute_harmonic_at_fixed_freq,
    compute_cpp,
    compute_hnr,
    compute_shr,
    compute_jitter_shimmer,
    compute_spectral_slope,
    compute_soe,
    compute_spectral_features_batch,
    correct_formants,
    compute_H1A1A2A3_corrected,
    compute_H1H2_H2H4_corrected,
    compute_corrections_H2KH5K,
    compute_energy
)

def benchmark():
    # 1. Find longest wav file
    search_dir = Path("C:/Users/13680/Desktop/测试音频/textgrid测试")
    if not search_dir.exists():
        print(f"Directory not found: {search_dir}")
        return

    wav_files = list(search_dir.glob("*.wav"))
    if not wav_files:
        print("No wav files found.")
        return

    # Find the largest file (assuming size correlates with duration)
    target_file = max(wav_files, key=lambda p: p.stat().st_size)
    print(f"Benchmarking with file: {target_file}")
    print("-" * 50)

    # Config
    config = AcousticConfig()
    
    # Timing dictionary
    timings = {}
    
    # --- 1. Read Audio ---
    t0 = time.perf_counter()
    fs, y_int = wavfile.read(str(target_file))
    if y_int.dtype == np.int16:
        y = y_int.astype(np.float64) / 32768.0
    elif y_int.dtype == np.int32:
        y = y_int.astype(np.float64) / 2147483648.0
    else:
        y = y_int.astype(np.float64)
    
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    t1 = time.perf_counter()
    timings["Read Audio"] = (t1 - t0) * 1000
    
    duration = len(y) / fs
    print(f"Audio Duration: {duration:.2f} s")
    print(f"Sampling Rate: {fs} Hz")
    print("-" * 50)

    frameshift_ms = config.frameshift_ms
    min_f0 = config.min_f0
    max_f0 = config.max_f0
    n_periods = config.n_periods

    # --- 2. Formants (Praat) ---
    t0 = time.perf_counter()
    try:
        formant_res = compute_praat_formants(
            target_file, 
            frameshift_ms, 
            max_formant=config.max_formant,
            num_formants=config.num_formants
        )
        pF1 = formant_res.get("pF1")
        pF2 = formant_res.get("pF2")
        pF3 = formant_res.get("pF3")
        pF4 = formant_res.get("pF4")
        pB1 = formant_res.get("pB1")
        pB2 = formant_res.get("pB2")
        pB3 = formant_res.get("pB3")
        pB4 = formant_res.get("pB4")
    except Exception as e:
        print(f"Formants failed: {e}")
        pF1 = None
    t1 = time.perf_counter()
    timings["Formants (Praat)"] = (t1 - t0) * 1000

    # --- 3. F0 (Praat) ---
    t0 = time.perf_counter()
    try:
        pf0 = compute_praat_f0(target_file, frameshift_ms, min_f0, max_f0, "cc")
    except Exception as e:
        print(f"Praat F0 failed: {e}")
        pf0 = np.array([])
    t1 = time.perf_counter()
    timings["F0 (Praat)"] = (t1 - t0) * 1000

    # --- 4. F0 (REAPER) ---
    t0 = time.perf_counter()
    try:
        # REAPER usually faster via CLI, check config
        rf0_res = compute_reaper_f0(
            target_file, 
            frameshift_ms / 1000.0, 
            min_f0, 
            max_f0,
            hilbert=config.reaper_hilbert,
            no_highpass=config.reaper_no_highpass,
            reaper_bin=config.reaper_bin_path
        )
        rf0 = rf0_res.get("rF0", np.array([]))
    except Exception as e:
        print(f"REAPER F0 failed: {e}")
        rf0 = np.array([])
    t1 = time.perf_counter()
    timings["F0 (REAPER)"] = (t1 - t0) * 1000

    # Align lengths
    target_len = len(pF1) if pF1 is not None else 0
    if len(pf0) > target_len: pf0 = pf0[:target_len]
    if len(rf0) > target_len: rf0 = rf0[:target_len]
    
    # Use pF0 for subsequent calcs
    f0 = pf0
    voiced_mask = (f0 > 0)

    # --- 5. Harmonics & Amplitudes & H2K/H5K (Batch) ---
    t0 = time.perf_counter()
    try:
        spec_res = compute_spectral_features_batch(
            y, fs, frameshift_ms, f0, pF1, pF2, pF3, n_periods, voiced_mask
        )
        h1 = spec_res["H1"]
        h2 = spec_res["H2"]
        h4 = spec_res["H4"]
        a1 = spec_res["A1"]
        a2 = spec_res["A2"]
        a3 = spec_res["A3"]
        h2k = spec_res["H2K"]
        h5k = spec_res["H5K"]
    except Exception as e:
        print(f"Spectral Batch failed: {e}")
        h1=h2=h4=a1=a2=a3=h2k=h5k=None
    t1 = time.perf_counter()
    timings["Spectral Batch (H1-H4, A1-A3, H2K, H5K)"] = (t1 - t0) * 1000

    # --- 7. Corrections (Tilt & Iseli) ---
    t0 = time.perf_counter()
    try:
        # H1H2c, H2H4c
        compute_H1H2_H2H4_corrected(h1, h2, h4, fs, f0, pF1, pF2, pB1, pB2)
        # H1A1c, etc.
        compute_H1A1A2A3_corrected(h1, a1, a2, a3, fs, f0, pF1, pF2, pF3, pB1, pB2, pB3)
        
        # Iseli correction (using pre-computed h2k, h5k)
        f4_arr = pF4 if pF4 is not None else np.full(target_len, np.nan)
        b4_arr = pB4 if pB4 is not None else np.full(target_len, np.nan)
        compute_corrections_H2KH5K(h4, h2k, h5k, int(fs), f0, pF1, pF2, pF3, f4_arr, pB1, pB2, pB3, b4_arr)
    except Exception as e:
        print(f"Corrections failed: {e}")
    t1 = time.perf_counter()
    timings["Corrections (Tilt & Iseli)"] = (t1 - t0) * 1000

    # --- 8. CPP ---
    t0 = time.perf_counter()
    try:
        compute_cpp(y, fs, frameshift_ms, f0, n_periods, voiced_mask)
    except Exception as e:
        print(f"CPP failed: {e}")
    t1 = time.perf_counter()
    timings["CPP"] = (t1 - t0) * 1000

    # --- 9. HNR ---
    t0 = time.perf_counter()
    try:
        compute_hnr(y, fs, frameshift_ms, f0, n_periods, voiced_mask=voiced_mask)
    except Exception as e:
        print(f"HNR failed: {e}")
    t1 = time.perf_counter()
    timings["HNR"] = (t1 - t0) * 1000

    # --- 10. SHR ---
    t0 = time.perf_counter()
    try:
        compute_shr(y, fs, frameshift_ms, f0, min_f0, max_f0, voiced_mask=voiced_mask)
    except Exception as e:
        print(f"SHR failed: {e}")
    t1 = time.perf_counter()
    timings["SHR"] = (t1 - t0) * 1000

    # --- 11. Spectral Slope ---
    t0 = time.perf_counter()
    try:
        compute_spectral_slope(y, fs, frameshift_ms, f0, min_pitch=min_f0, voiced_mask=voiced_mask)
    except Exception as e:
        print(f"Spectral Slope failed: {e}")
    t1 = time.perf_counter()
    timings["Spectral Slope"] = (t1 - t0) * 1000

    # --- 12. SoE ---
    t0 = time.perf_counter()
    try:
        compute_soe(y, fs, frameshift_ms, f0, target_len)
    except Exception as e:
        print(f"SoE failed: {e}")
    t1 = time.perf_counter()
    timings["SoE"] = (t1 - t0) * 1000

    # --- 13. Jitter/Shimmer ---
    t0 = time.perf_counter()
    try:
        compute_jitter_shimmer(y, fs, frameshift_ms, 40, voiced_mask=(f0 > 0), min_f0=min_f0, max_f0=max_f0)
    except Exception as e:
        print(f"Jitter/Shimmer failed: {e}")
    t1 = time.perf_counter()
    timings["Jitter/Shimmer"] = (t1 - t0) * 1000

    # --- 14. Intensity ---
    t0 = time.perf_counter()
    try:
        compute_energy(y, fs, frameshift_ms, f0)
    except Exception as e:
        print(f"Intensity failed: {e}")
    t1 = time.perf_counter()
    timings["Intensity"] = (t1 - t0) * 1000

    # Print Results
    print("\nBenchmark Results (ms):")
    print("-" * 50)
    
    # Sort by time desc
    sorted_timings = sorted(timings.items(), key=lambda item: item[1], reverse=True)
    
    for k, v in sorted_timings:
        print(f"{k:<30}: {v:.2f} ms")
    
    total_time = sum(timings.values())
    print("-" * 50)
    print(f"Total Computation Time: {total_time:.2f} ms ({total_time/1000:.2f} s)")

if __name__ == "__main__":
    benchmark()
