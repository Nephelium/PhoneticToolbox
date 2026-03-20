
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
import sys
import os

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from phonetic_toolbox.models.config import AcousticConfig
from phonetic_toolbox.core.acoustic import (
    compute_praat_f0,
    compute_praat_formants,
    compute_spectral_features_batch
)

def compare_methods():
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
    
    # Get F0 and Formants (Common Inputs)
    print("Computing F0 and Formants...")
    pf0 = compute_praat_f0(target_file, frameshift_ms, min_f0, max_f0, "cc")
    
    formant_res = compute_praat_formants(target_file, frameshift_ms, max_formant=config.max_formant, num_formants=config.num_formants)
    pF1 = formant_res.get("pF1")
    pF2 = formant_res.get("pF2")
    pF3 = formant_res.get("pF3")
    
    # Align
    target_len = len(pF1)
    if len(pf0) > target_len: pf0 = pf0[:target_len]
    f0 = pf0
    voiced_mask = (f0 > 0)
    
    print("-" * 60)
    print("Running spectral batch method")
    print("-" * 60)
    spec_res_new = compute_spectral_features_batch(y, fs, frameshift_ms, f0, pF1, pF2, pF3, n_periods, voiced_mask)

    required = ["H1", "H2", "H4", "A1", "A2", "A3", "H2K", "H5K"]
    print(f"{'Param':<6} | {'Valid Frames':<12} | {'Min':<10} | {'Max':<10}")
    print("-" * 60)
    for name in required:
        arr = np.array(spec_res_new.get(name, []), dtype=float)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            print(f"{name:<6} | {0:<12} | {'-':<10} | {'-':<10}")
        else:
            print(f"{name:<6} | {len(valid):<12} | {np.min(valid):<10.3f} | {np.max(valid):<10.3f}")

if __name__ == "__main__":
    compare_methods()
