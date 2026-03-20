import numpy as np
from typing import Dict, Optional
from scipy.signal import medfilt

def hawks_miller_bw(Fx: float, F0: float) -> float:
    if np.isnan(Fx) or Fx <= 0:
        return np.nan
    S = 1.0 + 0.25 * ((F0 if not np.isnan(F0) else 132.0) - 132.0) / 88.0
    C1 = np.array([165.327516, -6.73636734e-1, 1.80874446e-3, -4.52201682e-6, 7.49514000e-9, -4.70219241e-12])
    C2 = np.array([15.8146139, 8.10159009e-2, -9.79728215e-5, 5.28725064e-8, -1.07099364e-11, 7.91528509e-16])
    F = np.array([1.0, Fx, Fx ** 2, Fx ** 3, Fx ** 4, Fx ** 5], dtype=float)
    bw = float(C1 @ F) if Fx < 500.0 else float(C2 @ F)
    return float(S * bw)

def iseli_correction(f: float, Fx: float, Bx: float, fs: int) -> float:
    """Iseli & Alwan (1999) harmonic-formant correction (dB)."""
    if any(np.isnan(x) for x in [f, Fx, Bx]) or f <= 0 or Fx <= 0 or Bx <= 0:
        return 0.0
    r = np.exp(-np.pi * Bx / float(fs))
    omega_x = 2.0 * np.pi * Fx / float(fs)
    omega = 2.0 * np.pi * f / float(fs)
    a = r ** 2 + 1.0 - 2.0 * r * np.cos(omega_x + omega)
    b = r ** 2 + 1.0 - 2.0 * r * np.cos(omega_x - omega)
    num = r ** 2 + 1.0 - 2.0 * r * np.cos(omega_x)
    return float(-10.0 * (np.log10(a) + np.log10(b)) + 20.0 * np.log10(num))

def compute_corrections_H2KH5K(
    H4: np.ndarray,
    H2K: np.ndarray,
    H5K: np.ndarray,
    fs: int,
    F0: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    F3: np.ndarray,
    F4: np.ndarray,
    B1: Optional[np.ndarray] = None,
    B2: Optional[np.ndarray] = None,
    B3: Optional[np.ndarray] = None,
    B4: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Correct H4*-2K* and 2K*-5K using VoiceSauce-compatible Iseli logic."""
    n = int(F0.shape[0])
    H42Kc = np.full(n, np.nan, dtype=float)
    H2KH5Kc = np.full(n, np.nan, dtype=float)
    
    for i in range(n):
        f0 = float(F0[i])
        if np.isnan(f0) or f0 <= 0:
            continue
            
        f1 = float(F1[i]) if i < F1.shape[0] else np.nan
        f2 = float(F2[i]) if i < F2.shape[0] else np.nan
        f3 = float(F3[i]) if i < F3.shape[0] else np.nan
        
        b1 = float(B1[i]) if B1 is not None else hawks_miller_bw(f1, f0)
        b2 = float(B2[i]) if B2 is not None else hawks_miller_bw(f2, f0)
        b3 = float(B3[i]) if B3 is not None else hawks_miller_bw(f3, f0)
        
        # Calculate frequencies for H4, H2K, H5K
        h4_freq = 4.0 * f0
        
        # H2K is the harmonic closest to 2000Hz
        h2k_idx = max(1, int(round(2000.0 / f0)))
        h2k_freq = h2k_idx * f0
        
        H4_corr = float(H4[i])
        for fx, bx in [(f1, b1), (f2, b2)]:
            if not np.isnan(fx):
                 H4_corr -= iseli_correction(h4_freq, fx, bx, fs)
                 
        H2K_corr = float(H2K[i])
        for fx, bx in [(f1, b1), (f2, b2), (f3, b3)]:
            if not np.isnan(fx):
                 H2K_corr -= iseli_correction(h2k_freq, fx, bx, fs)
                 
        H42Kc[i] = H4_corr - H2K_corr
        H2KH5Kc[i] = H2K_corr - float(H5K[i])
        
    return {"H42Kc": H42Kc, "H2KH5Kc": H2KH5Kc}

def compute_H1A1A2A3_corrected(
    H1: np.ndarray,
    A1: np.ndarray,
    A2: np.ndarray,
    A3: np.ndarray,
    fs: int,
    F0: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    F3: np.ndarray,
    B1: Optional[np.ndarray] = None,
    B2: Optional[np.ndarray] = None,
    B3: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Correct H1*-A1*, H1*-A2*, H1*-A3*."""
    n = int(H1.shape[0])
    H1A1c = np.full(n, np.nan, dtype=float)
    H1A2c = np.full(n, np.nan, dtype=float)
    H1A3c = np.full(n, np.nan, dtype=float)
    for i in range(n):
        f0 = float(F0[i])
        if np.isnan(f0) or f0 <= 0:
            continue
        f1 = float(F1[i]) if i < F1.shape[0] else np.nan
        f2 = float(F2[i]) if i < F2.shape[0] else np.nan
        f3 = float(F3[i]) if i < F3.shape[0] else np.nan
        
        b1 = float(B1[i]) if B1 is not None else hawks_miller_bw(f1, f0)
        b2 = float(B2[i]) if B2 is not None else hawks_miller_bw(f2, f0)
        b3 = float(B3[i]) if B3 is not None else hawks_miller_bw(f3, f0)
        
        H1_corr = float(H1[i])
        if not np.isnan(f1): H1_corr -= iseli_correction(f0, f1, b1, fs)
        if not np.isnan(f2): H1_corr -= iseli_correction(f0, f2, b2, fs)
        
        A1_corr = float(A1[i])
        if not np.isnan(f1): A1_corr -= iseli_correction(f1, f1, b1, fs)
        if not np.isnan(f2): A1_corr -= iseli_correction(f1, f2, b2, fs)
        
        A2_corr = float(A2[i])
        if not np.isnan(f1): A2_corr -= iseli_correction(f2, f1, b1, fs)
        if not np.isnan(f2): A2_corr -= iseli_correction(f2, f2, b2, fs)
        
        A3_corr = float(A3[i])
        if not np.isnan(f1): A3_corr -= iseli_correction(f3, f1, b1, fs)
        if not np.isnan(f2): A3_corr -= iseli_correction(f3, f2, b2, fs)
        if not np.isnan(f3): A3_corr -= iseli_correction(f3, f3, b3, fs)
        
        H1A1c[i] = H1_corr - A1_corr
        H1A2c[i] = H1_corr - A2_corr
        H1A3c[i] = H1_corr - A3_corr
    return {"H1A1c": H1A1c, "H1A2c": H1A2c, "H1A3c": H1A3c}

def compute_H1H2_H2H4_corrected(
    H1: np.ndarray,
    H2: np.ndarray,
    H4: np.ndarray,
    fs: int,
    F0: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    B1: Optional[np.ndarray] = None,
    B2: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Correct H1*-H2*, H2*-H4*."""
    n = int(F0.shape[0])
    out_H1H2c = np.full(n, np.nan, dtype=float)
    out_H2H4c = np.full(n, np.nan, dtype=float)
    for i in range(n):
        f0 = float(F0[i])
        if np.isnan(f0) or f0 <= 0:
            continue
        f1 = float(F1[i]) if i < F1.shape[0] else np.nan
        f2 = float(F2[i]) if i < F2.shape[0] else np.nan
        
        b1 = float(B1[i]) if B1 is not None else hawks_miller_bw(f1, f0)
        b2 = float(B2[i]) if B2 is not None else hawks_miller_bw(f2, f0)
        
        H1_corr = float(H1[i])
        if not np.isnan(f1): H1_corr -= iseli_correction(f0, f1, b1, fs)
        if not np.isnan(f2): H1_corr -= iseli_correction(f0, f2, b2, fs)
        
        H2_corr = float(H2[i])
        if not np.isnan(f1): H2_corr -= iseli_correction(2.0 * f0, f1, b1, fs)
        if not np.isnan(f2): H2_corr -= iseli_correction(2.0 * f0, f2, b2, fs)
        
        H4_corr = float(H4[i])
        if not np.isnan(f1): H4_corr -= iseli_correction(4.0 * f0, f1, b1, fs)
        if not np.isnan(f2): H4_corr -= iseli_correction(4.0 * f0, f2, b2, fs)
        
        out_H1H2c[i] = H1_corr - H2_corr
        out_H2H4c[i] = H2_corr - H4_corr
        
    return {"H1H2c": out_H1H2c, "H2H4c": out_H2H4c}

def correct_formants(
    formants: Dict[str, np.ndarray],
    window_length: int = 5
) -> Dict[str, np.ndarray]:
    """
    Smooth formant tracks using a median filter.
    
    Args:
        formants: Dictionary containing formant tracks (e.g., 'pF1', 'pB1', etc.)
        window_length: Window size for median filter (must be odd, default 5)
        
    Returns:
        Dictionary with smoothed formant tracks.
    """
    result = {}
    w = window_length if window_length % 2 == 1 else window_length + 1
    
    for k, v in formants.items():
        if isinstance(v, np.ndarray) and v.size > 0:
            if k.startswith("pF") or k.startswith("pB"):
                # Use scipy.signal.medfilt for smoothing
                try:
                    result[k] = medfilt(v, kernel_size=w)
                except Exception:
                    result[k] = v.copy()
            else:
                result[k] = v.copy()
        else:
            result[k] = v
            
    return result
