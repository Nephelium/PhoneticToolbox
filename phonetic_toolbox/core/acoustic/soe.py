import numpy as np
from scipy.signal import lfilter, resample_poly
from typing import Tuple, Optional

def compute_soe(
    y: np.ndarray,
    fs: int,
    frameshift_ms: float,
    f0: np.ndarray,
    target_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Strength of Excitation (SoE) and Epochs using ZFF method.
    Based on func_getSoE.m by Soo Jin Park.
    """
    
    # 1. Resample to 10kHz
    target_fs = 10000
    if fs != target_fs:
        # Use resample_poly for efficiency and stability
        # Find integer ratio approximation
        import math
        gcd = math.gcd(int(fs), int(target_fs))
        up = int(target_fs // gcd)
        down = int(fs // gcd)
        
        try:
            snd = resample_poly(y, up, down)
        except Exception as e:
            print(f"Resample failed: {e}")
            return np.full(target_len, np.nan), np.array([])
    else:
        snd = y.copy()
        
    # 2. ZFF parameters
    # METHOD 'whole' (from MATLAB script)
    # Calculate average pitch period in samples (at 10kHz)
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) == 0:
        return np.full(target_len, np.nan), np.array([])
        
    mean_f0 = np.mean(f0_clean)
    if mean_f0 <= 0:
        mean_n0 = target_fs / 100.0 # Default fallback
    else:
        mean_n0 = target_fs / mean_f0
        
    # 3. ZFF Algorithm
    z = zff(snd, mean_n0)
    
    # 4. Normalize ZFF
    max_abs_z = np.max(np.abs(z))
    if max_abs_z > 0:
        z = 0.95 * z / max_abs_z
        
    # 5. Epoch Detection
    # Zero crossings: z[n] <= 0 and z[n-1] > 0 (falling slope)
    # MATLAB: tf = z1>0 & z<=0;
    z1 = np.roll(z, 1)
    z1[0] = np.nan # First sample has no previous
    tf = (z1 > 0) & (z <= 0)
    
    # Derivative for SoE
    # diff = -get_delta(z.', 2);
    # get_delta uses a window. Simplified: diff[n] = z[n+1] - z[n-1] approx?
    # The MATLAB get_delta uses a weighted sum window D=2.
    # numer = 1*(x(n+1)-x(n-1)) + 2*(x(n+2)-x(n-2))
    # denom = 2*1^2 + 2*2^2 = 2 + 8 = 10
    d_z = -get_delta(z, 2)
    
    # 6. Fit into frames
    soe = np.full(target_len, np.nan)
    # epochs indices at 10kHz
    epoch_indices_10k = np.where(tf)[0]
    
    for idx in epoch_indices_10k:
        # Map to frame index
        # time = idx / target_fs
        # frame = round(time / (frameshift_ms / 1000))
        # 0-based index
        t_sec = idx / target_fs
        frm_idx = int(round(t_sec / (frameshift_ms / 1000)))
        
        if 0 <= frm_idx < target_len:
            soe[frm_idx] = d_z[idx]
            
    return soe, epoch_indices_10k

def zff(x: np.ndarray, n0: float) -> np.ndarray:
    """Zero Frequency Filtering."""
    alpha = 0.999
    
    # Differenced signal
    # s = filter([1 -1], 1, x) -> s[n] = x[n] - x[n-1]
    # In scipy.signal.lfilter: b=[1, -1], a=[1]
    s = lfilter([1, -1], [1], x)
    
    # 1st ZFR (Zero Frequency Resonator)
    # u = filter(1, [1 -2*alpha alpha^2], s)
    # b=[1], a=[1, -2*alpha, alpha^2]
    a_poly = [1, -2*alpha, alpha**2]
    u = lfilter([1], a_poly, s)
    
    # Trend removal
    v = remove_trend(u, int(round(n0 / 1.5)))
    
    # 2nd ZFR
    y = lfilter([1], a_poly, v)
    
    # Trend removal
    z = remove_trend(y, int(round(n0 / 1.5)))
    
    return z

def remove_trend(x: np.ndarray, N: int) -> np.ndarray:
    """Subtract smoothed curve (moving average)."""
    if N <= 0:
        return x
        
    width = 2 * N + 1
    # Moving average filter
    # c = filter(ones(width,1)/width, 1, x)
    b = np.ones(width) / width
    c = lfilter(b, [1], x)
    
    # MATLAB's filter introduces delay of (width-1)/2 ?
    # Wait, MATLAB filter(b, a, x) is causal.
    # The script does:
    # cbegin = cumsum(x(1:width-2)); ... complex boundary handling
    # The MATLAB script calculates 'c' using causal filter, then fixes boundaries?
    # Actually, lfilter is causal. The moving average should be centered.
    # But the MATLAB script uses causal filter 'filter' then constructs 'c' by stitching?
    # Let's look closely at MATLAB code:
    # c = filter(..., x); -> Causal moving average. value at i is mean of i-width+1 .. i
    # Then it constructs c using cbegin, c(width:end), cend.
    # c(width:end) corresponds to x(width:end) processed.
    # Essentially it shifts the result to compensate for delay.
    # delay of causal average of width W is (W-1)/2.
    # So we should shift 'c' back by N samples (since width=2N+1).
    
    # Let's replicate the shift behavior.
    # x: [0, 1, 2, ...]
    # width = 3 (N=1). b = [1/3, 1/3, 1/3].
    # y[0] = x[0]/3
    # y[1] = (x[0]+x[1])/3
    # y[2] = (x[0]+x[1]+x[2])/3  <- This is average centered at 1 (if we consider lag)
    
    # MATLAB code:
    # c = [cbegin; c(width:end); cend]
    # c(width) is the first "full" average.
    # It replaces the first 'width-1' samples with 'cbegin' and last with 'cend'.
    # But wait, c(width:end) takes the valid part of the filtered output.
    # It effectively shifts left by 'width-1'? No.
    # Let's use scipy.ndimage.uniform_filter1d or similar for zero-phase, 
    # but the script seems to want to replicate a specific boundary behavior.
    # I will just use a centered moving average for simplicity and robustness.
    # The "trend" is the low frequency component.
    
    # Implementation using convolution with 'valid' mode + boundary padding
    w = np.ones(width) / width
    # We want y[i] = mean(x[i-N : i+N+1])
    # correlate or convolve.
    c = np.convolve(x, w, mode='same')
    
    return x - c

def get_delta(x: np.ndarray, D: int) -> np.ndarray:
    """Calculate derivative using regression window."""
    # MATLAB:
    # for theta = 1:D
    #   numer = numer + theta*(x(:, nf+theta)-x(:, nf-theta));
    #   denom = denom + 2*theta^2;
    # dx(:,nf) = numer/denom;
    
    dx = np.zeros_like(x)
    denom = 0
    for theta in range(1, D + 1):
        denom += 2 * theta**2
        
    # Vectorized
    # Pad x with NaNs or edge values? MATLAB code handles boundaries with catch block (fallback to simple diff)
    # We can use np.pad
    x_pad = np.pad(x, (D, D), mode='edge')
    
    numer = np.zeros_like(x)
    for theta in range(1, D + 1):
        # x[n+theta] corresponds to x_pad[n+D+theta]
        # x[n-theta] corresponds to x_pad[n+D-theta]
        # range of n is 0..len(x)-1
        # indices in pad: D..len(x)+D-1
        term = theta * (x_pad[D+theta : D+len(x)+theta] - x_pad[D-theta : D+len(x)-theta])
        numer += term
        
    dx = numer / denom
    return dx
