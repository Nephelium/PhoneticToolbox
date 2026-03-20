import numpy as np
from typing import Optional

def compute_silence_mask(
    intensity: np.ndarray,
    threshold_ratio: float = 0.03
) -> np.ndarray:
    """
    根据音强 (Intensity) 计算静音掩码。
    
    Args:
        intensity: 音强数组 (单位: dB)。
        threshold_ratio: 静音阈值比例 (线性值，相对于最大音强)。
                         例如 0.03 表示最大音强 - 20*log10(0.03) dB 以下为静音。
                         Praat 默认使用 0.03 (relative to maximum intensity)。
    
    Returns:
        np.ndarray: 布尔掩码，True 表示静音 (Silence)，False 表示非静音。
    """
    if intensity is None or intensity.size == 0:
        return np.array([], dtype=bool)
        
    max_intensity = np.nanmax(intensity)
    if np.isnan(max_intensity):
        # 全是 NaN
        return np.ones_like(intensity, dtype=bool)
        
    # 计算相对阈值 (dB)
    # 20 * log10(threshold_ratio)
    # limit = max_intensity + relative_db
    
    ratio = max(1e-9, threshold_ratio) # 避免 log(0)
    relative_db = 20.0 * np.log10(ratio)
    
    limit = max_intensity + relative_db
    
    # 静音判定: 小于阈值 或 已经是 NaN
    # 注意: Intensity 即使在静音段也可能有小数值，或者 Praat 设为 NaN?
    # compute_energy 可能会设为 0.0 或 NaN.
    
    mask = (intensity < limit) | np.isnan(intensity)
    return mask

def compute_voiced_mask(
    f0: np.ndarray,
    silence_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    根据 F0 和静音掩码计算浊音掩码 (Voiced Mask)。
    
    Args:
        f0: F0 数组 (Hz)。
        silence_mask: 可选的静音掩码 (True=Silence)。
        
    Returns:
        np.ndarray: 布尔掩码，True 表示浊音 (Voiced)，False 表示清音或静音 (Unvoiced/Silence)。
    """
    if f0 is None or f0.size == 0:
        return np.array([], dtype=bool)
        
    # 基础判定: F0 > 0 即为 Voiced
    voiced = (f0 > 0) & (~np.isnan(f0))
    
    # 如果提供了静音掩码，静音段强制为 Unvoiced
    if silence_mask is not None:
        if len(silence_mask) != len(f0):
            # 长度不一致，尝试对齐 (取交集长度或填充?)
            # 这里为了安全，只处理长度一致或截断的情况
            min_len = min(len(f0), len(silence_mask))
            voiced[:min_len] &= (~silence_mask[:min_len])
            # 如果 f0 更长，剩下的保持原样 (假设 silence_mask 覆盖了主要区域?) 
            # 或者抛出警告。通常在 acoustic_service 中会对齐长度。
        else:
            voiced &= (~silence_mask)
            
    return voiced

def compute_zcr_voicing_mask(
    y: np.ndarray,
    fs: int,
    frameshift_ms: float,
    window_ms: float = 25.0,
    zcr_threshold: float = 3000.0,
    noise_floor_ratio: float = 0.01
) -> np.ndarray:
    """
    基于过零率 (ZCR) 和短时能量的清浊音检测。
    参考 shrp.m 中的 vda 函数实现。
    
    Args:
        y: 音频信号
        fs: 采样率
        frameshift_ms: 帧移 (ms)
        window_ms: 窗长 (ms)
        zcr_threshold: ZCR 阈值 (Hz)，高于此值且能量足够通常判为清音 (Unvoiced)。
                       但在 shrp.m 中，vda 函数逻辑似乎是排除低能量段，然后用 ZCR 辅助?
                       shrp.m 的 vda 实际上只用了 energy <= noise_floor * 3 来置为 0 (unvoiced/silence)。
                       postvda 中才用了 ZCR: if zcr < minzcr (3500) -> voiced.
                       所以: High ZCR -> Unvoiced (清音), Low ZCR -> Voiced (浊音)。
                       
                       这里我们返回的是 "Voiced Mask" (浊音掩码)，即:
                       1. 能量 > 阈值
                       2. ZCR < 阈值 (浊音通常 ZCR 低)
        noise_floor_ratio: 噪声底阈值比例 (相对于最大能量)
        
    Returns:
        np.ndarray: Voiced Mask (True = Voiced/浊音, False = Unvoiced/清音/Silence)
    """
    if y is None or y.size == 0:
        return np.array([], dtype=bool)
        
    # Frame blocking
    n_samples = len(y)
    frame_len = int(round(window_ms * fs / 1000.0))
    frame_shift = int(round(frameshift_ms * fs / 1000.0))
    
    n_frames = (n_samples - frame_len) // frame_shift + 1
    if n_frames <= 0:
        return np.array([], dtype=bool)
        
    voiced = np.ones(n_frames, dtype=bool)
    
    # Pre-calculate energy and ZCR for each frame
    # Vectorized approach or loop? Loop is easier to read and safer for edge cases.
    
    # Calculate global max energy for noise floor reference
    # Or should we use local? shrp.m uses sum(frames(1,:).^2) as noisefloor passed in?
    # Actually shrp.m vda uses noisefloor input.
    # Let's use a relative threshold.
    
    energies = np.zeros(n_frames)
    zcrs = np.zeros(n_frames)
    
    # Window function? shrp.m uses hamming.
    win = np.hamming(frame_len)
    
    for i in range(n_frames):
        start = i * frame_shift
        end = start + frame_len
        if end > n_samples:
            break
            
        seg = y[start:end]
        # Remove DC? shrp.m does it globally. We assume y is already DC removed or do it locally.
        seg = seg - np.mean(seg)
        
        # Apply window
        seg_w = seg * win
        
        # Energy
        energies[i] = np.sum(seg_w ** 2)
        
        # ZCR: count sign changes
        # zcr = sum(abs(diff(sign(seg)))) / 2 / duration
        # duration in seconds = frame_len / fs
        sign_changes = np.sum(np.abs(np.diff(np.sign(seg)))) / 2.0
        duration = frame_len / fs
        zcrs[i] = sign_changes / duration
        
    # Thresholds
    max_energy = np.nanmax(energies) if len(energies) > 0 else 0
    energy_thresh = max_energy * noise_floor_ratio
    
    # Apply logic
    # 1. Silence detection (Low Energy) -> Unvoiced
    voiced[energies <= energy_thresh] = False
    
    # 2. ZCR detection
    # High ZCR -> Unvoiced (Fricatives, etc.)
    # Low ZCR -> Voiced (Vowels, etc.)
    # shrp.m postvda: if zcr < minzcr (3500) -> voiced
    voiced[zcrs >= zcr_threshold] = False
    
    return voiced
