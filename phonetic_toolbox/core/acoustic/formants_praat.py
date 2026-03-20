import numpy as np
from pathlib import Path
from typing import Dict, Optional

try:
    import parselmouth
except ImportError:
    parselmouth = None

def compute_praat_formants(
    wav_path: Path,
    frameshift_ms: float,
    max_formant: float = 6000.0,
    num_formants: int = 5,
    window_length_s: float = 0.025,
    pre_emphasis: float = 50.0,
    pf0: Optional[np.ndarray] = None, 
) -> Dict[str, np.ndarray]:
    """
    使用 Praat (Burg 算法) 计算共振峰 (F1-F4) 和带宽 (B1-B4)。
    包含自动共振峰修正逻辑：
    - F1 限额 200-1200Hz
    - F2 限额 < 3200Hz
    - F3 限额 < 4600Hz
    - F4 限额 < 6000Hz
    
    如果共振峰频率超出限额，将自动移位至下一共振峰槽位。
    
    Args:
        wav_path (Path): 音频路径。
        frameshift_ms (float): 帧移 (ms)。
        max_formant (float): 最大共振峰频率 (Hz)。
        num_formants (int): 共振峰数量。
        window_length_s (float): 窗长 (秒)。
        pre_emphasis (float): 预加重频率 (Hz)。
        pf0 (Optional[np.ndarray]): 可选的 F0 数组，用于对齐帧数。

    Returns:
        Dict[str, np.ndarray]: 包含 pF1..pF4, pB1..pB4 的字典。
    """
    if parselmouth is None:
        raise ImportError("praat-parselmouth 未安装")

    snd = parselmouth.Sound(str(wav_path))
    time_step = frameshift_ms / 1000.0
    
    # Request at least 5 formants to allow for shifting
    # If user requests more, respect it.
    request_n_formants = max(num_formants, 5)
    
    formant = snd.to_formant_burg(
        time_step=time_step,
        max_number_of_formants=request_n_formants,
        maximum_formant=max_formant,
        window_length=window_length_s,
        pre_emphasis_from=pre_emphasis,
    )
    
    if pf0 is not None:
        n = len(pf0)
    else:
        n = int(round(snd.duration / time_step))

    # Limits definition
    F1_MIN, F1_MAX = 200.0, 1200.0
    F2_MAX = 3200.0
    F3_MAX = 4600.0
    F4_MAX = 6000.0
    
    # Output buffers
    out_F = {i: [] for i in range(1, 5)}
    out_B = {i: [] for i in range(1, 5)}
    
    for i in range(n):
        t = i * time_step
        
        # 1. Get all raw candidates for this frame
        candidates = []
        try:
            # Check up to requested number
            for k in range(1, request_n_formants + 1):
                f = formant.get_value_at_time(k, t)
                b = formant.get_bandwidth_at_time(k, t)
                if f is not None and not np.isnan(f):
                    candidates.append((f, b))
        except Exception:
            pass # Ignore errors, use what we have
            
        # 2. Assign candidates to slots with limits
        slots = {1: None, 2: None, 3: None, 4: None}
        cand_idx = 0
        current_slot = 1
        
        while cand_idx < len(candidates) and current_slot <= 4:
            f, b = candidates[cand_idx]
            
            if current_slot == 1:
                if f < F1_MIN:
                    # Too low (noise?), discard and check next candidate for Slot 1
                    cand_idx += 1
                    continue
                elif f <= F1_MAX:
                    # Fits F1
                    slots[1] = (f, b)
                    cand_idx += 1
                    current_slot += 1
                else:
                    # > F1_MAX. Too high for F1.
                    # Leave Slot 1 empty (NaN).
                    # Try this candidate for Slot 2
                    current_slot += 1
                    
            elif current_slot == 2:
                if f <= F2_MAX:
                    slots[2] = (f, b)
                    cand_idx += 1
                    current_slot += 1
                else:
                    # > F2_MAX. Too high for F2.
                    # Leave Slot 2 empty. Try Slot 3.
                    current_slot += 1
                    
            elif current_slot == 3:
                if f <= F3_MAX:
                    slots[3] = (f, b)
                    cand_idx += 1
                    current_slot += 1
                else:
                    current_slot += 1
            
            elif current_slot == 4:
                if f <= F4_MAX:
                    slots[4] = (f, b)
                    cand_idx += 1
                    current_slot += 1
                else:
                    current_slot += 1
                    
        # 3. Store results
        for k in range(1, 5):
            if slots[k]:
                out_F[k].append(slots[k][0])
                out_B[k].append(slots[k][1])
            else:
                out_F[k].append(np.nan)
                out_B[k].append(np.nan)

    result = {}
    for i in range(1, 5):
        result[f"pF{i}"] = np.array(out_F[i], dtype=float)
        result[f"pB{i}"] = np.array(out_B[i], dtype=float)
        
    return result
