import numpy as np
from pathlib import Path
import warnings

try:
    import parselmouth
except ImportError:
    parselmouth = None

def compute_praat_f0(
    wav_path: Path,
    frameshift_ms: float,
    min_f0: float,
    max_f0: float,
    method: str = "cc",
) -> np.ndarray:
    """
    使用 Praat (Parselmouth) 计算 F0。

    Args:
        wav_path (Path): 音频文件路径。
        frameshift_ms (float): 帧移 (ms)。
        min_f0 (float): 最小基频 (Hz)。
        max_f0 (float): 最大基频 (Hz)。
        method (str): "cc" (互相关) 或 "ac" (自相关)。

    Returns:
        np.ndarray: F0 数组 (Hz)。无声段为 NaN。
    """
    if parselmouth is None:
        raise ImportError("praat-parselmouth 未安装")

    snd = parselmouth.Sound(str(wav_path))
    time_step = frameshift_ms / 1000.0

    try:
        if method and method.lower() == "cc":
            pitch = parselmouth.praat.call(
                snd,
                "To Pitch (cc)",
                time_step,
                float(min_f0),
                float(max_f0),
            )
        else:
            pitch = snd.to_pitch(
                time_step=time_step,
                pitch_floor=float(min_f0),
                pitch_ceiling=float(max_f0),
            )
    except Exception:
        # Fallback
        pitch = snd.to_pitch(
            time_step=time_step,
            pitch_floor=float(min_f0),
            pitch_ceiling=float(max_f0),
        )
    
    f0_raw = np.array(pitch.selected_array["frequency"], dtype=float)
    f0_values = np.where(f0_raw <= 0.0, np.nan, f0_raw)
    return f0_values
