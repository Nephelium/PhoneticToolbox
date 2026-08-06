import numpy as np
from pathlib import Path

from phonetic_toolbox.models.acoustic_models import PitchTrack

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
    return compute_praat_f0_track(
        wav_path=wav_path,
        frameshift_ms=frameshift_ms,
        min_f0=min_f0,
        max_f0=max_f0,
        method=method,
    ).values


def compute_praat_f0_track(
    wav_path: Path,
    frameshift_ms: float,
    min_f0: float,
    max_f0: float,
    method: str = "cc",
) -> PitchTrack:
    """Compute F0 and preserve Praat's actual frame times."""
    if parselmouth is None:
        raise ImportError("praat-parselmouth 未安装")

    snd = parselmouth.Sound(str(wav_path))
    time_step = frameshift_ms / 1000.0
    method_name = (method or "cc").lower()

    if method_name == "cc":
        pitch = snd.to_pitch_cc(
            time_step=time_step,
            pitch_floor=float(min_f0),
            pitch_ceiling=float(max_f0),
        )
    elif method_name == "ac":
        pitch = snd.to_pitch(
            time_step=time_step,
            pitch_floor=float(min_f0),
            pitch_ceiling=float(max_f0),
        )
    else:
        raise ValueError(f"不支持的 Praat F0 方法: {method}")

    f0_raw = np.asarray(pitch.selected_array["frequency"], dtype=float)
    times = np.asarray(pitch.xs(), dtype=float)
    if times.shape != f0_raw.shape:
        raise ValueError("Praat F0 时间轴与数值长度不一致")
    values = np.where(f0_raw <= 0.0, np.nan, f0_raw)
    return PitchTrack(times=times, values=values)
