from __future__ import annotations
import numpy as np

# Original V2 implementation; estimate only, not a physiological alignment truth.
def estimate_lip_audio_offset_seconds(
    audio_samples: np.ndarray,
    sample_rate: int,
    lip_times: np.ndarray,
    lip_open: np.ndarray,
    search_seconds: float = 2.0,
) -> float:
    if audio_samples.size < 8 or lip_times.size < 8 or lip_open.size < 8 or sample_rate <= 0:
        return 0.0
    audio = np.abs(audio_samples.astype(np.float64))
    win = max(1, int(round(0.01 * sample_rate)))
    env = np.convolve(audio, np.ones(win, dtype=np.float64) / float(win), mode="same")
    audio_t = np.arange(len(env), dtype=np.float64) / float(sample_rate)

    lip_t = np.array(lip_times, dtype=np.float64)
    lip_y = np.array(lip_open, dtype=np.float64)
    n = min(lip_t.size, lip_y.size)
    lip_t = lip_t[:n]
    lip_y = lip_y[:n]
    if n < 8:
        return 0.0

    t0 = max(float(audio_t[0]), float(lip_t[0]) - search_seconds)
    t1 = min(float(audio_t[-1]), float(lip_t[-1]) + search_seconds)
    if t1 - t0 <= 1.0:
        return 0.0
    grid_fs = 200.0
    grid = np.arange(t0, t1, 1.0 / grid_fs, dtype=np.float64)
    if grid.size < 100:
        return 0.0

    audio_grid = np.interp(grid, audio_t, env)
    audio_grid = (audio_grid - np.mean(audio_grid)) / (np.std(audio_grid) + 1e-12)

    best_offset = 0.0
    best_score = -1e12
    for off in np.arange(-search_seconds, search_seconds + 0.0001, 0.005):
        shifted = np.interp(grid, lip_t + off, lip_y, left=np.nan, right=np.nan)
        valid = np.isfinite(shifted)
        if np.count_nonzero(valid) < 100:
            continue
        lv = shifted[valid]
        lv = (lv - np.mean(lv)) / (np.std(lv) + 1e-12)
        av = audio_grid[valid]
        score = float(np.mean(av * lv))
        if score > best_score:
            best_score = score
            best_offset = float(off)
    return best_offset
