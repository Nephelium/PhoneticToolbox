import numpy as np
import scipy.signal
from scipy.linalg import solve_toeplitz


def compute_lpc_spectrum(
    audio: np.ndarray,
    fs: int,
    order: int,
    n_freqs: int = 1024,
    pre_emphasis: float = 0.97,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(audio, dtype=np.float64).reshape(-1)
    if y.size <= max(order + 1, 2):
        raise ValueError("音频长度不足，无法计算当前阶数的 LPC。")

    y_pre = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    y_win = y_pre * np.hamming(y_pre.size)
    autocorr = np.correlate(y_win, y_win, mode="full")
    autocorr = autocorr[autocorr.size // 2:]
    if autocorr.size < order + 1:
        raise ValueError("自相关序列长度不足，无法完成 LPC 计算。")

    r0 = autocorr[:order]
    rhs = autocorr[1 : order + 1]
    try:
        coeff = solve_toeplitz((r0, r0), rhs)
        a = np.concatenate(([1.0], -coeff))
    except Exception as exc:
        raise ValueError(f"LPC 求解失败: {exc}") from exc

    freq_hz, response = scipy.signal.freqz(1.0, a, worN=n_freqs, fs=fs)
    magnitude_db = 20.0 * np.log10(np.abs(response) + 1e-10)
    return freq_hz, magnitude_db
