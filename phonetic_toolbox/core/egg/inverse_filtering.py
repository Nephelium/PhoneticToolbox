import numpy as np
from scipy import signal
from scipy.linalg import solve_toeplitz
import warnings
from typing import Optional

def autocorrelation(y: np.ndarray, order: int) -> np.ndarray:
    """计算自相关函数"""
    r = np.correlate(y, y, mode='full')
    needed_len = 2 * order + 1
    if len(r) < needed_len:
        padding_needed = needed_len - len(r)
        r = np.pad(r, (padding_needed // 2, padding_needed - padding_needed // 2))
        print(f"警告: 自相关序列短于预期 ({len(r)} < {needed_len})，已填充。LPC 可能不准确。")

    midpoint = len(r) // 2
    if midpoint + order + 1 > len(r):
        print(f"警告: 调整自相关滞后点数，因为信号长度相对于阶数过短。")
        return r[midpoint:]
    return r[midpoint : midpoint + order + 1]

def solve_lpc_autocorr(r: np.ndarray, order: int) -> Optional[np.ndarray]:
    """
    使用自相关函数通过解 Yule-Walker 方程 (Toeplitz 矩阵) 来计算 LPC 系数。
    返回系数 a，形式为 [1, -a1, -a2, ...]。
    """
    if r is None or len(r) <= order:
        print(f"错误: 自相关序列为 None 或过短 ({len(r) if r is not None else 'None'})，无法计算 LPC 阶数 {order}。")
        return None

    if r[0] == 0:
        print(f"警告: r[0] 为零，LPC 计算无意义或不稳定。返回单位滤波器。")
        return np.concatenate(([1], np.zeros(order)))

    if len(r) < order + 1:
         print(f"错误: 自相关滞后点数不足 ({len(r)})，无法计算 LPC 阶数 {order}。")
         return None

    try:
        if np.all(r[1:order+1] == 0) and r[0] != 0:
            print(f"警告: 自相关序列异常 (阶数 {order})，可能导致滤波器不稳定。")
            pass

        a_coeffs = solve_toeplitz(r[:order], r[1:order+1])
        a = np.concatenate(([1], -a_coeffs))

        if not np.all(np.isfinite(a)):
            print(f"错误: 计算得到的 LPC 系数包含非有限值 (阶数 {order})。")
            return None
        return a
    except np.linalg.LinAlgError as e:
        print(f"错误: 求解 Toeplitz 系统时出错 (阶数 {order}，可能是奇异矩阵): {e}")
        return None
    except IndexError as e:
        print(f"错误: LPC 计算期间发生索引错误 (阶数 {order}，可能由于 r 过短): {e}")
        return None

def apply_simplified_cp_inverse_filtering(
    audio_signal: np.ndarray, 
    fs: int, 
    gci_times_relative_to_roi_start: np.ndarray, 
    lp_order: Optional[int] = None, 
    closed_phase_duration_ms: float = 3.0, 
    min_segments_for_avg: int = 3, 
    tilt_order: int = 1, 
    pre_emphasis_alpha: float = 0.97
) -> Optional[np.ndarray]:
    """
    应用简化的基于闭合相的逆滤波 - ARMA + Pre-emphasis 版本。
    对信号进行预加重，估计声道和倾斜滤波器，应用 ARMA 逆滤波，然后去加重。

    Args:
        audio_signal (np.ndarray): 输入音频信号段 (ROI)。
        fs (int): 采样率。
        gci_times_relative_to_roi_start (np.ndarray): ROI 内的 GCI 时间点 (秒), 相对于 ROI 开始。
        lp_order (int, optional): 声道 LPC 阶数。如果为 None，则自动计算 (使用稍高阶)。
        closed_phase_duration_ms (float): GCI 后用于估计 LPC 的窗口时长 (毫秒)。
        min_segments_for_avg (int): 计算平均声道 LPC 所需的最少段数。
        tilt_order (int): 频谱倾斜 LPC 模型的阶数 (通常为 1 或 2)。
        pre_emphasis_alpha (float): 预加重系数 (通常为 0.95-0.98)。

    Returns:
        np.ndarray: 逆滤波后的信号 (声门流导数估计，尝试保留谱倾斜)。
        None: 如果无法执行滤波。
    """
    if lp_order is None:
        lp_order = int(fs / 1000) + 6
        print(f"自动设置声道 LPC 阶数: {lp_order}")

    if len(audio_signal) < lp_order + 1 or len(audio_signal) < tilt_order + 1:
        print("警告: 音频信号对于指定的 LPC 阶数过短。")
        return None

    if gci_times_relative_to_roi_start is None or len(gci_times_relative_to_roi_start) == 0:
        print("警告: 未提供或在 ROI 内未找到 GCI 时间点。无法执行基于 CP 的滤波。")
        return None

    # --- 1. Pre-emphasis ---
    audio_preemphasized = signal.lfilter([1, -pre_emphasis_alpha], [1], audio_signal)

    # --- 2. Estimate Tract Filter A_tract(z) using Closed Phase Averaging on Pre-emphasized signal ---
    all_lpc_coeffs = []
    closed_phase_samples = int(closed_phase_duration_ms / 1000.0 * fs)
    gci_samples_in_roi = (gci_times_relative_to_roi_start * fs).astype(int)

    for gci_sample in gci_samples_in_roi:
        start_idx = gci_sample + 1
        end_idx = start_idx + closed_phase_samples
        end_idx = min(end_idx, len(audio_preemphasized))
        start_idx = max(0, start_idx)
        start_idx = min(start_idx, end_idx)

        if end_idx > start_idx:
            segment = audio_preemphasized[start_idx:end_idx]
            if len(segment) >= lp_order + 1:
                try:
                    r_segment = autocorrelation(segment, lp_order)
                    if r_segment is not None and len(r_segment) > lp_order:
                        a_segment = solve_lpc_autocorr(r_segment, lp_order)
                        if a_segment is not None:
                            all_lpc_coeffs.append(a_segment)
                except Exception as e:
                    pass

    if len(all_lpc_coeffs) < min_segments_for_avg:
        print(f"警告: 仅找到 {len(all_lpc_coeffs)} 个有效的声道 LPC 估计 (最少需要: {min_segments_for_avg})。无法执行可靠的滤波。")
        return None

    lpc_matrix = np.array(all_lpc_coeffs)
    a_tract = np.mean(lpc_matrix, axis=0)
    if not np.all(np.isfinite(a_tract)):
        print("错误: 平均声道 LPC 系数包含非有限值。")
        return None

    # --- 3. Estimate Tilt Filter A_tilt(z) using Low-Order LPC on whole PRE-EMPHASIZED ROI ---
    a_tilt = None
    try:
        if len(audio_preemphasized) >= tilt_order + 1:
            r_roi = autocorrelation(audio_preemphasized, tilt_order)
            if r_roi is not None and len(r_roi) > tilt_order:
                a_tilt = solve_lpc_autocorr(r_roi, tilt_order)
                if a_tilt is None:
                    a_tilt = np.array([1.0])
            else:
                a_tilt = np.array([1.0])
        else:
             a_tilt = np.array([1.0])
    except Exception as e:
        a_tilt = np.array([1.0])

    if a_tilt is None: a_tilt = np.array([1.0])

    # --- 4. Apply Combined ARMA Filter to Pre-emphasized signal ---
    filtered_preemphasized = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            filtered_preemphasized = signal.lfilter(a_tract, a_tilt, audio_preemphasized)

        if not np.all(np.isfinite(filtered_preemphasized)):
            # Fallback to FIR
            filtered_preemphasized = signal.lfilter(a_tract, [1.0], audio_preemphasized)
            if not np.all(np.isfinite(filtered_preemphasized)):
                 return None

    except Exception as e:
        try:
            filtered_preemphasized = signal.lfilter(a_tract, [1.0], audio_preemphasized)
            if not np.all(np.isfinite(filtered_preemphasized)):
                 return None
        except Exception as e2:
             return None

    # --- 5. De-emphasis ---
    if filtered_preemphasized is not None:
        filtered_signal = signal.lfilter([1], [1, -pre_emphasis_alpha], filtered_preemphasized)
        return filtered_signal
    else:
        return None
