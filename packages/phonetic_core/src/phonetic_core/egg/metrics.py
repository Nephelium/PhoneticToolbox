"""M03 local migration of v2 EGG behavior. Source: PENDING-EGG; provenance unresolved.
See NOTICE.txt. Pure numerical compatibility layer, no file/device/GUI access.
"""
import numpy as np
from typing import Tuple, List

def calculate_cq_sq(
    gci_times_all_s: List[float],
    goi_times_all_s: List[float],
    peak_times_all_s: List[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据每个 GCI 事件、紧随其后的 GOI、下一个 GCI 以及相应的接触阶段峰值，
    计算接触商 (CQ) 和速度商 (SQ)。不使用窗口化。

    Args:
        gci_times_all_s (list): 所有检测到的 GCI 时间列表 (秒)。必须已排序。
        goi_times_all_s (list): 所有检测到的 GOI 时间列表 (秒)。必须已排序。
        peak_times_all_s (list): 所有检测到的 EGG 波峰时间列表 (秒)。必须已排序。

    Returns:
        tuple: (times, cq_values, sq_values)
            - times (np.ndarray): 计算了 CQ/SQ (或尝试计算) 的 GCI 时间数组。
            - cq_values (np.ndarray): 对应的 CQ 值数组。
            - sq_values (np.ndarray): 对应的 SQ 值数组。
    """
    # 1. 输入验证和准备
    if gci_times_all_s is None or len(gci_times_all_s) < 2:
        return np.array([]), np.array([]), np.array([])
    if goi_times_all_s is None or len(goi_times_all_s) == 0:
        num_gcis_to_try = len(gci_times_all_s) - 1
        times_only = np.array([gci_times_all_s[k] for k in range(num_gcis_to_try)])
        return times_only, np.full(num_gcis_to_try, np.nan), np.full(num_gcis_to_try, np.nan)
    if peak_times_all_s is None or len(peak_times_all_s) == 0:
        num_gcis_to_try = len(gci_times_all_s) - 1
        times_only = np.array([gci_times_all_s[k] for k in range(num_gcis_to_try)])
        return times_only, np.full(num_gcis_to_try, np.nan), np.full(num_gcis_to_try, np.nan)

    gci_times = np.sort(np.array(gci_times_all_s, dtype=float))
    goi_times = np.sort(np.array(goi_times_all_s, dtype=float))
    peak_times = np.sort(np.array(peak_times_all_s, dtype=float))

    num_cycles = len(gci_times) - 1
    times_out = gci_times[:-1].copy()
    cq_values_out = np.full(num_cycles, np.nan, dtype=float)
    sq_values_out = np.full(num_cycles, np.nan, dtype=float)

    goi_i = 0
    peak_i = 0
    num_goi = len(goi_times)
    num_peak = len(peak_times)

    for k in range(num_cycles):
        g0 = float(gci_times[k])
        g1 = float(gci_times[k + 1])
        period_s = g1 - g0
        if period_s <= 1e-9:
            continue

        while goi_i < num_goi and goi_times[goi_i] <= g0:
            goi_i += 1
        if goi_i >= num_goi:
            break
        goi_k = float(goi_times[goi_i])
        if not (g0 < goi_k < g1):
            continue

        contact_duration_s = goi_k - g0
        if not (0.0 < contact_duration_s < period_s):
            continue

        cq = contact_duration_s / period_s
        if 0.05 < cq < 0.95:
            cq_values_out[k] = cq

        while peak_i < num_peak and peak_times[peak_i] <= g0:
            peak_i += 1

        if peak_i >= num_peak:
            continue

        peak_j = peak_i
        if peak_times[peak_j] >= goi_k:
            continue

        peak_time = float(peak_times[peak_j])
        peak_j += 1
        if peak_j < num_peak and peak_times[peak_j] < goi_k:
            peak_i = peak_j
            continue

        peak_i = peak_j
        contacting_duration_s = peak_time - g0
        decontacting_duration_s = goi_k - peak_time
        if contacting_duration_s >= 0.0 and decontacting_duration_s >= 0.0 and contact_duration_s > 1e-9:
            sq_values_out[k] = (decontacting_duration_s - contacting_duration_s) / contact_duration_s

    return times_out, cq_values_out, sq_values_out
