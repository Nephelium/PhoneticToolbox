import numpy as np
from scipy import signal
import warnings
from typing import Tuple, List, Optional

def find_gci_goi_peak_min_criterion(
    egg_segment: np.ndarray, 
    fs: int, 
    min_f0: float = 50, 
    max_f0: float = 500, 
    criterion_level: float = 0.25,
    peak_prominence: float = 0.01, 
    valley_prominence: float = 0.01,
    use_local_prominence: bool = False, 
    local_window_s: float = 0.2, 
    local_hop_s: float = 0.1, 
    min_auto_prom: float = 0.01,
    gci_method: str = "slope", 
    goi_method: str = "slope", 
    cancel_event = None
) -> Tuple[List[float], List[float], List[float]]:
    """
    基于波峰、特定规则的前置波谷（用于GCI）、显著后置波谷（用于GOI）以及阈值水平查找GCI和GOI。
    同时返回检测到的所有波峰的时间点。
    GCI阈值固定为前置波谷到波峰幅度的0.25处。
    GOI阈值使用传入的criterion_level。
    使用线性插值进行阈值交叉点定位。
    包含用于峰谷查找的显著度参数。

    Args:
        egg_segment (np.ndarray): 输入的 EGG 信号段。
        fs (int): 采样率 (Hz)。
        min_f0 (float): 预期的最低基频 (Hz)，用于峰谷查找的距离约束。
        max_f0 (float): 预期的最高基频 (Hz)，用于峰谷查找的距离约束。
        criterion_level (float): 用于计算 GOI 阈值的幅度比例 (0 到 1)。
        peak_prominence (float): find_peaks 中用于波峰检测的最小显著度。
        valley_prominence (float): find_peaks 中用于波谷检测的最小显著度。
        use_local_prominence (bool): 是否使用局部自适应显著度。
        local_window_s (float): 局部窗口大小（秒）。
        local_hop_s (float): 局部窗口步长（秒）。
        min_auto_prom (float): 自动显著度下限。
        gci_method (str): GCI 检测方法 ("slope" 或 "scale").
        goi_method (str): GOI 检测方法 ("slope" 或 "scale").
        cancel_event: threading.Event 对象，用于取消操作。

    Returns:
        tuple: (gci_times_s, goi_times_s, peak_times_s)
            - gci_times_s (list): GCI 时间点列表 (秒)。
            - goi_times_s (list): GOI 时间点列表 (秒)。
            - peak_times_s (list): 检测到的所有波峰的时间点列表 (秒)。
              如果未找到波峰，则为空列表。
    """
    if egg_segment is None or len(egg_segment) < 2:
        return [], [], [] # Return three empty lists

    peak_times_s = []
    try:
        with warnings.catch_warnings():
            peak_min_dist = max(1, int(fs / max_f0 * 0.5))
            if use_local_prominence:
                N = len(egg_segment)
                win = max(1, int(local_window_s * fs))
                hop = max(1, int(local_hop_s * fs))
                peaks_all = []
                for start in range(0, N, hop):
                    end = min(N, start + win)
                    if end - start < 2:
                        continue
                    seg = egg_segment[start:end]
                    peak_amp = float(np.max(np.abs(seg))) if len(seg) > 0 else 0.0
                    prom = max(min_auto_prom, 0.6 * peak_amp)
                    p, _ = signal.find_peaks(seg, distance=peak_min_dist, prominence=prom)
                    if len(p) > 0:
                        peaks_all.extend((p + start).tolist())
                if len(peaks_all) > 0:
                    peaks_all = np.array(sorted(peaks_all))
                    if len(peaks_all) > 1:
                        dedup = [int(peaks_all[0])]
                        for idx in peaks_all[1:]:
                            if (idx - dedup[-1]) > 1:
                                dedup.append(int(idx))
                        peaks = np.array(dedup)
                    else:
                        peaks = peaks_all
                else:
                    peaks = np.array([])
                valleys, _ = signal.find_peaks(-egg_segment, distance=peak_min_dist, prominence=valley_prominence)
            else:
                peaks, _ = signal.find_peaks(egg_segment, distance=peak_min_dist, prominence=peak_prominence)
                valleys, _ = signal.find_peaks(-egg_segment, distance=peak_min_dist, prominence=valley_prominence)
    except Exception as e:
        print(f"Error during peak/valley finding: {e}")
        return [], [], []
    if len(peaks) < 1:
        # print(f"警告: 未找到足够的波峰")
        return [], [], []
    else:
        peak_times_s = (peaks / fs).tolist()

    if len(valleys) < 1 and len(peaks) > 0:
         # print(f"警告: 未找到显著波谷 (valley_prom={valley_prominence:.3f})，将尝试使用绝对最小值进行GOI计算。")
         pass

    # Ensure valleys are sorted (even if empty)
    valleys = np.sort(valleys)

    gci_indices_final = []
    goi_indices_final = []

    # --- 2. 遍历每个波峰，计算 GCI 和 GOI ---
    deriv_all = np.diff(egg_segment)
    N = len(egg_segment)
    for i in range(len(peaks)):
        if cancel_event is not None and getattr(cancel_event, "is_set", None) is not None and cancel_event.is_set():
            return [], [], []
        current_peak_idx = int(peaks[i])
        left_valleys = valleys[valleys < current_peak_idx] if len(valleys) > 0 else np.array([])
        right_valleys = valleys[valleys > current_peak_idx] if len(valleys) > 0 else np.array([])

        if gci_method == "slope":
            left_start = int(left_valleys[-1]) if len(left_valleys) > 0 else max(0, current_peak_idx - int(0.005 * fs))
            left_end = max(0, current_peak_idx)
            if left_end - left_start > 1:
                dseg = deriv_all[left_start:left_end]
                if len(dseg) > 0:
                    j = int(np.argmax(dseg))
                    gci_idx = float(left_start + j)
                    gci_indices_final.append(gci_idx)
        else:
            if len(left_valleys) > 0:
                lv = int(left_valleys[-1])
                rising = egg_segment[lv:current_peak_idx + 1]
                if len(rising) > 1:
                    vval = egg_segment[lv]
                    pval = egg_segment[current_peak_idx]
                    thresh = vval + 0.25 * (pval - vval)
                    crossings = np.where(np.diff(rising >= thresh) > 0)[0]
                    if len(crossings) > 0:
                        ib = crossings[0]
                        ia = ib + 1
                        if ia < len(rising):
                            vb = rising[ib]
                            va = rising[ia]
                            if abs(va - vb) > 1e-12:
                                frac = max(0.0, min(1.0, (thresh - vb) / (va - vb)))
                                gci_indices_final.append(float(lv + ib + frac))
                            else:
                                gci_indices_final.append(float(lv + ia))

        if goi_method == "slope":
            right_end = int(right_valleys[0]) if len(right_valleys) > 0 else min(N - 1, current_peak_idx + int(0.005 * fs))
            right_start = current_peak_idx
            if right_end - right_start > 1:
                dseg = np.abs(deriv_all[right_start:right_end])
                if len(dseg) > 0:
                    j = int(np.argmax(dseg))
                    goi_indices_final.append(float(right_start + j))
        else:
            if len(right_valleys) > 0:
                rv = int(right_valleys[0])
                falling = egg_segment[current_peak_idx:rv + 1]
                if len(falling) > 1:
                    vval = egg_segment[rv]
                    pval = egg_segment[current_peak_idx]
                    thresh = vval + 0.25 * (pval - vval)
                    crossings = np.where(np.diff(falling < thresh) > 0)[0]
                    if len(crossings) > 0:
                        ib = crossings[0]
                        ia = ib + 1
                        if ia < len(falling):
                            vb = falling[ib]
                            va = falling[ia]
                            if abs(va - vb) > 1e-12:
                                frac = max(0.0, min(1.0, (thresh - vb) / (va - vb)))
                                goi_indices_final.append(float(current_peak_idx + ib + frac))
                            else:
                                goi_indices_final.append(float(current_peak_idx + ia))

    # --- 3. 清理结果并转换为时间 ---
    gci_indices_final_unique = np.unique(gci_indices_final)
    goi_indices_final_unique = np.unique(goi_indices_final)

    gci_times_s = (gci_indices_final_unique / fs).tolist()
    goi_times_s = (goi_indices_final_unique / fs).tolist()

    return gci_times_s, goi_times_s, peak_times_s

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
