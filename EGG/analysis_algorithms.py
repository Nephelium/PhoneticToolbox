import numpy as np
from scipy import signal # For filtering, spectrogram, peak finding
import pywt
import warnings # Import warnings module

# --- Filtering Functions (Unchanged) ---
def apply_highpass_filter(data, cutoff_freq, fs, order=4):
    nyq = 0.5 * fs
    cutoff = cutoff_freq / nyq
    cutoff = max(cutoff, 1e-6)
    cutoff = min(cutoff, 1 - 1e-6)
    if cutoff >= 1.0:
        print(f"Warning: High-pass cutoff frequency ({cutoff_freq} Hz) is too high relative to Nyquist ({nyq} Hz). Skipping filtering.")
        return data
    try:
        b, a = signal.butter(order, cutoff, btype='high')
        y = signal.filtfilt(b, a, data)
        print(f"Applied high-pass filter at {cutoff_freq} Hz.")
        return y
    except ValueError as e:
        print(f"Error during high-pass filtering: {e}. Cutoff: {cutoff}. Returning unfiltered data.")
        return data

def apply_lowpass_filter(data, cutoff_freq, fs, order=4):
    nyq = 0.5 * fs
    cutoff = cutoff_freq / nyq
    cutoff = max(cutoff, 1e-6)
    cutoff = min(cutoff, 1 - 1e-6)
    if cutoff <= 0.0:
        print(f"Warning: Low-pass cutoff frequency ({cutoff_freq} Hz) is too low relative to Nyquist ({nyq} Hz). Skipping filtering.")
        return data
    try:
        b, a = signal.butter(order, cutoff, btype='low')
        y = signal.filtfilt(b, a, data)
        print(f"Applied low-pass filter at {cutoff_freq} Hz.")
        return y
    except ValueError as e:
        print(f"Error during low-pass filtering: {e}. Cutoff: {cutoff}. Returning unfiltered data.")
        return data

def apply_wavelet_denoising(data, wavelet='db4', level=4, mode='soft'):
    if data is None or len(data) < 2:
        return data
    try:
        coeffs = pywt.wavedec(data, wavelet, level=level)
        detail_coeffs = [c for c in coeffs[1:] if c is not None and len(c) > 0]
        if not detail_coeffs:
            print("Warning: No valid detail coefficients found for noise estimation. Skipping thresholding.")
            threshold = 0
        else:
            last_detail_coeffs = detail_coeffs[-1]
            sigma = np.median(np.abs(last_detail_coeffs - np.median(last_detail_coeffs))) / 0.6745
            if sigma < 1e-9: sigma = np.std(last_detail_coeffs)
            if sigma < 1e-9: sigma = 1e-9
            threshold = sigma * np.sqrt(2 * np.log(len(data))) if len(data) > 1 else 0

        coeffs_thresh = [coeffs[0]]
        for i in range(1, len(coeffs)):
             if coeffs[i] is not None and len(coeffs[i]) > 0:
                coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode=mode))
             else:
                coeffs_thresh.append(coeffs[i])

        denoised_data = pywt.waverec(coeffs_thresh, wavelet)
        original_len = len(data)
        current_len = len(denoised_data)
        if current_len > original_len:
            denoised_data = denoised_data[:original_len]
        elif current_len < original_len:
            padding = original_len - current_len
            denoised_data = np.pad(denoised_data, (0, padding), 'edge')

        print(f"Applied wavelet denoising (wavelet={wavelet}, level={level}, threshold={threshold:.4f}).")
        return denoised_data
    except Exception as e:
        print(f"Error during wavelet denoising: {e}. Returning original data.")
        return data

# --- EGG Analysis Functions ---
def find_gci_goi_peak_min_criterion(egg_segment, fs, min_f0=50, max_f0=500, criterion_level=0.25,
                                    peak_prominence=0.01, valley_prominence=0.01,
                                    use_local_prominence=False, local_window_s=0.2, local_hop_s=0.1, min_auto_prom=0.01,
                                    gci_method="slope", goi_method="slope"):
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
        print(f"警告: 未找到足够的波峰")
        return [], [], []
    else:
        peak_times_s = (peaks / fs).tolist()

    if len(valleys) < 1 and len(peaks) > 0:
         print(f"警告: 未找到显著波谷 (valley_prom={valley_prominence:.3f})，将尝试使用绝对最小值进行GOI计算。")
         # valleys remains empty, GOI logic will handle it

    # Ensure valleys are sorted (even if empty)
    valleys = np.sort(valleys)

    gci_indices_final = []
    goi_indices_final = []

    # --- 2. 遍历每个波峰，计算 GCI 和 GOI ---
    deriv_all = np.diff(egg_segment)
    N = len(egg_segment)
    for i in range(len(peaks)):
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

    # Return GCI times, GOI times, and the times of ALL detected peaks
    return gci_times_s, goi_times_s, peak_times_s

# --- calculate_cq_sq function remains unchanged from the previous correct version ---
# (No need to include it again here unless you want the full file content)


# --- REVISED: CQ calculation per GCI event (No Windowing) ---
def calculate_cq_sq(gci_times_all_s, goi_times_all_s, peak_times_all_s):
    """
    根据每个 GCI 事件、紧随其后的 GOI、下一个 GCI 以及相应的接触阶段峰值，
    计算接触商 (CQ) 和速度商 (SQ)。不使用窗口化。

    Args:
        gci_times_all_s (list or np.ndarray): 所有检测到的 GCI 时间列表 (秒)。必须已排序。
        goi_times_all_s (list or np.ndarray): 所有检测到的 GOI 时间列表 (秒)。必须已排序。
        peak_times_all_s (list or np.ndarray): 所有检测到的 EGG 波峰时间列表 (秒)。必须已排序。
                                              (由 find_gci_goi_peak_min_criterion 返回)

    Returns:
        tuple: (times, cq_values, sq_values)
            - times (np.ndarray): 计算了 CQ/SQ (或尝试计算) 的 GCI 时间数组。对应 gci_k_time。
            - cq_values (np.ndarray): 对应的 CQ 值数组。如果无法计算或未通过滤波器，则包含 NaN。
            - sq_values (np.ndarray): 对应的 SQ 值数组。如果无法计算或未找到唯一峰值，则包含 NaN。
    """
    # 1. 输入验证和准备
    if gci_times_all_s is None or len(gci_times_all_s) < 2:
        print("警告: 需要至少 2 个 GCI 来计算任何 CQ/SQ 周期。")
        return np.array([]), np.array([]), np.array([])
    if goi_times_all_s is None or len(goi_times_all_s) == 0:
        print("警告: 未提供 GOI，无法计算 CQ/SQ。")
        num_gcis_to_try = len(gci_times_all_s) - 1
        times_only = np.array([gci_times_all_s[k] for k in range(num_gcis_to_try)])
        return times_only, np.full(num_gcis_to_try, np.nan), np.full(num_gcis_to_try, np.nan)
    if peak_times_all_s is None or len(peak_times_all_s) == 0:
        print("警告: 未提供波峰时间，无法计算 SQ。")
        # Fallback to calculating only CQ if possible
        num_gcis_to_try = len(gci_times_all_s) - 1
        times_only = np.array([gci_times_all_s[k] for k in range(num_gcis_to_try)])
        # We need a separate CQ-only function or logic here if we want to return partial results
        # For simplicity now, return all NaNs if peaks are missing for SQ.
        # cq_t, cq_v = calculate_cq_only(gci_times_all_s, goi_times_all_s) # Hypothetical CQ-only function
        # return cq_t, cq_v, np.full(len(cq_t), np.nan)
        return times_only, np.full(num_gcis_to_try, np.nan), np.full(num_gcis_to_try, np.nan)


    # Ensure inputs are sorted numpy arrays
    gci_times = np.sort(np.array(gci_times_all_s))
    goi_times = np.sort(np.array(goi_times_all_s))
    peak_times = np.sort(np.array(peak_times_all_s))

    num_gcis = len(gci_times)
    times_out = []
    cq_values_out = []
    sq_values_out = [] # 用于存储 SQ 值的新列表

    # 2. 遍历每个 GCI (除了最后一个)
    for k in range(num_gcis - 1):
        gci_k_time = gci_times[k]
        gci_k_plus_1_time = gci_times[k+1]
        current_cq = np.nan  # 默认为 NaN
        current_sq = np.nan  # 默认为 NaN

        # 3. 计算周期
        period_s = gci_k_plus_1_time - gci_k_time

        # 基本的周期有效性检查
        if period_s > 1e-9: # 避免除以零或无意义的周期

            # 4. 查找 *严格位于* gci_k 和 gci_k+1 之间的第一个 GOI
            relevant_gois_indices = np.where((goi_times > gci_k_time) & (goi_times < gci_k_plus_1_time))[0]

            if len(relevant_gois_indices) > 0:
                # 如果找到一个或多个 GOI，取第一个
                first_goi_index = relevant_gois_indices[0]
                goi_k_time = goi_times[first_goi_index]

                # 5. 计算接触时长
                contact_duration_s = goi_k_time - gci_k_time

                # 6. 检查接触时长有效性并计算 CQ
                if 0 < contact_duration_s < period_s:
                    cq = contact_duration_s / period_s

                    # 7. 应用 CQ 滤波器
                    if 0.05 < cq < 0.95:
                        current_cq = cq # 分配有效的 CQ
                    # else: CQ 计算出来但未通过滤波器，保持 NaN

                    # --- 8. 计算 SQ ---
                    # 查找 *严格位于* gci_k 和 goi_k 之间的 *唯一* 峰值
                    peaks_in_contact_phase = peak_times[(peak_times > gci_k_time) & (peak_times < goi_k_time)]

                    if len(peaks_in_contact_phase) == 1:
                        # 找到了唯一的峰值
                        peak_time_k = peaks_in_contact_phase[0]

                        contacting_duration_s = peak_time_k - gci_k_time
                        decontacting_duration_s = goi_k_time - peak_time_k
                        contact_duration_s = goi_k_time - gci_k_time

                        # 确保关闭段和分离段时长有效 (理论上应该总是有效，如果峰值在中间)
                        if contacting_duration_s >= 0 and decontacting_duration_s >= 0:
                             # 检查有效除法 (接触时长)s
                             if contact_duration_s > 1e-9:
                                 sq = (decontacting_duration_s - contacting_duration_s) / contact_duration_s
                                 # 可选：在此处应用 SQ 有效性滤波器
                                 # 例如: if -1.0 <= sq <= 1.0:
                                 current_sq = sq
                             # else: 接触时长过小，SQ 保持 NaN
                        # else: 峰值时间计算或 GCI/GOI 顺序有问题，SQ 保持 NaN

                    elif len(peaks_in_contact_phase) == 0:
                         print(f"警告: 在 GCI {gci_k_time:.4f} 和 GOI {goi_k_time:.4f} 之间未找到波峰，无法计算 SQ。")
                         # current_sq 保持 NaN
                    else:
                         print(f"警告: 在 GCI {gci_k_time:.4f} 和 GOI {goi_k_time:.4f} 之间找到多个波峰 ({len(peaks_in_contact_phase)})，无法确定唯一峰值计算 SQ。")
                         # current_sq 保持 NaN

                # else: 接触时长无效，CQ/SQ 保持 NaN
            # else: 在区间内未找到 GOI，CQ/SQ 保持 NaN
        # else: 周期无效，CQ/SQ 保持 NaN

        # 9. 存储此 GCI 的结果
        times_out.append(gci_k_time)
        cq_values_out.append(current_cq)
        sq_values_out.append(current_sq) # 存储 SQ 值

    return np.array(times_out), np.array(cq_values_out), np.array(sq_values_out)
