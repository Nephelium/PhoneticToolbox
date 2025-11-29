# inverse_filtering.py
import numpy as np
from scipy import signal
from scipy.linalg import solve_toeplitz
from scipy.fft import fft, fftfreq
from scipy.signal import windows
import matplotlib.pyplot as plt
import warnings

# --- autocorrelation 函数保持不变 ---
def autocorrelation(y, order):
    """计算自相关函数"""
    r = np.correlate(y, y, mode='full')
    # 确保 r 足够长以提取所需的阶数
    needed_len = 2 * order + 1
    if len(r) < needed_len:
        # 如果 correlate 结果太短，可能输入 y 就很短
        # 进行适当的填充以避免索引错误，但这可能不是理想情况
        padding_needed = needed_len - len(r)
        # 尝试对称填充，虽然对于 correlate 结果可能不完美
        r = np.pad(r, (padding_needed // 2, padding_needed - padding_needed // 2))
        print(f"警告: 自相关序列短于预期 ({len(r)} < {needed_len})，已填充。LPC 可能不准确。") # Chinese comment kept as requested

    midpoint = len(r) // 2
    # 再次检查，确保可以提取 mid 到 mid+order
    if midpoint + order + 1 > len(r):
        print(f"警告: 调整自相关滞后点数，因为信号长度相对于阶数过短。") # Chinese comment kept as requested
        # 如果无法提取足够的滞后点，返回尽可能多的点，但这可能导致后续错误
        return r[midpoint:] # 返回从中心点开始的所有可用点
    return r[midpoint : midpoint + order + 1]

# --- solve_lpc_autocorr 函数保持不变 ---
def solve_lpc_autocorr(r, order):
    """
    使用自相关函数通过解 Yule-Walker 方程 (Toeplitz 矩阵) 来计算 LPC 系数。
    返回系数 a，形式为 [1, -a1, -a2, ...]。
    """
    if r is None or len(r) <= order:
        print(f"错误: 自相关序列为 None 或过短 ({len(r) if r is not None else 'None'})，无法计算 LPC 阶数 {order}。") # Chinese comment kept as requested
        return None

    # 检查 r[0] 是否为零
    if r[0] == 0:
        # 如果 r[0] 为零，意味着信号能量为零（或接近零）
        print(f"警告: r[0] 为零，LPC 计算无意义或不稳定。返回单位滤波器。") # Chinese comment kept as requested
        return np.concatenate(([1], np.zeros(order)))

    # 检查是否有足够的滞后点
    if len(r) < order + 1:
         print(f"错误: 自相关滞后点数不足 ({len(r)})，无法计算 LPC 阶数 {order}。") # Chinese comment kept as requested
         return None

    try:
        # 检查潜在的奇异矩阵情况 (虽然 solve_toeplitz 通常能处理)
        if np.all(r[1:order+1] == 0) and r[0] != 0:
            print(f"警告: 自相关序列异常 (阶数 {order})，可能导致滤波器不稳定。") # Chinese comment kept as requested
            # 这种情况下，模型可能很简单，只返回 [1, 0, ..., 0]
            # return np.concatenate(([1], np.zeros(order)))

        # 解 Toeplitz 系统
        a_coeffs = solve_toeplitz(r[:order], r[1:order+1])
        a = np.concatenate(([1], -a_coeffs))

        # 检查结果是否有限
        if not np.all(np.isfinite(a)):
            print(f"错误: 计算得到的 LPC 系数包含非有限值 (阶数 {order})。") # Chinese comment kept as requested
            return None
        return a
    except np.linalg.LinAlgError as e:
        # 捕获线性代数错误，通常是奇异矩阵
        print(f"错误: 求解 Toeplitz 系统时出错 (阶数 {order}，可能是奇异矩阵): {e}") # Chinese comment kept as requested
        return None
    except IndexError as e:
        # 捕获索引错误，通常是因为 r 不够长
        print(f"错误: LPC 计算期间发生索引错误 (阶数 {order}，可能由于 r 过短): {e}") # Chinese comment kept as requested
        return None

# --- apply_simplified_cp_inverse_filtering 函数保持不变 ---
def apply_simplified_cp_inverse_filtering(audio_signal, fs, gci_times_relative_to_roi_start, lp_order=None, closed_phase_duration_ms=3.0, min_segments_for_avg=3, tilt_order=1, pre_emphasis_alpha=0.97):
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
        # --- Increased LPC Order Heuristic ---
        lp_order = int(fs / 1000) + 6 # Increased from +4 to +6
        print(f"自动设置声道 LPC 阶数: {lp_order}") # Chinese comment kept as requested

    if len(audio_signal) < lp_order + 1 or len(audio_signal) < tilt_order + 1:
        print("警告: 音频信号对于指定的 LPC 阶数过短。") # Chinese comment kept as requested
        return None

    if gci_times_relative_to_roi_start is None or len(gci_times_relative_to_roi_start) == 0:
        print("警告: 未提供或在 ROI 内未找到 GCI 时间点。无法执行基于 CP 的滤波。") # Chinese comment kept as requested
        return None

    # --- 1. Pre-emphasis ---
    print(f"应用预加重，alpha={pre_emphasis_alpha}") # Chinese comment kept as requested
    audio_preemphasized = signal.lfilter([1, -pre_emphasis_alpha], [1], audio_signal)

    # --- 2. Estimate Tract Filter A_tract(z) using Closed Phase Averaging on Pre-emphasized signal ---
    all_lpc_coeffs = []
    closed_phase_samples = int(closed_phase_duration_ms / 1000.0 * fs)
    gci_samples_in_roi = (gci_times_relative_to_roi_start * fs).astype(int)

    for gci_sample in gci_samples_in_roi:
        start_idx = gci_sample + 1 # 从 GCI 后一个样本开始
        end_idx = start_idx + closed_phase_samples
        # 确保索引在预加重信号的有效范围内
        end_idx = min(end_idx, len(audio_preemphasized))
        start_idx = max(0, start_idx)
        start_idx = min(start_idx, end_idx) # 确保 start 不超过 end

        if end_idx > start_idx:
            # 从 *预加重* 信号中提取片段
            segment = audio_preemphasized[start_idx:end_idx]
            if len(segment) >= lp_order + 1: # 确保片段足够长以计算 LPC
                try:
                    r_segment = autocorrelation(segment, lp_order)
                    if r_segment is not None and len(r_segment) > lp_order:
                        a_segment = solve_lpc_autocorr(r_segment, lp_order)
                        if a_segment is not None: # solve_lpc 已检查有限性
                            all_lpc_coeffs.append(a_segment)
                        # else: 错误信息在 solve_lpc 内部打印
                    # else: 错误信息在 autocorrelation 或长度检查中打印
                except Exception as e:
                    print(f"警告: 在 GCI ~{gci_sample/fs:.4f}s (相对) 处计算段 LPC 时发生意外错误: {e}。跳过此段。") # Chinese comment kept as requested

    if len(all_lpc_coeffs) < min_segments_for_avg:
        print(f"警告: 仅找到 {len(all_lpc_coeffs)} 个有效的声道 LPC 估计 (最少需要: {min_segments_for_avg})。无法执行可靠的滤波。") # Chinese comment kept as requested
        return None

    # 计算平均 LPC 系数
    lpc_matrix = np.array(all_lpc_coeffs)
    a_tract = np.mean(lpc_matrix, axis=0)
    if not np.all(np.isfinite(a_tract)):
        print("错误: 平均声道 LPC 系数包含非有限值。") # Chinese comment kept as requested
        return None
    print(f"从 {len(all_lpc_coeffs)} 个段中平均得到声道 LPC 系数 (阶数={lp_order})。") # Chinese comment kept as requested

    # --- 3. Estimate Tilt Filter A_tilt(z) using Low-Order LPC on whole PRE-EMPHASIZED ROI ---
    a_tilt = None
    try:
        if len(audio_preemphasized) >= tilt_order + 1:
            # 同样在 *预加重* 信号上估计倾斜
            r_roi = autocorrelation(audio_preemphasized, tilt_order)
            if r_roi is not None and len(r_roi) > tilt_order:
                a_tilt = solve_lpc_autocorr(r_roi, tilt_order)
                if a_tilt is None:
                    print(f"警告: 未能计算有效的低阶倾斜 LPC (阶数={tilt_order})。将回退到标准逆滤波。") # Chinese comment kept as requested
                    a_tilt = np.array([1.0]) # 回退为无倾斜补偿
                else:
                    print(f"计算得到倾斜 LPC 系数 (阶数={tilt_order})。") # Chinese comment kept as requested
            else:
                print(f"警告: 自相关失败或对于倾斜 LPC 过短 (阶数={tilt_order})。将回退。") # Chinese comment kept as requested
                a_tilt = np.array([1.0])
        else:
             print(f"警告: ROI 对于倾斜 LPC 过短 (阶数={tilt_order})。将回退。") # Chinese comment kept as requested
             a_tilt = np.array([1.0])
    except Exception as e:
        print(f"警告: 计算倾斜 LPC 时出错 (阶数={tilt_order}): {e}。将回退。") # Chinese comment kept as requested
        a_tilt = np.array([1.0])

    # 确保 a_tilt 是一个有效的数组
    if a_tilt is None: a_tilt = np.array([1.0])

    # --- 4. Apply Combined ARMA Filter to Pre-emphasized signal ---
    # 滤波器形式: H(z) = A_tract(z) / A_tilt(z)
    # 应用 lfilter(b, a, x) 对应 H(z) = B(z) / A(z)
    # 所以 b = a_tract, a = a_tilt
    print(f"应用 ARMA 滤波器: 分子阶数={len(a_tract)-1}, 分母阶数={len(a_tilt)-1}") # Chinese comment kept as requested
    filtered_preemphasized = None
    try:
        # 忽略可能的运行时警告 (例如，除以接近零的值)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            filtered_preemphasized = signal.lfilter(a_tract, a_tilt, audio_preemphasized)

        # 检查滤波结果是否包含 NaN 或 Inf
        if not np.all(np.isfinite(filtered_preemphasized)):
            print("错误: ARMA 滤波结果包含非有限值。请检查滤波器稳定性。") # Chinese comment kept as requested
            print("将回退到标准 FIR 逆滤波 (仅使用 a_tract)。") # Chinese comment kept as requested
            # 回退：在预加重信号上应用 FIR 滤波器
            filtered_preemphasized = signal.lfilter(a_tract, [1.0], audio_preemphasized)
            if not np.all(np.isfinite(filtered_preemphasized)):
                 print("错误: 回退的 FIR 滤波也产生了非有限值。") # Chinese comment kept as requested
                 return None # 放弃

    except Exception as e:
        print(f"错误: 应用 ARMA 逆滤波器时出错: {e}") # Chinese comment kept as requested
        # 尝试回退到 FIR 滤波
        try:
            print("由于 ARMA 错误，将回退到标准 FIR 逆滤波。") # Chinese comment kept as requested
            # 在预加重信号上应用 FIR 滤波器
            filtered_preemphasized = signal.lfilter(a_tract, [1.0], audio_preemphasized)
            if not np.all(np.isfinite(filtered_preemphasized)):
                 print("错误: 回退的 FIR 滤波也产生了非有限值。") # Chinese comment kept as requested
                 return None
        except Exception as e2:
             print(f"错误: 应用回退 FIR 滤波器时出错: {e2}") # Chinese comment kept as requested
             return None

    # --- 5. De-emphasis ---
    if filtered_preemphasized is not None:
        print(f"应用去加重，alpha={pre_emphasis_alpha}") # Chinese comment kept as requested
        # 去加重滤波器是 1 / (1 - alpha * z^-1) -> lfilter([1], [1, -alpha], ...)
        filtered_signal = signal.lfilter([1], [1, -pre_emphasis_alpha], filtered_preemphasized)
        print(f"已应用 ARMA 逆滤波和去加重。") # Chinese comment kept as requested
        return filtered_signal
    else:
        # 如果回退逻辑正确，这里不应该发生，但作为保险
        print("错误: 去加重之前的滤波后预加重信号为 None。") # Chinese comment kept as requested
        return None


# --- plot_inverse_filtering_results 函数 **已修改为英文** ---
def plot_inverse_filtering_results(original_audio, filtered_audio, egg_signal, fs, roi_start_s, roi_end_s, style_settings):
    """
    Plots inverse filtering results in a new figure (Modified: Zero-padding + Windowing + Twin Axes).

    Args:
        original_audio (np.ndarray): Original audio signal segment (ROI).
        filtered_audio (np.ndarray or None): Inverse filtered audio signal segment, or None.
        egg_signal (np.ndarray): Corresponding EGG signal segment (ROI).
        fs (int): Sampling rate.
        roi_start_s (float): Start time of the ROI (seconds).
        roi_end_s (float): End time of the ROI (seconds).
        style_settings (dict): Style settings dictionary for matplotlib.
    """
    plt.style.use('dark_background')
    plt.rcParams.update(style_settings)
    common_style = {'color': 'lightgray'}
    freq_limit = 5000  # Spectrum plot frequency limit
    db_floor = -80     # Spectrum plot dynamic range floor (dB)
    epsilon = 1e-12    # Avoid log10(0)
    min_fft_len = 44100 # Minimum FFT length (for zero-padding)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"Inverse Filtering Results ({roi_start_s:.2f}s - {roi_end_s:.2f}s)", color='lightgray')

    # --- 1. Top Left: Audio Spectra Overlay (Zero-padded & Windowed) ---
    ax1 = fig.add_subplot(2, 2, 1)
    plot_success_ax1 = False
    N_orig = len(original_audio) if original_audio is not None else 0

    # Process original audio
    if N_orig > 0:
        audio_to_fft = original_audio
        N_fft_orig = N_orig
        # Check if zero-padding is needed
        if N_orig < min_fft_len:
            pad_width = min_fft_len - N_orig
            # Pad at the end of the signal
            audio_to_fft = np.pad(audio_to_fft, (0, pad_width), mode='constant', constant_values=0)
            N_fft_orig = min_fft_len
            print(f"Original audio zero-padded to {N_fft_orig} points for FFT.") # English print

        # Apply Hamming window
        win_orig = windows.hamming(N_fft_orig, sym=False)
        audio_windowed = audio_to_fft * win_orig

        # Calculate FFT
        yf = fft(audio_windowed)
        xf = fftfreq(N_fft_orig, 1 / fs)
        mask = (xf >= 0) & (xf <= freq_limit)
        yf_masked = yf[mask]
        xf_masked = xf[mask]
        if len(yf_masked) > 0:
            magnitude_db = 20 * np.log10(np.abs(yf_masked) + epsilon)
            # Limit dynamic range floor
            max_mag_db = np.nanmax(magnitude_db) if np.any(np.isfinite(magnitude_db)) else 0
            magnitude_db = np.maximum(magnitude_db, max_mag_db + db_floor)

            ax1.plot(xf_masked, magnitude_db, color='cyan', label='Original Audio', alpha=0.8) # English label
            plot_success_ax1 = True

    # Process filtered audio
    N_filt = len(filtered_audio) if filtered_audio is not None else 0
    if N_filt > 0:
        filtered_to_fft = filtered_audio
        N_fft_filt = N_filt
        # Check if zero-padding is needed
        if N_filt < min_fft_len:
            pad_width = min_fft_len - N_filt
            filtered_to_fft = np.pad(filtered_to_fft, (0, pad_width), mode='constant', constant_values=0)
            N_fft_filt = min_fft_len
            print(f"Filtered audio zero-padded to {N_fft_filt} points for FFT.") # English print

        # Apply Hamming window
        win_filt = windows.hamming(N_fft_filt, sym=False)
        filtered_windowed = filtered_to_fft * win_filt

        # Calculate FFT
        yf_filt = fft(filtered_windowed)
        xf_filt = fftfreq(N_fft_filt, 1 / fs)
        mask_filt = (xf_filt >= 0) & (xf_filt <= freq_limit)
        yf_filt_masked = yf_filt[mask_filt]
        xf_filt_masked = xf_filt[mask_filt]
        if len(yf_filt_masked) > 0:
            magnitude_filt_db = 20 * np.log10(np.abs(yf_filt_masked) + epsilon)
            # Limit dynamic range floor
            max_mag_filt_db = np.nanmax(magnitude_filt_db) if np.any(np.isfinite(magnitude_filt_db)) else 0
            magnitude_filt_db = np.maximum(magnitude_filt_db, max_mag_filt_db + db_floor)

            ax1.plot(xf_filt_masked, magnitude_filt_db, color='lime', label='Filtered (Est. Source Deriv.)', alpha=0.8) # English label
            plot_success_ax1 = True

    # Set spectrum plot properties
    if plot_success_ax1:
        ax1.set_title("Audio Spectra Overlay (Zero-padded & Windowed)", **common_style) # English title
        ax1.set_xlabel("Frequency (Hz)", **common_style) # English label
        ax1.set_ylabel("Magnitude (dB)", **common_style) # English label
        ax1.tick_params(axis='both', colors='lightgray')
        ax1.grid(True, linestyle=':', alpha=0.4, color='gray')
        ax1.set_xlim(0, freq_limit)
        ax1.legend(fontsize='small', facecolor='#444444', edgecolor='gray', labelcolor='lightgray')
    else:
        ax1.text(0.5, 0.5, 'No Audio Data', ha='center', va='center', transform=ax1.transAxes, color='gray') # English text
        ax1.set_xlim(0, freq_limit)

    # --- 2. Top Right: Audio Waveform Overlay (Twin Axes) ---
    ax2 = fig.add_subplot(2, 2, 2)
    plot_success_ax2 = False
    if N_orig > 0 and N_filt > 0: # Ensure both signals have data
        min_len = min(N_orig, N_filt) # Use original lengths for comparison
        center_sample = min_len // 2
        # Display +/- 50ms window
        window_samples = int(0.050 * fs) # Total window duration
        start_idx = max(0, center_sample - window_samples // 2) # Center alignment
        end_idx = min(min_len, center_sample + window_samples // 2)

        if end_idx > start_idx:
            audio_zoom = original_audio[start_idx:end_idx]
            filtered_zoom = filtered_audio[start_idx:end_idx]
            # Time axis relative to center sample
            time_zoom_ms = (np.arange(start_idx, end_idx) - center_sample) / fs * 1000.0

            color_orig = 'cyan'
            line1 = ax2.plot(time_zoom_ms, audio_zoom, color=color_orig, label='Original Audio', alpha=0.9) # English label
            ax2.set_xlabel("Time relative to center (ms)", **common_style) # English label
            ax2.set_ylabel("Original Amp", color=color_orig) # English label
            ax2.tick_params(axis='y', labelcolor=color_orig, colors=color_orig)
            ax2.tick_params(axis='x', colors='lightgray')
            ax2.grid(True, linestyle=':', alpha=0.4, color='gray')
            if len(time_zoom_ms) > 0:
                ax2.set_xlim(time_zoom_ms[0], time_zoom_ms[-1])
            else:
                ax2.set_xlim(-25, 25) # Fallback

            # Create twin Y axis
            ax2_twin = ax2.twinx()
            color_filt = 'lime'
            line2 = ax2_twin.plot(time_zoom_ms, filtered_zoom, color=color_filt, label='Filtered (Est. Source Deriv.)', alpha=0.9) # English label
            ax2_twin.set_ylabel("Filtered Amp", color=color_filt) # English label
            ax2_twin.tick_params(axis='y', labelcolor=color_filt, colors=color_filt)

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper right', fontsize='small', facecolor='#444444', edgecolor='gray', labelcolor='lightgray')

            ax2.set_title("Audio Waveform Overlay (Center +/- 50ms)", **common_style) # English title
            plot_success_ax2 = True
        else:
            ax2.text(0.5, 0.5, 'ROI too short for zoom', ha='center', va='center', transform=ax2.transAxes, color='gray') # English text
            ax2.set_xlim(-25, 25)

    if not plot_success_ax2:
         ax2.text(0.5, 0.5, 'Waveform data unavailable\nor incompatible', ha='center', va='center', transform=ax2.transAxes, color='gray') # English text
         ax2.set_xlim(-25, 25) # Keep X axis range

    # --- 3. Bottom Left: EGG Spectrum (Zero-padded & Windowed) ---
    ax3 = fig.add_subplot(2, 2, 3)
    N_egg = len(egg_signal) if egg_signal is not None else 0

    if N_egg > 0:
        egg_to_fft = egg_signal
        N_fft_egg = N_egg
        # Check if zero-padding is needed
        if N_egg < min_fft_len:
            pad_width = min_fft_len - N_egg
            egg_to_fft = np.pad(egg_to_fft, (0, pad_width), mode='constant', constant_values=0)
            N_fft_egg = min_fft_len
            print(f"EGG signal zero-padded to {N_fft_egg} points for FFT.") # English print

        # Apply Hamming window
        win_egg = windows.hamming(N_fft_egg, sym=False)
        egg_windowed = egg_to_fft * win_egg

        # Calculate FFT
        yf_egg = fft(egg_windowed)
        xf_egg = fftfreq(N_fft_egg, 1 / fs)
        mask_egg = (xf_egg >= 0) & (xf_egg <= freq_limit)
        yf_egg_masked = yf_egg[mask_egg]
        xf_egg_masked = xf_egg[mask_egg]

        if len(yf_egg_masked) > 0:
            magnitude_egg_db = 20 * np.log10(np.abs(yf_egg_masked) + epsilon)
            # Limit dynamic range floor
            max_mag_egg_db = np.nanmax(magnitude_egg_db) if np.any(np.isfinite(magnitude_egg_db)) else 0
            magnitude_egg_db = np.maximum(magnitude_egg_db, max_mag_egg_db + db_floor)

            ax3.plot(xf_egg_masked, magnitude_egg_db, color='magenta')
            ax3.set_title("EGG Spectrum (Zero-padded & Windowed)", **common_style) # English title
            ax3.set_xlabel("Frequency (Hz)", **common_style) # English label
            ax3.set_ylabel("Magnitude (dB)", **common_style) # English label
            ax3.tick_params(axis='both', colors='lightgray')
            ax3.grid(True, linestyle=':', alpha=0.4, color='gray')
            ax3.set_xlim(0, freq_limit)
        else:
             ax3.text(0.5, 0.5, 'No EGG Freq Data in Range', ha='center', va='center', transform=ax3.transAxes, color='gray') # English text
             ax3.set_xlim(0, freq_limit)
    else:
        ax3.text(0.5, 0.5, 'No EGG Data', ha='center', va='center', transform=ax3.transAxes, color='gray') # English text
        ax3.set_xlim(0, freq_limit)

    # --- 4. Bottom Right: EGG Waveform (Center +/- 50ms) ---
    ax4 = fig.add_subplot(2, 2, 4)
    if N_egg > 0:
        center_sample_egg = N_egg // 2
        window_samples_egg = int(0.050 * fs) # Total window duration
        start_idx_egg = max(0, center_sample_egg - window_samples_egg // 2)
        end_idx_egg = min(N_egg, center_sample_egg + window_samples_egg // 2)

        if end_idx_egg > start_idx_egg:
            egg_zoom = egg_signal[start_idx_egg:end_idx_egg]
            time_zoom_ms_egg = (np.arange(start_idx_egg, end_idx_egg) - center_sample_egg) / fs * 1000.0
            ax4.plot(time_zoom_ms_egg, egg_zoom, color='wheat')
            ax4.set_title("EGG Waveform (Center +/- 50ms)", **common_style) # English title
            ax4.set_xlabel("Time relative to center (ms)", **common_style) # English label
            ax4.set_ylabel("Amplitude", **common_style) # English label
            ax4.tick_params(axis='both', colors='lightgray')
            ax4.grid(True, linestyle=':', alpha=0.4, color='gray')
            if len(time_zoom_ms_egg) > 0:
                ax4.set_xlim(time_zoom_ms_egg[0], time_zoom_ms_egg[-1])
            else:
                ax4.set_xlim(-25, 25)
        else:
             ax4.text(0.5, 0.5, 'ROI too short for zoom', ha='center', va='center', transform=ax4.transAxes, color='gray') # English text
             ax4.set_xlim(-25, 25)
    else:
        ax4.text(0.5, 0.5, 'No EGG Data', ha='center', va='center', transform=ax4.transAxes, color='gray') # English text
        ax4.set_xlim(-25, 25)

    # --- Final Adjustments ---
    try:
        # Use constrained layout for automatic spacing
        fig.set_layout_engine('constrained')
    except Exception as e:
        print(f"Note: Error during layout adjustment: {e}")
        try:
            # Fallback to tight_layout
            fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Leave space for suptitle
        except Exception as e2:
            print(f"Note: Fallback tight_layout also failed: {e2}")

    plt.show()