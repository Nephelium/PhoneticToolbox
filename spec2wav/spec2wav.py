import numpy as np
import cv2
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
# Use numpy.fft instead of scipy.fft for better PyInstaller compatibility
from numpy.fft import fft, ifft


# ============================================================================
# Pure NumPy/SciPy implementation of Griffin-Lim algorithm
# This avoids the numba/llvmlite dependency that causes DLL loading issues
# when packaged with PyInstaller.
# ============================================================================

def _stft(y, n_fft, hop_length, win_length, window):
    """
    Short-time Fourier Transform using pure NumPy/SciPy.
    """
    # Pad signal
    pad_length = n_fft // 2
    y_padded = np.pad(y, (pad_length, pad_length), mode='reflect')
    
    # Calculate number of frames
    n_frames = 1 + (len(y_padded) - n_fft) // hop_length
    
    # Create output matrix
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    
    # Pad window to n_fft if needed
    if len(window) < n_fft:
        window_padded = np.zeros(n_fft)
        start = (n_fft - len(window)) // 2
        window_padded[start:start + len(window)] = window
        window = window_padded
    
    for i in range(n_frames):
        start = i * hop_length
        frame = y_padded[start:start + n_fft] * window
        spectrum = fft(frame)
        stft_matrix[:, i] = spectrum[:n_fft // 2 + 1]
    
    return stft_matrix


def _istft(stft_matrix, hop_length, win_length, n_fft, window, length=None):
    """
    Inverse Short-time Fourier Transform using pure NumPy/SciPy.
    """
    n_frames = stft_matrix.shape[1]
    
    # Reconstruct full spectrum (conjugate symmetric)
    full_spectrum = np.zeros((n_fft, n_frames), dtype=np.complex128)
    full_spectrum[:n_fft // 2 + 1, :] = stft_matrix
    full_spectrum[n_fft // 2 + 1:, :] = np.conj(stft_matrix[-2:0:-1, :])
    
    # Pad window to n_fft if needed
    if len(window) < n_fft:
        window_padded = np.zeros(n_fft)
        start = (n_fft - len(window)) // 2
        window_padded[start:start + len(window)] = window
        window = window_padded
    
    # Calculate output length
    expected_length = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_length)
    window_sum = np.zeros(expected_length)
    
    for i in range(n_frames):
        start = i * hop_length
        frame = np.real(ifft(full_spectrum[:, i]))
        y[start:start + n_fft] += frame * window
        window_sum[start:start + n_fft] += window ** 2
    
    # Normalize by window sum (avoid division by zero)
    window_sum = np.maximum(window_sum, 1e-8)
    y = y / window_sum
    
    # Remove padding
    pad_length = n_fft // 2
    y = y[pad_length:-pad_length]
    
    if length is not None:
        if len(y) > length:
            y = y[:length]
        elif len(y) < length:
            y = np.pad(y, (0, length - len(y)))
    
    return y


def griffinlim_numpy(S, n_iter=32, hop_length=512, win_length=None, n_fft=2048):
    """
    Griffin-Lim algorithm for phase reconstruction using pure NumPy/SciPy.
    
    This implementation avoids the numba/llvmlite dependency that causes
    DLL loading issues when packaged with PyInstaller.
    """
    if win_length is None:
        win_length = n_fft
    
    # Ensure win_length <= n_fft
    win_length = min(win_length, n_fft)
    
    # Create Hann window
    window = signal.windows.hann(win_length, sym=False)
    
    # Initialize with random phase
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = S * angles
    
    # Estimate output length
    n_frames = S.shape[1]
    length = hop_length * (n_frames - 1) + n_fft - 2 * (n_fft // 2)
    
    # Griffin-Lim iterations
    for _ in range(n_iter):
        # Inverse STFT
        y = _istft(S_complex, hop_length, win_length, n_fft, window, length)
        
        # Forward STFT
        S_rebuilt = _stft(y, n_fft, hop_length, win_length, window)
        
        # Update phase while keeping magnitude
        angles = np.exp(1j * np.angle(S_rebuilt))
        S_complex = S * angles
    
    # Final inverse STFT
    y = _istft(S_complex, hop_length, win_length, n_fft, window, length)
    
    return y


def load_spectrogram_image(image_path, time_end, freq_end, time_start=0, freq_start=0, min_dB=-30.0, max_dB=0.0):
    """
    从语谱图图像中提取频谱数据，并根据时间和频率范围进行缩放。
    
    参数:
        image_path (str): 语谱图图像路径。
        time_start (float): 语谱图横坐标的起始时间（秒），默认为0。
        time_end (float): 语谱图横坐标的结束时间（秒）。
        freq_start (float): 语谱图纵坐标的起始频率（Hz），默认为0。
        freq_end (float): 语谱图纵坐标的结束频率（Hz）。
        min_dB (float): 对数幅度谱的最小值（灰度值255对应的dB）。
        max_dB (float): 对数幅度谱的最大值（灰度值0对应的dB）。
        
    返回:
        np.ndarray: 线性幅度谱矩阵。
        int: hop_length，用于时间分辨率。
        int: n_fft（FFT大小），根据频谱高度自动计算。
    """
    # 读取图像并转换为灰度图（假设亮度表示频率强度）
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("无法加载图像，请检查路径是否正确。")

    # 获取图像的尺寸
    img_height, img_width = img.shape

    # 翻转图像上下顺序, 修正频谱颠倒
    img_flipped = np.flipud(img)

    # 修正灰度值到dB的映射：颜色越深幅度越强，颜色越浅幅度越弱。
    # 灰度0对应最强 (max_dB)，灰度255对应最弱 (min_dB)
    log_spectrogram = max_dB - (img_flipped / 255.0) * (max_dB - min_dB) 

    # 从dB转换为线性幅度谱（并且声音大一点2333）
    linear_spectrogram = 10 ** (log_spectrogram / 10.0) * 10 

    # 计算每个像素对应的时间间隔和频率间隔
    time_range = np.linspace(time_start, time_end, img_width)
    freq_range = np.linspace(freq_start, freq_end, img_height)

    # 计算 hop_length (步长) 来确保音频的长度与图像时间范围一致
    duration = time_end - time_start
    hop_length = int(duration / img_width * int(2 * freq_end))  # 采样率设置为截止频率的两倍

    # 计算 n_fft，确保它与频谱矩阵的高度匹配
    n_fft = 2 * (img_height - 1)

    return img, linear_spectrogram, log_spectrogram, hop_length, n_fft


def spectrogram_to_audio(spectrogram, hop_length, window_length, n_fft, sr=22050, n_iter=32):
    """
    使用Griffin-Lim算法从频谱矩阵重建音频。
    
    使用纯 NumPy/SciPy 实现，避免 numba/llvmlite 依赖导致的 DLL 加载问题。
    
    参数:
        spectrogram (np.ndarray): 频谱矩阵。
        hop_length (int): 每帧的步长，用于控制音频的时间长度。
        window_length (int): 窗长，用于控制频率分辨率。
        n_fft (int): 用于FFT的尺寸，需与频谱矩阵的高度匹配。
        sr (int): 采样率，默认为 22050 Hz。
        n_iter (int): Griffin-Lim算法的迭代次数，默认为32次。
    
    返回:
        np.ndarray: 重建的音频信号。
    """
    # Ensure window_length is not greater than n_fft
    if window_length > n_fft:
        window_length = n_fft
    
    # 使用纯 NumPy/SciPy 实现的 Griffin-Lim 算法
    audio = griffinlim_numpy(spectrogram, n_iter=n_iter, hop_length=hop_length, 
                             win_length=window_length, n_fft=n_fft)
    return audio

def main():
    # 输入图像路径
    image_path = 'spectrogram_image.png'

    # 手动输入语谱图的结束时间和结束频率范围
    time_end = float(input("请输入语谱图的结束时间（秒）："))
    freq_end = float(input("请输入语谱图的结束频率（Hz）："))

    # 设置采样率为截止频率的两倍
    sr = int(2 * freq_end)

    # 加载语谱图图像并提取频谱矩阵，默认起始时间和起始频率为0
    img, linear_spectrogram, log_spectrogram, hop_length, n_fft = load_spectrogram_image(
        image_path, time_end, freq_end
    )

    # 可视化灰度值映射到dB的结果 (log_spectrogram)
    plt.figure(figsize=(8, 8))
    plt.subplot(3, 1, 1)
    plt.imshow(log_spectrogram, cmap='gray') 
    plt.colorbar() 
    plt.title('Log Spectrogram (dB, after grayscale to dB conversion)')

    # 可视化线性幅度谱 (linear_spectrogram)
    plt.subplot(3, 1, 2)
    plt.imshow(linear_spectrogram, cmap='gray') 
    plt.colorbar() 
    plt.title('Linear Spectrogram (after dB to linear conversion)')

    # 继续定义窗长并重建音频
    window_length = 128  #  窗长为128

    # 重建音频
    audio = spectrogram_to_audio(linear_spectrogram, hop_length, window_length, n_fft, sr=sr)

    # 保存音频
    output_audio_path = f'reconstructed_audio_window_{window_length}.wav'
    sf.write(output_audio_path, audio, sr)
    print(f"音频已保存为 {output_audio_path}，窗长为 {window_length}")

    # 计算重建后的幅度谱 (使用纯 NumPy/SciPy 实现的 STFT)
    from scipy import signal as scipy_signal
    window = scipy_signal.windows.hann(min(window_length, n_fft), sym=False)
    reconstructed_spectrogram = np.abs(_stft(audio, n_fft, hop_length, window_length, window))

    plt.subplot(3, 1, 3)
    plt.imshow(reconstructed_spectrogram, cmap='gray') 
    plt.colorbar() 
    plt.title('Reconstructed Spectrogram (after Griffin-Lim)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()