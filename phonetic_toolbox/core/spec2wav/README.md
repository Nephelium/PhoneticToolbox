# 语谱图转音频核心模块 (Spec2Wav Core)

本模块 (`phonetic_toolbox.core.spec2wav`) 实现了从语谱图图像 (Spectrogram Image) 逆向重建音频波形的核心算法。

## 1. 核心原理

该模块主要依赖 **Griffin-Lim 算法** 来从仅包含幅度信息 (Magnitude Spectrogram) 的语谱图中恢复丢失的相位信息 (Phase)，从而重建时域波形。

### 流程概览
1.  **图像预处理**: 将 RGB/RGBA 图像转换为灰度图。
2.  **数值映射**: 将像素灰度值 (0-255) 映射回对数幅度 (dB)，再转换为线性幅度谱 (Linear Magnitude Spectrogram)。
3.  **Griffin-Lim 迭代**:
    *   初始化随机相位。
    *   进行 ISTFT (逆短时傅里叶变换) 得到估计波形。
    *   对估计波形进行 STFT (短时傅里叶变换)。
    *   保留原幅度谱，更新相位为 STFT 结果的相位。
    *   重复上述步骤多次 (默认 32 次) 以收敛。
4.  **后处理**: 根据目标采样率进行重采样 (Resampling)。

## 2. 模块结构

*   **`griffin_lim.py`**:
    *   **`griffinlim_numpy`**: Griffin-Lim 算法的纯 NumPy/SciPy 实现。避免了 `librosa` 可能引入的复杂依赖或 `numba` 在打包时的兼容性问题。
    *   **`_stft` / `_istft`**: 自定义的 STFT 和 ISTFT 实现，确保 FFT 参数的精确控制。
    *   **`spectrogram_to_audio`**: 高层封装函数，直接将线性语谱图矩阵转换为音频数组。

*   **`image_processing.py`**:
    *   **`load_spectrogram_image`**: 负责加载图像并计算声学参数。
        *   **输入**: 图像路径或 NumPy 数组、时间范围 (Time Range)、频率范围 (Freq Range)、动态范围 (Dynamic Range dB)。
        *   **输出**: 线性幅度谱矩阵、Hop Length、N_FFT。
        *   **逻辑**: 根据图像宽度和给定的时间时长，自动计算 `hop_length`；根据图像高度，自动计算 `n_fft`。

*   **`common.py`**:
    *   **`resample_audio`**: 基于线性插值 (`np.interp`) 的重采样工具，用于将生成的非标准采样率音频转换为 44.1kHz 等标准采样率。
    *   **`amplitude_to_db`**: 幅度谱转 dB 谱工具，用于可视化验证。

## 3. 关键算法细节

### 3.1 频率与 FFT 尺寸的对应关系
语谱图的高度 ($H$) 对应于频率范围 ($F_{max} - F_{min}$)。在 STFT 中，频率桶数 (Bins) 为 $N_{fft}/2 + 1$。
因此，我们根据图像高度反推 $N_{fft}$：
$$ N_{fft} = 2 \times (H - 1) $$
这确保了生成的频谱图在频域上与原图像素一一对应。

### 3.2 时间与 Hop Length 的对应关系
语谱图的宽度 ($W$) 对应于时间长度 ($T$)。
$$ Hop Length = \frac{T \times SR}{W} $$
其中 $SR$ 为采样率。根据奈奎斯特采样定理，我们通常设定内部处理的采样率为 $2 \times F_{max}$。

## 4. 依赖说明
本模块仅依赖：
*   `numpy`
*   `scipy` (用于信号窗函数)
*   `cv2` (OpenCV, 用于图像读取与处理)

不依赖 `librosa` 或 `torch`，保持了轻量化和高兼容性。
