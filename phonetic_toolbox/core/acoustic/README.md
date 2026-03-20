# 声学参数估计核心模块 (Core Acoustic Analysis Module)

本目录 (`phonetic_toolbox/core/acoustic`) 包含了用于提取语音声学参数的核心算法实现。所有功能均已模块化拆分，每个文件对应特定的声学特性计算。

## 目录结构与功能说明

| 文件名 | 主要功能 | 核心函数 | 说明 |
| :--- | :--- | :--- | :--- |
| **`f0_praat.py`** | 基频 (F0) | `compute_praat_f0` | 调用 `parselmouth` (Praat) 计算 F0，支持自相关 (AC) 和互相关 (CC) 方法。 |
| **`f0_reaper.py`** | 基频 (F0) | `compute_reaper_f0` | 调用 REAPER 算法计算 F0。优先尝试调用外部 `reaper` 可执行文件 (C++版)，若失败则自动降级使用内置 Python 实现 (`reaper_python.py`)。 |
| **`reaper_python.py`** | 基频 (F0) | `run_python_impl` | REAPER 算法的纯 Python 实现，作为 `f0_reaper.py` 的后备方案，无需外部二进制依赖。 |
| **`formants_praat.py`** | 共振峰 | `compute_praat_formants` | 使用 Praat 的 Burg 算法计算 F1-F4 及其带宽。**包含自动限额移位逻辑**：若 F1/F2/F3/F4 超出预设频率范围 (如 F1>1200Hz)，自动将其归入下一共振峰槽位。 |
| **`spectral_batch.py`** | 频谱参数 (批量) | `compute_spectral_features_batch` | **(推荐)** 高效批量计算 H1-H4, A1-A3, H2K, H5K 等参数。采用 FFT + 抛物线插值，比逐个计算快数百倍。 |
| **`jitter_shimmer.py`** | 微扰参数 | `compute_jitter_shimmer` | 计算频率微扰 (Jitter) 和振幅微扰 (Shimmer)。支持多种标准：**Jitter (Local, RAP, PPQ5)**, **Shimmer (Local, APQ3, APQ5, APQ11)**。 |
| **`cpp.py`** | 倒谱峰值显著度 | `compute_cpp` | 计算 Cepstral Peak Prominence (CPP)，用于评估嗓音的周期性与气息感。 |
| **`hnr.py`** | 谐波噪声比 | `compute_hnr` | 计算多频带的 Harmonics-to-Noise Ratio (如 0-500Hz, 0-1500Hz 等)。 |
| **`shr.py`** | 次谐波-谐波比 | `compute_shr` | 计算 Subharmonic-to-Harmonic Ratio (SHR)，用于检测次谐波 (Subharmonics) 和音高感知的显著性。 |
| **`soe.py`** | 激励强度 | `compute_soe` | 计算 Strength of Excitation (SoE)，基于零频率滤波 (ZFF) 方法评估声门闭合瞬间的激励强度。 |
| **`spectral_slope.py`** | 频谱斜率 | `compute_spectral_slope` | 通过对对数幅度谱进行线性回归计算频谱斜率。 |
| **`lpc.py`** | LPC 谱包络 | `compute_lpc_spectrum` | 基于线性预测编码 (LPC) 计算频率-幅度谱包络，供 LPC 谱图界面与服务层调用。 |
| **`energy.py`** | 能量/音强 | `compute_energy` | 计算短时能量 (Intensity)，结果以 dB 为单位，与 Praat 一致。 |
| **`voicing.py`** | 清浊/静音检测 | `compute_silence_mask`<br>`compute_zcr_voicing_mask` | 基于音强阈值 (Intensity Threshold) 判断静音；基于过零率 (ZCR) 和能量判断清浊音 (Voiced/Unvoiced)。 |
| **`corrections.py`** | 参数校正 | `compute_H1A1A2A3_corrected`<br>`compute_H1H2_H2H4_corrected`<br>`compute_corrections_H2KH5K` | 根据 Iseli & Alwan (1999) 算法，利用 F1-F4 和 B1-B4 对 H1, H2, H4, A1-A3, H2K, H5K 进行校正。 |
| **`common.py`** | 通用工具 | `segment_for_frame` | 提供音频分帧、切片等底层工具函数。 |
| **`__init__.py`** | 包入口 | - | 统一导出上述所有核心函数，方便外部调用。 |

## 算法详细说明

### 1. Jitter & Shimmer (微扰参数)
基于 Praat 的 `To PointProcess (periodic, cc)` 获取脉冲点，计算相邻周期的变化。
- **Jitter**:
  - `Jitter_Local`: 相邻周期差的平均值 (百分比)。
  - `Jitter_RAP`: 3点平滑后的相对平均微扰 (Relative Average Perturbation)。
  - `Jitter_PPQ5`: 5点平滑后的周期微扰商 (Period Perturbation Quotient)。
- **Shimmer**:
  - `Shimmer_Local`: 相邻振幅差的平均值 (百分比)。
  - `Shimmer_APQ3`: 3点平滑后的振幅微扰商 (Amplitude Perturbation Quotient)。
  - `Shimmer_APQ5`: 5点平滑后的振幅微扰商。
  - `Shimmer_APQ11`: 11点平滑后的振幅微扰商。

### 2. Formants (共振峰 - 自动移位)
使用 Praat Burg 算法提取原始候选值，然后根据以下规则分配到 F1-F4 槽位：
- **F1**: 200Hz - 1200Hz
- **F2**: < 3200Hz
- **F3**: < 4600Hz
- **F4**: < 6000Hz
若某候选频率超出当前槽位的上限 (例如第一个候选值 > 1200Hz)，它将不会被标记为 F1，而是作为 F2 的候选值进行尝试。这有效防止了高频能量误判为低频共振峰。

### 3. Voicing Detection (清浊音检测)
提供两种机制：
1.  **基于 F0**: 凡是能计算出有效 F0 (>0) 的帧视为浊音。
2.  **基于 ZCR (Zero-Crossing Rate)**:
    -   计算每帧的过零率和能量。
    -   **静音判定**: 能量 < `max_energy * noise_floor_ratio` (默认 0.01)。
    -   **清音判定**: ZCR > `zcr_threshold` (默认 3000Hz) 或 能量过低。
    -   **浊音判定**: 能量足够且 ZCR 较低。

### 4. Iseli Corrections (参数校正)
实现了 Iseli & Alwan (1999) 的校正公式 `z(f)`.
- **H1, H2, H4**: 受 F1, F2 (有时 F3, F4) 影响校正。
- **A1, A2, A3**: 受 F1, F2, F3 影响校正 (自身频率附近的共振峰影响最大)。
- **H2K, H5K**: 新增支持。H2K (约2000Hz) 和 H5K (约5000Hz) 也会受到 F1-F4 的旁瓣影响，通过 `compute_corrections_H2KH5K` 进行校正。

### 5. LPC Spectrum (线性预测谱包络)
`lpc.py` 提供 `compute_lpc_spectrum(audio, fs, order)`，返回频率轴与 dB 幅度轴：
- 先进行稳定的 LPC 系数估计；
- 再将全极点模型转换为频率响应；
- 最终输出适合绘图与后续服务层处理的 `frequencies_hz` 与 `magnitude_db`。

## 依赖说明

本项目依赖以下 Python 库：
*   `numpy`: 数值计算基础。
*   `scipy`: 信号处理 (FFT, 滤波, 优化)。
*   `praat-parselmouth`: 调用 Praat 核心算法。

对于 `f0_reaper.py`，推荐系统路径中存在 `reaper` 可执行文件以获得最佳性能。如果未找到或执行失败，将自动使用 `reaper_python.py` (纯 Python 实现，速度较慢但兼容性好)。

## 使用示例

### 1. 基础参数计算 (推荐使用 Batch 模式)

```python
import numpy as np
import scipy.io.wavfile as wavfile
from phonetic_toolbox.core.acoustic import (
    compute_praat_f0,
    compute_praat_formants,
    compute_spectral_features_batch
)

# 读取音频
fs, y = wavfile.read("example.wav")
y = y.astype(float) / 32768.0  # 归一化

# 配置参数
frameshift_ms = 10.0
n_periods = 4

# 1. 计算 F0
f0 = compute_praat_f0("example.wav", frameshift_ms, min_f0=75, max_f0=600)
voiced_mask = (f0 > 0)

# 2. 计算共振峰 (带自动移位)
formants = compute_praat_formants("example.wav", frameshift_ms)
F1, F2, F3 = formants["pF1"], formants["pF2"], formants["pF3"]

# 3. 批量计算频谱参数 (H1-H4, A1-A3, H2K, H5K) - 高效
spec_res = compute_spectral_features_batch(
    y, fs, frameshift_ms, f0, F1, F2, F3, n_periods, voiced_mask
)

print("Mean H1:", np.nanmean(spec_res["H1"]))
print("Mean A1:", np.nanmean(spec_res["A1"]))
```

### 2. 参数校正 (Corrections)

使用 `corrections.py` 对原始测量的幅度进行校正，消除共振峰对邻近谐波的影响。

```python
from phonetic_toolbox.core.acoustic import compute_H1H2_H2H4_corrected, compute_H1A1A2A3_corrected

# 假设已从 spec_res 获取了 H1, H2, H4, A1, A2, A3 以及 F0, F1, F2, F3
# B1, B2 为带宽，如果未提供将使用 Hawks & Miller (1995) 公式估算

# 校正 H1-H2, H2-H4
corr_h = compute_H1H2_H2H4_corrected(
    spec_res["H1"], spec_res["H2"], spec_res["H4"],
    fs, f0, F1, F2
)
print("H1*-H2*:", np.nanmean(corr_h["H1H2c"]))

# 校正 H1-A1, H1-A2, H1-A3
corr_a = compute_H1A1A2A3_corrected(
    spec_res["H1"], spec_res["A1"], spec_res["A2"], spec_res["A3"],
    fs, f0, F1, F2, F3
)
print("H1*-A1*:", np.nanmean(corr_a["H1A1c"]))
```

### 3. LPC 谱包络计算

```python
import scipy.io.wavfile as wavfile
from phonetic_toolbox.core.acoustic import compute_lpc_spectrum

fs, y = wavfile.read("example.wav")
y = y.astype(float)
if y.ndim > 1:
    y = y[:, 0]
y = y / max(1.0, abs(y).max())

freq_hz, mag_db = compute_lpc_spectrum(y, fs=fs, order=50)
print(freq_hz.shape, mag_db.shape)
```

## 注意事项

1.  **帧中心对齐**: `common.py` 中的 `segment_for_frame` 默认使用 `(k + 1) * frameshift` 作为第 k 帧 (0-based) 的中心时间点。
2.  **单位**: 频率单位为 Hz，时间单位通常为 ms (参数设置) 或 s (内部计算)。幅度通常为 dB。
3.  **空值处理**: 当 F0 无法计算或音频片段无效时，函数通常返回 `np.nan`。
