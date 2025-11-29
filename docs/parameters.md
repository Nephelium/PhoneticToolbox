# PhoneticToolbox 声学参数详解

本文档详细介绍了 PhoneticToolbox (PyQt6) 能够提取的各类声学参数。这些参数的定义、计算方法及命名规范旨在与原始 MATLAB 版 VoiceSauce 保持高度一致，以确保研究结果的可比性。

## 1. 基本设置与时域参数

### `frameshift` (帧移)
-   **定义**：参数计算的时间步长。
-   **默认值**：1 ms (MATLAB 版默认)，用户可在设置中修改（如 5 ms, 10 ms）。
-   **说明**：所有随时间变化的参数（Time-varying parameters）均以此帧移进行采样。

### `Fs` (采样率)
-   **定义**：音频文件的采样频率，单位 Hz。

### `Energy` (能量)
-   **定义**：信号在以当前时刻为中心的窗口内的均方根能量（的对数形式，通常不做对数也可，视具体实现而定，本工具箱输出为能量和）。
-   **计算方法**：
    -   窗口长度：5 个基频周期（$5/F_0$）。如果 $F_0$ 未定义，则使用默认窗口长度。
    -   实现参考：对应 MATLAB 版 `func_GetEnergy.m`。
-   **用途**：反映语音的响度变化。

---

## 2. 基频 (F0) 与共振峰 (Formants)

### `pF0` (Praat F0)
-   **定义**：使用 Praat 的自相关算法 (To Pitch (cc)) 估计的基频。
-   **单位**：Hz。
-   **计算细节**：
    -   时间步长：同 `frameshift`。
    -   静音阈值、发声阈值等参数遵循 Praat 默认或用户设置。
    -   无声段（Unvoiced）或无法检测到基频的帧标记为 `NaN`。

### `strF0` (Straight F0)
-   **定义**：使用 STRAIGHT 算法估计的基频（如已集成）。
-   **说明**：本 Python 版本目前主要依赖 Praat 与 READER 算法，STRAIGHT 算法视具体安装情况而定。

### `pF1`, `pF2`, `pF3`, `pF4` (共振峰频率)
-   **定义**：使用 Praat 的 Burg 算法估计的前四个共振峰频率。
-   **单位**：Hz。
-   **计算细节**：
    -   最大共振峰频率：通常男性设为 5000 Hz，女性设为 5500 Hz。
    -   共振峰数量：通常设为 5（以获得较好的 F1-F4 估计）。
    -   预加重：50 Hz。

### `pB1`, `pB2`, `pB3`, `pB4` (共振峰带宽)
-   **定义**：对应上述共振峰的带宽 (Bandwidth)。
-   **单位**：Hz。

---

## 3. 频谱幅度参数 (Spectral Amplitudes)

此类参数测量特定谐波或共振峰处的频谱幅度，通常用于计算声源谱倾斜（Spectral Tilt）。所有幅度均以 dB 为单位。

### `H1`, `H2`, `H4`
-   **定义**：第一、第二、第四谐波的频谱幅度。
-   **计算方法**：
    -   首先获取 $F_0$ 估计值。
    -   在频谱上 $1 \times F_0$, $2 \times F_0$, $4 \times F_0$ 附近寻找局部最大值。
    -   为了提高精度，通常使用抛物线插值或其他优化方法定位峰值。
    -   实现参考：`func_GetH1_H2_H4.m`。

### `A1`, `A2`, `A3`
-   **定义**：最接近第一、第二、第三共振峰（F1, F2, F3）的谐波的频谱幅度。
-   **计算方法**：
    -   在 $F_1, F_2, F_3$ 频率附近搜索幅度最大的谐波峰值。
    -   实现参考：`func_GetA1A2A3.m`。

### `H2K`, `H5K`
-   **定义**：最接近 2000 Hz 和 5000 Hz 的谐波的频谱幅度。
-   **计算方法**：
    -   在 2000 Hz 和 5000 Hz 附近的频带内搜索最大峰值。
    -   实现参考：`func_Get2K.m`, `func_Get5K.m`。

---

## 4. 频谱倾斜与差值参数 (Spectral Tilt / Differences)

这些参数通过计算谐波或共振峰幅度的差值，反映声源的频谱倾斜特性，是分析发声类型（如气声、嘎裂声）的关键指标。

### 未校正参数 (Uncorrected, suffix 'u')
直接利用上述测量的幅度相减得到。

-   **`H1H2u`** ($H_1 - H_2$)：
    -   **物理意义**：反映声门开合商 (Open Quotient)。值越大，声门开放时间越长，声音越倾向于气声（Breathy）。
-   **`H2H4u`** ($H_2 - H_4$)：
    -   **物理意义**：反映声门脉冲的偏斜度 (Skewness) 或频谱的中频倾斜。
-   **`H1A1u`** ($H_1 - A_1$)：
    -   **物理意义**：反映第一共振峰区域的带宽或声门泄露。
-   **`H1A2u`** ($H_1 - A_2$)：
    -   **物理意义**：反映频谱的整体倾斜度。
-   **`H1A3u`** ($H_1 - A_3$)：
    -   **物理意义**：反映频谱的高频倾斜度。
-   **`H42Ku`** ($H_4 - H_{2K}$)：
    -   **物理意义**：中高频频谱倾斜。
-   **`2K5Ku`** ($H_{2K} - H_{5K}$)：
    -   **物理意义**：高频频谱倾斜。

### 校正参数 (Corrected, suffix 'c')
由于共振峰会提升其附近谐波的幅度，为了获得纯粹的声源特性，需要去除声道共振（Vocal Tract Resonance）的影响。本工具箱使用 **Iseli & Alwan (2004)** 的算法进行校正。

-   **`H1H2c`**, **`H2H4c`**
-   **`H1A1c`**, **`H1A2c`**, **`H1A3c`**
-   **校正原理**：
    -   利用估计的共振峰频率 ($F_1 \dots F_4$) 和带宽 ($B_1 \dots B_4$)，计算声道传递函数在各频率点的增益。
    -   从测量幅度中减去该增益。
    -   带宽估计：默认使用 **Hawks & Miller (1995)** 的公式基于 $F_i$ 估算 $B_i$，也可选择使用 Praat 实测带宽 ($pB_i$)。

---

## 5. 嗓音质量与噪声参数 (Voice Quality / Noise)

### `CPP` (Cepstral Peak Prominence - 倒谱峰显著性)
-   **定义**：倒谱图（Cepstrum）上基频倒谱峰幅度与回归直线的距离。
-   **单位**：dB。
-   **计算方法**：
    -   对每一帧信号计算倒谱。
    -   在预期的基频周期范围内（quefrency）寻找最大峰值。
    -   对倒谱进行线性回归拟合基线。
    -   计算峰值高出基线的幅度。
    -   实现参考：Hillenbrand et al. (1994); `func_GetCPP.m`。
-   **物理意义**：反映发声的周期性/稳定性。CPP 越高，声音周期性越好；CPP 越低，声音越粗糙或包含更多噪声（如病理嗓音、严重气声）。

### `HNR` (Harmonic-to-Noise Ratio - 谐波噪声比)
-   **定义**：谐波能量与噪声能量的比值。
-   **参数变体**：
    -   `HNR05`: 0-500 Hz 频带内的 HNR。
    -   `HNR15`: 0-1500 Hz 频带内的 HNR。
    -   `HNR25`: 0-2500 Hz 频带内的 HNR。
    -   `HNR35`: 0-3500 Hz 频带内的 HNR。
-   **计算方法**：
    -   使用 de Krom (1993) 的倒谱梳状滤波法 (Cepstral Comb Filtering) 或类似方法分离谐波与噪声分量。
    -   实现参考：`func_GetHNR.m`。
-   **物理意义**：值越大，噪声明分量越少，音质越“纯净”；值越小，嗓音中的气流噪声或不规则震动越多。

### `SHR` (Subharmonic-to-Harmonic Ratio - 次谐波谐波比)
-   **定义**：次谐波（如 $0.5 F_0$）幅度与主谐波（$F_0$）幅度的比值。
-   **计算方法**：
    -   基于 Sun (2002) 的算法 (`shrp.m`)。
    -   在对数频率域利用奇偶点移位差分来检测是否存在次谐波。
-   **物理意义**：用于检测**嘎裂声 (Creaky Voice/Fry)** 或**复音 (Diplophonia)**。当存在明显的倍频程或次谐波震动时，SHR 值会显著升高。

---

## 6. 参考文献 (References)

1.  **VoiceSauce (MATLAB)**: Shue, Y.-L., Keating, P., Vicenik, C., & Yu, K. (2011). VoiceSauce: A program for voice analysis. *Proceedings of ICPhS XVII*, 1846-1849.
2.  **Correction Formula**: Iseli, M., & Alwan, A. (2004). An improved correction formula for the estimation of harmonic magnitudes and its application to open quotient estimation. *IEEE Transactions on Speech and Audio Processing*, 12(6), 669-676.
3.  **Bandwidth Estimation**: Hawks, J. W., & Miller, J. D. (1995). A non-invasive technique for estimating glottal polynomial coefficients. *JASA*, 97(2), 1343-1344.
4.  **CPP**: Hillenbrand, J., Cleveland, R. A., & Erickson, R. L. (1994). Acoustic correlates of breathy vocal quality. *Journal of Speech, Language, and Hearing Research*, 37(4), 769-778.
5.  **SHR**: Sun, X. (2002). Pitch determination and voice quality analysis using subharmonic-to-harmonic ratio. *ICASSP 2002*.
6.  **HNR**: de Krom, G. (1993). A cepstrum-based technique for determining a harmonics-to-noise ratio in speech signals. *Journal of Speech, Language, and Hearing Research*, 36(2), 254-266.

---

**注意**：本工具箱在实现过程中，对于 FFT 窗口类型、插值算法等细节均尽量贴合 MATLAB 原版，但在极个别边缘情况（如极短音频、极端噪声）下，数值可能存在微小差异（通常 < 1%），属于正常浮点运算误差范畴。
