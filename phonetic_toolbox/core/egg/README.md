# EGG 信号处理核心算法 (Core EGG Signal Processing)

本模块 (`phonetic_toolbox/core/egg`) 包含用于处理电声门图 (Electroglottography, EGG) 信号的核心数学算法。这些算法独立于 GUI 或具体的业务逻辑，专注于纯粹的信号分析与处理。

## 模块组成

### 1. `analysis.py`: 事件检测与参数计算

该文件提供了从 EGG 信号中提取关键声门事件（GCI, GOI）以及计算声门商数（CQ, SQ）的函数。

#### 主要函数

*   **`find_gci_goi_peak_min_criterion(...)`**
    *   **功能**: 检测 EGG 信号中的以下关键事件：
        *   **GCI (Glottal Closure Instant)**: 声门闭合时刻，通常对应 dEGG (微分信号) 的正向峰值。
        *   **GOI (Glottal Opening Instant)**: 声门开放时刻，通常对应 dEGG 的负向峰值或下降沿的特定阈值点。
        *   **Peak**: 每个声门周期的最大接触点。
    *   **算法**: 支持两种检测策略：
        *   `"slope"` (斜率法): 基于 dEGG 的极值点（最大值对应 GCI，最小值对应 GOI）进行检测。这是最常用的方法。
        *   `"scale"` (尺度/阈值法): 基于幅度的百分比阈值（如 25% 或 50%）来确定事件时刻。
    *   **特性**: 包含了自动峰值/谷值显著度 (Prominence) 搜索，以过滤噪声和伪峰。

*   **`calculate_cq_sq(...)`**
    *   **功能**: 基于检测到的 GCI, GOI 和 Peak 时间点，计算逐周期的声门商数。
    *   **参数**:
        *   **CQ (Contact Quotient, 接触商)**: 声门接触时长占整个声门周期的比例。$CQ = \frac{T_{closed}}{T_{period}}$
        *   **SQ (Speed Quotient, 速度商)**: 声门闭合阶段与开放阶段持续时间的比率，或去接触时长与接触建立时长的比率。反映了声带振动的对称性。

### 2. `inverse_filtering.py`: 逆滤波

该文件实现了从声学信号（Audio）中估计声门气流波形（Glottal Flow Waveform）的算法。

#### 主要函数

*   **`apply_simplified_cp_inverse_filtering(...)`**
    *   **功能**: 实现简化的闭合相逆滤波 (Closed-Phase Inverse Filtering, CP-IF)。
    *   **原理**:
        1.  **闭合相识别**: 利用 EGG 信号确定的 GCI 位置，定位声门闭合区间。理论上，在闭合区间内，声门气流为零（或恒定），此时语音信号主要由声道共振产生。
        2.  **LPC 估计**: 仅在闭合区间内对预加重后的信号进行 LPC (线性预测编码) 分析，从而准确估计声道的共振特性（共振峰），避免声门源的干扰。
        3.  **逆滤波**: 使用估计出的声道滤波器对原始语音信号进行逆滤波，抵消声道共振，从而还原声门源信号（及其导数）。
    *   **特性**: 包含自动阶数选择、频谱倾斜 (Spectral Tilt) 补偿以及预加重/去加重处理。

## 依赖

*   `numpy`: 用于数值计算和数组操作。
*   `scipy.signal`: 用于信号处理（滤波、峰值查找、LPC 求解）。
*   `scipy.linalg`: 用于求解 Toeplitz 方程 (LPC 系数计算)。
