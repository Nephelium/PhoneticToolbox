# 音频修改与合成核心模块 (Core Manipulation Module)

本模块 (`phonetic_toolbox/core/manipulation`) 包含音频基频（F0）和时长的修改与合成算法。

## 核心功能

*   **基频合成 (Pitch Synthesis)**: 使用 Praat/Parselmouth 的 PSOLA (Pitch Synchronous Overlap and Add) 算法进行高质量的基频修改和语音重合成。
*   **批量处理 (Batch Processing)**: 支持批量生成线性变化的基频序列（Linear Interpolation），用于生成一系列具有特定音高轮廓的刺激材料。

## 文件说明

### 1. `synthesis.py`

核心合成逻辑，封装了 `parselmouth` 的底层操作。

*   **`synthesize_from_pitch(snd, times, modified_f0, xmin, xmax, speed=1.0)`**
    *   **功能**: 根据修改后的基频轨迹和语速因子合成新音频。
    *   **参数**:
        *   `snd`: 原始 `parselmouth.Sound` 对象。
        *   `times`: 时间轴数组。
        *   `modified_f0`: 修改后的 F0 值数组。
        *   `xmin`, `xmax`: 合成的时间范围。
        *   `speed`: 语速倍率 (1.0 为原速，>1.0 加速，<1.0 减速)。
    *   **流程**:
        1.  提取指定时间段的音频 (`extract_part`)。
        2.  创建一个新的 `PitchTier` 对象，填入 `modified_f0` 数据。
        3.  将音频转换为 `Manipulation` 对象 (`To Manipulation`)。
        4.  替换 `PitchTier` (`Replace pitch tier`)。
        5.  如果 `speed != 1.0`，创建并替换 `DurationTier`。
        6.  执行重合成 (`Get resynthesis (overlap-add)`)。

### 2. `batch_utils.py`

批量生成逻辑，支持复杂的线性插值和组合生成。

*   **`generate_batch_linear(...)`**
    *   **功能**: 生成一系列基频呈线性变化的音频文件。
    *   **核心概念**:
        *   **关键点 (Key Points)**: 包括起始点 (`t1`)、终止点 (`t2`) 以及中间的拐点 (`knot_points`)。
        *   **频率列表**: 每个关键点可以对应一个频率列表 (如 `[100, 110, 120]`)。
        *   **连接模式 (Connection Modes)**:
            *   `"order"` (顺序): 第 i 个文件使用列表中的第 i 个值。要求所有列表长度一致。
            *   `"reverse"` (逆序): 与顺序相反。
            *   `"full"` (全连接/笛卡尔积): 该点的所有值与其他点的所有值进行组合。
            *   `"constant"` (常量): 所有文件使用同一个值。
        *   **Offset Mode (偏移模式)**:
            *   `False` (默认): 指定的是绝对频率值 (Hz)。
            *   `True`: 指定的是相对于原始 F0 的偏移量 (Hz)。
    *   **输出**:
        *   自动在原音频目录下生成修改后的 WAV 文件。
        *   文件名包含参数信息，如 `..._lin_Fpath_100.0-200.0_combo1.wav`。
