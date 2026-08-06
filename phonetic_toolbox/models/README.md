# 数据模型定义 (Data Models Definition)

本模块 (`phonetic_toolbox/models`) 用于定义跨层共享的数据结构。为了保证代码的类型安全和可维护性，我们使用 Python 的 `dataclasses` 来封装配置参数和分析结果，避免在不同层（GUI, Service, Core）之间传递非结构化的字典 (`dict`)。

## 核心模型

### 1. `config.py`: 配置模型

该文件定义了应用程序各模块所需的配置参数。所有默认参数值均在此处定义，遵循“代码即配置”原则。

*   **`AcousticConfig`**
    *   **用途**: 通用声学分析的配置。
    *   **包含参数**:
        *   基频范围 (`min_f0`, `max_f0`)
        *   窗长与步长 (`frameshift_ms`, `windowsize_ms`)
        *   算法选择 (`use_reaper` 等)
        *   共振峰设置 (`max_formant`, `num_formants`)
        *   静音/清浊音阈值 (`silence_threshold`)

*   **`EGGConfig`**
    *   **用途**: EGG 分析专用的配置。
    *   **包含参数**:
        *   峰值搜索门限 (`peak_prominence`, `auto_prominence`)
        *   滤波器设置 (`highpass_cutoff`, `lowpass_cutoff`)
        *   GCI/GOI 检测算法 (`gci_method`, `goi_method`)
        *   语谱图显示参数 (`spec_window_ms`, `spec_vmin`, `spec_vmax`)
        *   逆滤波阶数 (`if_order_heuristic_add`)

*   **`PitchManipulationConfig`**
    *   **用途**: 变调与语音合成的配置。
    *   **包含参数**: 目标基频范围、时间步长、批量处理模式（正序/倒序/固定值）等。

*   **`AnalysisResult`** (类)
    *   **用途**: 存储通用声学分析的最终结果。
    *   **结构**: 包含时间轴、F0 (Praat/REAPER)、共振峰 (F1-F4, B1-B4)、强度 (Intensity) 以及其他动态扩展的声学参数字典。提供 `to_dataframe()` 方法方便导出为 Pandas DataFrame。

### 2. `egg_models.py`: EGG 结果模型

该文件定义了 EGG 分析特有的结果容器。

*   **`EGGAnalysisResult`**
    *   **用途**: 存储 EGG 分析过程中的所有中间数据和最终结果。
    *   **字段**:
        *   **信号**: 原始 EGG (`egg_signal_raw`)、处理后 EGG (`egg_signal_processed`)、同步音频 (`audio_signal`)。
        *   **事件**: GCI 时间点、GOI 时间点、峰值时间点。
        *   **参数**: CQ (接触商) 轨迹、SQ (速度商) 轨迹。
        *   **F0**: 基于 GCI 计算的精确 F0 轨迹 (`gci_f0_values`) 以及音频提取的参考 F0。
        *   **元数据**: 采样率、文件时长。
        *   **其他**: 声门运动事件标记（Rise/Fall）。

### 3. `acoustic_models.py`: 显式时间轨迹

*   **`PitchTrack`**
    *   **用途**: 同时保存基频采样时间 (`times`) 与对应数值 (`values`)，避免只传数组时丢失 Praat/REAPER 的真实帧起点。
    *   **约定**: 时间单位为秒，基频单位为 Hz，无效或无声帧使用 `np.nan`。
    *   **不可变性**: 数据类使用 `frozen=True`，轨迹对象创建后不重新绑定字段。

### 4. `spec2wav_models.py`: 语谱图转音频模型

*   **`Spec2WavConfig`**
    *   **用途**: 定义语谱图重建参数（频率范围、时间范围、动态范围、采样率等）。
*   **`Spec2WavResult`**
    *   **用途**: 封装重建结果（波形、采样率、提示信息等），供 GUI 直接消费。

### 5. 启动结果模型

`lip_models.py`、`ipa_models.py`、`perception_models.py` 与 `articulatory_models.py` 遵循同一设计思路：用结构化结果替代松散字典，承载“是否成功 + 消息 + 关键路径”。

*   **`LipLaunchResult` (`lip_models.py`)**
    *   **用途**: 返回唇形提取入口启动状态、入口路径与工作目录。
*   **`IPATransLaunchResult` (`ipa_models.py`)**
    *   **用途**: 返回 IPA 页面生成/打开状态、HTML 路径及生成日志。
*   **`PerceptionLaunchResult` (`perception_models.py`)**
    *   **用途**: 返回感知实验页面打开状态、HTML 路径与工作目录。
*   **`ArticulatorySynthLaunchResult` (`articulatory_models.py`)**
    *   **用途**: 返回发音物理模拟器页面打开状态、HTML 路径与工作目录。

### 6. 语音合成模块的数据边界 (Speech Synthesis Boundary)

当前“语音合成实验室”采用以下数据边界约定：
*   **Core 合成参数**: `core/synthesis/klatt/tdklatt.py` 中 `KlattParam1980` 作为核心参数对象。
*   **GUI 编辑状态**: `speech_synthesis_widget.py` 内部使用 `ParameterCurve` 维护交互态曲线，不跨层传递到 Core 之外。
*   **服务层职责**: 与外部程序/文件 I/O 相关的逻辑应放在 `services/`，避免在 Core 层出现 Service 语义实现。

## 设计原则

1.  **单一数据源**: 所有的默认配置值都应在此处修改，不依赖外部 JSON 文件。
2.  **类型提示**: 所有字段都应包含类型注解 (`float`, `int`, `np.ndarray` 等)，以便 IDE 进行静态检查。
3.  **解耦**: 这些模型类不包含任何复杂的业务逻辑或算法实现，仅用于数据的存储和传递。
4.  **跨层统一契约**: GUI/API/Services 的返回值优先使用 `*Result` 数据类，避免魔法字符串键名。

## 备注 (Updated 2026-03-15)

1.  已补充与当前代码一致的启动结果模型说明（lip / ipa / perception）。
2.  本文档仅做增量更新，不涉及删除任何未审阅目录或脚本。

## 备注 (Updated 2026-07-15)

1.  新增 `PitchTrack`，将声学轨迹的真实时间坐标纳入跨层数据契约。
2.  新增 `ArticulatorySynthLaunchResult`，保持发音物理模拟器的 `GUI -> API -> Services -> Models` 调用边界。
