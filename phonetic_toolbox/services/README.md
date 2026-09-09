# 业务逻辑与服务层 (Business Logic & Services Layer)

本目录 (`phonetic_toolbox/services`) 包含了应用程序的**中间层 (Middle Layer)** 代码。服务层的主要职责是串联底层的核心算法 (`core`) 与数据输入输出 (`io`)，为上层 GUI 或 API 提供完整的业务功能。

## 目录结构

| 模块/包 | 主要类/功能 | 说明 |
| :--- | :--- | :--- |
| **`acoustic_service.py`** | `AcousticAnalysisService` | **核心业务服务**。负责协调音频读取、参数配置、算法调用（F0, 共振峰, 频谱参数等）、结果整合与对齐。 |
| **`egg_service.py`** | `EGGService` | **EGG 分析服务**。负责 EGG 信号处理、事件检测 (GCI/GOI)、参数计算 (CQ/SQ) 及逆滤波。 |
| **`lpc_service.py`** | `LPCSpectrumService` | **LPC 谱图服务**。负责音频加载、TextGrid 标签提取、LPC 核心调用与图片导出。 |
| **`spec2wav_service.py`** | `Spec2WavService` | **语谱图转音频服务**。负责图像读取、参数组装与 Griffin-Lim 重建流程编排。 |
| **`manipulation_service.py`** | `ManipulationService` | **变调与合成服务**。负责音频加载、基频修改、合成及批量生成任务。 |
| **`phonation_synthesis_service.py`** | `PhonationSynthesisService` | **发声类型合成服务**。负责 WAV 加载、Praat/REAPER F0 复用、LPC 残差分析、连续统编排及 WAV/CSV 导出。 |
| **`web_praat_server.py`** | `ensure_server`, `save_lip_offset` | **语音标注对齐服务**。仅监听 127.0.0.1，提供 TextGrid/音频/唇形 REST API，并以临时文件 + 原子替换安全写回唇形偏移。 |
| **`lip_service.py`** | `LipExtractionService` | **唇形提取启动服务**。负责定位外部项目目录并打开入口脚本。 |
| **`ipa_trans_service.py`** | `IPATransService` | **普通话转 IPA 启动服务**。负责按需生成并打开前端 HTML 页面。 |
| **`perception_service.py`** | `PerceptionExperimentService` | **感知实验启动服务**。负责定位并打开外部纯前端 HTML 页面。 |
| **`settings_service.py`** | `SettingsService` | **配置管理服务** (单例模式)。负责持有并分发运行时 `AcousticConfig` 对象。 |
| **`io/`** | - | **输入输出子模块**。处理特定格式文件的读写。 |
| ├── `wav.py` | `read_wav`, `write_wav` | 音频文件读写与格式转换 (Int -> Float, Stereo -> Mono)。 |
| ├── `textgrid.py` | `TextGrid`, `parse_textgrid` | Praat TextGrid 标注文件的解析与生成 (支持长/短格式)。 |
| ├── `excel.py` | `save_excel` | 将分析结果保存为 Excel/CSV 文件，自动处理数组长度对齐。 |
| └── `lip.py` | `read_lip_data` | 读取唇形分析数据 (.pkl) 并根据时间戳进行插值对齐。 |

## 详细功能说明

### 1. AcousticAnalysisService (声学分析服务)

`AcousticAnalysisService` 是本软件最核心的业务流程控制器。它封装了复杂的声学分析流程，使 GUI 层只需调用 `analyze_file` 即可获得最终结果。

**主要流程 (`analyze_file`):**

1.  **音频加载**: 使用 `io.wav` 读取音频并归一化为浮点单声道信号。
2.  **预处理**:
    *   计算能量 (Intensity)。
    *   生成静音掩码 (Silence Mask) 和清浊音掩码 (Voiced Mask)。
3.  **核心参数提取**:
    *   **Formants**: 调用 `core.acoustic.compute_praat_formants` (含自动移位)。
    *   **F0**: 调用 `compute_praat_f0_track` 与 `compute_reaper_f0`，保留原始轨迹时间坐标。
    *   **统一时间网格**: 以共振峰帧数建立零起点固定帧移网格，并通过 `align_track_to_grid` 映射 Praat/REAPER；范围外和无声间隙保持 `NaN`。
4.  **频谱参数计算 (Batch)**:
    *   调用 `core.acoustic.compute_spectral_features_batch` 高效计算 H1, H2, A1-A3, H2K, H5K 等。
5.  **参数校正**:
    *   应用 Iseli & Alwan 算法校正谐波幅度 (如 `H1*-H2*`, `H1*-A1*`)。
6.  **数据融合**:
    *   如果提供了 `.TextGrid` 文件，自动解析并将其作为新列合并到结果中。
    *   如果提供了唇形数据 (`.pkl`)，自动进行时间对齐并合并。
7.  **平滑与掩码**: `smooth_preserving_gaps` 仅在连续有限区间内平滑，并在完成后重新应用静音/清音掩码。
8.  **输出**: 返回强类型 `AnalysisResult`；需要表格时调用 `result.to_dataframe()`。

### 2. ManipulationService (变调服务)

`ManipulationService` 处理与音频修改相关的所有业务逻辑。

*   **加载音频**: 读取音频文件并提取初始基频 (F0)。
*   **合成**: 调用 `core.manipulation.synthesis.synthesize_from_pitch` 进行单文件合成。
*   **批量处理**: 调用 `core.manipulation.batch_utils.generate_batch_linear` 生成一系列变调文件。

### 3. Launcher 服务 (唇形 / IPA / 感知实验)

这类服务的职责是“定位入口 + 打开界面”，不在 GUI 层写路径解析与打开逻辑。

*   **`LipExtractionService`**:
    *   定位外部唇形工程目录，寻找 `run_gui.py` 作为入口并启动。
*   **`IPATransService`**:
    *   检查 `ipa_converter.html` 是否存在或是否需要更新，必要时调用生成脚本。
    *   使用默认浏览器打开页面。
*   **`PerceptionExperimentService`**:
    *   默认优先打开 `perception_experiment.html`。
    *   支持环境变量 `PHONETIC_TOOLBOX_PERCEPTION_PROJECT_DIR` 覆盖目录。
    *   若默认页面缺失，自动回退到目录中首个 `*.html` 页面。

### 3.1 PhonationSynthesisService (发声类型合成服务)

`PhonationSynthesisService` 将文件 I/O 和任务编排放在 Services 层，Core 层只接收数组与强类型配置。

*   **分析**: 读取并重采样源/目标 WAV，复用项目已有的 Praat AC 与 REAPER F0 实现，统一映射到毫秒网格，再计算 LPC、残差和脉冲位置。
*   **生成**: 调用 Core 生成三类连续统；“生成全部”按源→目标、目标→源各三类输出六组刺激。
*   **导出**: 每组写出逐级 WAV 和拼接试听 WAV；【保存 F0 CSV】另行写出当前源/目标 F0 轨迹；支持后台线程进度回调与协作取消。

### 3.2 WebPraatServer (语音标注对齐服务)

*   **本机边界**: 在主进程守护线程中监听 `127.0.0.1` 随机端口，不依赖外网；主程序退出后服务结束。
*   **语料接口**: 递归扫描同名 WAV/TextGrid，读取 `.lab` 词表和会话目录中的真实唇形 pkl，并提供静态前端资源。
*   **保存语义**: TextGrid 默认另存为 `_webedit`；`metadata.lip_manual_offset` 独立写回 pkl。pkl 先完整写入同目录临时文件，再使用 `os.replace` 原子替换，失败时保留原文件。

### 4. EGGService (EGG 分析服务)

`EGGService` 专注于电声门图 (Electroglottography) 信号的处理。

*   **信号处理**: 加载 WAV 文件中的 EGG 通道（支持自动交换左右声道），执行去趋势、高通/低通滤波。
*   **事件检测**:
    *   计算 dEGG (微分信号)。
    *   检测声门闭合时刻 (GCI) 和 声门开放时刻 (GOI)。支持基于**斜率 (Slope)** 和 **多尺度 (Scale)** 两种算法。
    *   搜索每个声门周期的峰值 (Peak) 和谷值 (Valley)。
*   **参数计算**:
    *   基于 GCI/GOI 计算接触商 (CQ) 和 速度商 (SQ)。
*   **逆滤波**:
    *   实现了简化的 CP-IF (Closed Phase Inverse Filtering) 算法，用于从音频信号中去除声道共振，估计声门气流波形。

### 5. SettingsService (配置服务)

`SettingsService` 实现了单例模式，确保全局配置的一致性。

*   **唯一性**: 整个应用生命周期内只有一个实例。
*   **类型化配置**: 以强类型 `AcousticConfig` 数据类 (Dataclass) 作为唯一配置载体。
*   **代码即配置**: 默认值来源于 `models/config.py`，不依赖外部 JSON 文件。
*   **运行时修改**: GUI 修改仅影响当前进程内存对象，重启后恢复代码默认值。

### 6. IO 模块

IO 模块将文件格式的解析细节与业务逻辑分离。

*   **`wav.py`**: 封装了 `scipy.io.wavfile`，增加了自动转单声道 (Mono) 和 归一化 (Float -1.0~1.0) 的逻辑，确保核心算法接收到标准数据。
*   **`textgrid.py`**: 提供了完整的 Python dataclass 定义 (`TextGrid`, `Tier`, `Interval`)，方便在代码中操作标注数据，而不是处理原始字符串。
*   **`excel.py`**: 在保存 Excel 时，能够智能处理标量 (Scalar) 与数组 (Array) 的混合数据，自动广播标量以匹配数组长度。
    *   写入失败（例如文件被 Excel 占用）必须向上抛出；批处理统一将该文件计入 `failed`，不得误报为 `processed`。

## 使用示例

### 调用声学分析服务

```python
from phonetic_toolbox.services.acoustic_service import AcousticAnalysisService
from phonetic_toolbox.models.config import AcousticConfig

# 1. 准备配置
config = AcousticConfig(
    min_f0=75.0,
    max_f0=600.0,
    frameshift_ms=10.0,
    use_reaper=True
)

# 2. 初始化服务
service = AcousticAnalysisService()

# 3. 执行分析
# result 是 AnalysisResult，包含 F0、Formants、H1-A3、Jitter、Shimmer 等轨迹
try:
    result = service.analyze_file(
        wav_path="test.wav",
        config=config,
        textgrid_path="test.TextGrid" # 可选：合并标注
    )
    df = result.to_dataframe()
    print(df.head())
    
    # 4. 保存结果
    service.save_results(result, "result.xlsx")
    
except Exception as e:
    print(f"分析失败: {e}")
```

### 获取/修改设置

```python
from phonetic_toolbox.services.settings_service import SettingsService

settings = SettingsService()

# 获取配置对象 (推荐)
config_obj = settings.get_config_object()
print(f"当前 F0 范围: {config_obj.min_f0} - {config_obj.max_f0}")

# 修改设置
settings.set("min_f0", 60.0) # 更新当前运行时配置对象
```

## 备注 (Updated 2026-03-15)

1.  本文档已补充 launcher 类服务（lip / ipa / perception）的职责划分。
2.  文档描述与当前代码结构对齐，仅做增量补充，不涉及删除未审阅脚本。

## 备注 (Updated 2026-03-18)

1.  清理未被调用的历史条目，保持文档与当前代码一致。

## 备注 (Updated 2026-07-15)

1.  声学服务改为按显式时间坐标对齐 Praat/REAPER，并保留 `NaN` 间隙和最终掩码。
2.  Excel 写入异常由批处理统一记录，不再在 IO 层吞掉。
3.  新增发音物理模拟器启动服务及对应结构化结果。

## 备注 (Updated 2026-08-23)

1.  新增 `PhonationSynthesisService`，集中承接发声类型合成的音频读取、分析编排和结果导出。

## 数据一致性约定 (Updated 2026-09-06)

- 网页语料 ID 绑定本次扫描和绝对路径；重新扫描后旧 ID 失效，旧标签页的保存被拒绝，页面修改仍保留。
- `io.lip.resolve_lip_time_axis` 是网页与参数导出的共同时间轴入口：优先使用 `metadata.audio_first_frame_time`，其次使用配套时间戳文件的 `start_time`。有绝对时间和音频起点时保留实际延迟，包括旧录音；仅有相对时间的旧格式才以首帧归零。手动偏移在此基础上单独加一次。
- `export_continuum`、`generate_selected`、`generate_all` 每次均创建新批次；调用者应使用返回的实际输出路径。六组生成共享同一批次，组内目录命名不变。`manifest.json` 仅在全部输出完成后标记 `complete`，失败或取消保持 `incomplete`；原有批次不受影响。
- 分析结果保留 `source_path`，编辑 F0 时继续保留该路径。正常生成批次保存源/目标路径、分析/合成配置、实际 F0 CSV 和输出清单。

## 声道工作台（2.2.0）

`vocal_tract_service.py` 通过 API 启动主应用所有的原生工作进程。`vocal_tract/` 承接 HTTP、统一音频输出、关键帧缓存与用户配置；主应用退出主动关闭，异常退出由 Windows Job Object 与父句柄回收。配置写 LocalAppData，静态资源只读，生产代码不依赖原型。详见 [模块说明](vocal_tract/README.md)。
