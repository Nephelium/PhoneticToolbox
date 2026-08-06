# PhoneticToolbox v2 架构规范

本文件是 PhoneticToolbox_v2 的架构宪法。任何开发人员（包括 AI 助手）都必须严格遵守以下规则。

## 1. 核心原则

1.  **分层架构 (Layered Architecture)**: 依赖关系只能**由外向内**。
    *   Outer Layer (GUI) -> Middle Layer (Services/API) -> Inner Layer (Core/Models)
    *   Core 层**严禁**引用 Services 或 GUI 层。
    *   GUI 层**严禁**包含核心算法逻辑。

2.  **API 优先 (API First)**: 
    *   所有功能必须首先在 `core/` 或 `services/` 中作为独立的 Python 函数或类实现。
    *   GUI 仅仅是这些 API 的调用者。如果你的 GUI 代码中包含了大量 `if-else` 或循环计算逻辑，你做错了。

3.  **模块化 (Modularity)**:
    *   `core/` 下的各个子模块（如 `egg`, `acoustic`）应尽可能独立。
    *   跨模块调用应通过明确的接口进行。

## 2. 目录结构详解 (Updated 2026-07-15)

```text
PhoneticToolbox_v2/
├── docs/                       # 项目文档与已确认实施计划
├── phonetic_toolbox/           # 源代码根目录
│   ├── api/                    # [Middle Layer] 对外暴露的简洁 API (Facade)
│   │   └── __init__.py         # 导出 lip / IPA / perception / articulatory 等 Facade 入口
│   ├── core/                   # [Inner Layer] 纯粹的领域逻辑与算法
│   │   ├── acoustic/           # 声学参数提取
│   │   │   ├── README.md           # 声学模块说明文档
│   │   │   ├── __init__.py
│   │   │   ├── common.py           # 通用声学计算工具 (分帧、频率转换等)
│   │   │   ├── corrections.py      # 能量校正算法 (Iseli & Alwan)
│   │   │   ├── cpp.py              # 倒谱峰值突出度 (CPP) 计算
│   │   │   ├── energy.py           # 能量/强度计算 (Intensity dB)
│   │   │   ├── f0_praat.py         # Praat AC/CC 基频及显式时间轨迹
│   │   │   ├── f0_reaper.py        # REAPER 基频与打包二进制定位
│   │   │   ├── formants_praat.py   # Praat 算法共振峰提取 (含自动限额移位)
│   │   │   ├── hnr.py              # 谐波噪声比 (HNR) 计算 (多频带)
│   │   │   ├── jitter_shimmer.py   # 抖动与闪烁计算 (Local, RAP, PPQ5, APQ3/5/11)
│   │   │   ├── reaper_python.py    # REAPER 算法的原生 Python 实现
│   │   │   ├── shr.py              # 次谐波-谐波比 (SHR) 计算 (归一化)
│   │   │   ├── soe.py              # 激励强度 (SoE) 计算 (ZFF方法)
│   │   │   ├── spectral_batch.py   # 批量频谱参数计算 (H1-H4, A1-A3, H2K, H5K)
│   │   │   ├── spectral_slope.py   # 频谱倾斜计算
│   │   │   ├── lpc.py              # LPC 频谱计算核心算法
│   │   │   └── voicing.py          # 清浊音与静音检测 (基于ZCR和能量)
│   │   ├── alignment/          # 强制对齐 (MFA/Montreal Forced Aligner)
│   │   │   └── __init__.py
│   │   ├── articulatory/       # 发音运动数据处理 (EMA 等)
│   │   │   └── __init__.py
│   │   ├── egg/                # EGG (电声门图) 信号处理
        │   ├── __init__.py
        │   ├── analysis.py         # EGG 核心分析算法 (CQ, SQ, DECPA)
        │   └── inverse_filtering.py # 逆滤波算法 (CP-IF)
        ├── manipulation/       # 音频修改与合成
        │   ├── README.md
        │   ├── __init__.py
        │   ├── synthesis.py        # PSOLA 合成算法 (Praat/Parselmouth)
        │   └── batch_utils.py      # 批量变调与合成逻辑
        ├── perception/         # 感知实验数据处理（预留核心算法目录）
        │   └── __init__.py
        ├── signals/            # 基础数字信号处理 (DSP)
        │   ├── __init__.py
        │   └── filters.py          # 滤波器设计与应用 (低通/高通/带通)
        ├── spec2wav/           # 语谱图转音频 (Griffin-Lim)
        │   ├── __init__.py
        │   ├── common.py           # 通用工具 (重采样, dB转换)
        │   ├── griffin_lim.py      # Griffin-Lim 算法实现
        │   └── image_processing.py # 图像处理与加载
        ├── synthesis/          # 语音合成
        │   ├── __init__.py
        │   └── klatt/              # Klatt 参数化语音合成核心（纯算法/参数）
        │       ├── __init__.py
        │       ├── input_parser.py     # 元音序列解析与分段
        │       ├── klatt_config.py     # 默认参数与范围定义
        │       ├── smoothing_utils.py  # 平滑与过渡工具
        │       ├── spectral_filter.py  # 谱包络滤波组件
        │       └── tdklatt.py          # 时域 Klatt 合成核心
        └── transcription/      # 自动/手动标注处理
│   │       └── __init__.py
│   ├── gui/                    # [Outer Layer] 用户界面 (PyQt6)
│   │   ├── dialogs/            # 模态/非模态对话框
        │   ├── manipulation_dialogs.py # 变调相关对话框 (批量处理, 拐点编辑, 导入F0)
        │   ├── egg_batch_dialog.py # EGG 批量处理对话框
        │   └── settings_dialog.py  # 全局设置弹窗
        ├── resources/          # 静态资源 (图标, 图片, 前端页面资源)
        │   ├── articulatory_synth/  # 发音物理模拟器离线资源
        │   ├── ipa_trans/          # 普通话转 IPA 前端页面生成与产物
        │   │   ├── generate_ipa_website.py
        │   │   └── ipa_converter.html
        │   └── __init__.py
        ├── viewmodels/         # MVVM 模式的 ViewModel (可选)
        │   └── __init__.py
        ├── views/              # 复杂的视图布局
        │   └── __init__.py
        ├── widgets/            # 可复用 UI 组件
        │   ├── __init__.py
        │   ├── acoustic_widget.py          # 声学参数可视化组件
        │   ├── parameter_estimation_widget.py # 参数估计与批处理主组件
        │   ├── pitch_manipulation_widget.py # 变速变调 (基频实验室) 主组件
        │   ├── speech_synthesis_widget.py   # 语音合成实验室主组件（Klatt 参数编辑与联动视图）
│   │   │   ├── egg_widget.py       # EGG 分析主组件
│   │   │   ├── spec2wav_widget.py  # 语谱图转音频主组件
│   │   │   └── lpc_spectrum_widget.py # LPC 谱图主组件（波形选区 + TextGrid 联动）
        ├── workers/            # 后台工作线程
        │   ├── manipulation_workers.py # 批量变调处理线程
        │   └── egg_workers.py      # EGG 分析与加载线程
        ├── __init__.py
        ├── main_window.py      # 应用程序主窗口
        ├── styles.py           # QSS 样式表定义 (支持日间/夜间模式)
        └── utils.py            # GUI 专用工具 (主题应用, 音频播放)
│   ├── models/                 # [Inner Layer] 数据结构定义 (Pydantic/Dataclasses)
    │   ├── __init__.py         # 存放跨层共享的数据模型 (如 Config, AnalysisResult, LaunchResult)
    │   ├── acoustic_models.py  # PitchTrack：带真实时间坐标的基频轨迹
    │   ├── articulatory_models.py # 发音物理模拟器启动结果
    │   ├── config.py           # 配置模型定义
    │   ├── egg_models.py       # EGG 分析结果模型定义
    │   ├── ipa_models.py       # 普通话转 IPA 启动结果模型
    │   ├── lip_models.py       # 唇形提取启动结果模型
    │   ├── lpc_models.py       # LPC 配置与结果模型定义
    │   ├── perception_models.py # 感知实验启动结果模型
    │   └── spec2wav_models.py  # 语谱图转音频模型定义
    ├── services/               # [Middle Layer] 应用业务逻辑与 IO
│   │   ├── io/                 # 文件输入输出
│   │   │   ├── __init__.py
│   │   │   ├── excel.py            # Excel/CSV 导入导出
│   │   │   ├── lip.py              # 唇形数据文件 (.pkl) 读写
│   │   │   ├── textgrid.py         # TextGrid 标注文件解析与生成
│   │   │   └── wav.py              # 音频文件读写
│   │   ├── pipelines/          # 复杂处理流程
│   │   │   └── __init__.py
│   │   ├── __init__.py
        ├── acoustic_service.py # 声学参数分析服务 (串联 Core 算法与 IO)
        ├── articulatory_synth_service.py # 发音物理模拟器资源定位与启动
        ├── egg_service.py      # EGG 分析服务 (GCI/GOI 检测, CQ/SQ 计算, 逆滤波)
        ├── ipa_trans_service.py # 普通话转 IPA 服务（按需生成并打开前端页面）
        ├── lip_service.py      # 唇形提取服务（定位并打开外部项目入口）
        ├── lpc_service.py      # LPC 谱图服务（音频读取、TextGrid 解析、导出保存）
        ├── manipulation_service.py # 变速变调服务 (音频加载, 合成, 批处理)
        ├── perception_service.py # 感知实验服务（定位并打开外部 HTML）
        ├── settings_service.py # 配置管理服务 (单例模式, 持有运行时配置对象)
        └── spec2wav_service.py # 语谱图转音频服务
    ├── tests/                  # 单元测试与集成测试
│   │   ├── test_acoustic_time_alignment.py # F0 算法与时间网格回归测试
│   │   ├── test_acoustic_numerics.py # 插值、RMS、掩码回归测试
│   │   ├── test_resource_resolution.py # 开发态/_MEIPASS 资源测试
│   │   ├── test_version_consistency.py # 项目/运行时/EXE 版本一致性
│   │   ├── test_lip_service.py      # 唇形提取服务测试
│   │   ├── test_lpc_service.py      # LPC 服务测试
│   │   └── test_perception_service.py # 感知实验服务测试
│   ├── utils/                  # 通用工具库
│   │   └── __init__.py         # 日志, 装饰器, 辅助函数
│   ├── __init__.py
│   └── app.py                  # PyQt Application 初始化配置
├── ARCHITECTURE.md             # 架构规范文档
├── README.md                   # 项目说明文档
├── run.py                      # 程序启动与 PyInstaller 打包入口
├── run.spec                    # 可移植、版本化的 PyInstaller onefile 规格
├── pyproject.toml              # 项目依赖与构建配置
├── requirements.txt            # Python 依赖列表
└── settings.json               # 历史文件（已废弃，不再作为配置来源）
```

### 2.1 近期补充备注 (Updated 2026-03-15)

1.  **感知实验接入方式**:
    *   采用 `GUI -> API -> Services -> Models` 分层链路，不在 GUI 层直接调用 `webbrowser`。
    *   新增 `models/perception_models.py` 与 `services/perception_service.py`。
    *   API 统一从 `api/__init__.py` 暴露 `launch_perception_experiment()`。

2.  **外部前端项目打开策略**:
    *   感知实验优先在运行目录链路中查找 `perception_experiment/perception_experiment.html`。
    *   运行目录链路包含：EXE 所在目录、当前工作目录、开发仓库根目录等。
    *   支持通过环境变量 `PHONETIC_TOOLBOX_PERCEPTION_PROJECT_DIR` 覆盖目录。
    *   若默认文件缺失，服务层回退到目录内第一个 `*.html` 文件。

3.  **备注**:
    *   本文档仅补充当前已接入模块与目录映射，不涉及删除未审阅脚本或目录。

### 2.2 近期补充备注 (Updated 2026-03-18)

1.  **MFA 自动标注接入方式重构**:
    *   主程序入口采用 `main_window.py -> mfa_auto_alignment_dialog.py -> services/mfa_alignment_service.py` 的分层调用。
    *   外部批处理入口保持 `auto_alignment.bat -> mfa_auto_alignment_standalone.py -> mfa_auto_alignment_dialog.py`。
    *   `services/mfa_alignment_service.py` 负责运行 MFA 对齐流水线，同时保留定位与启动 `auto_alignment/auto_alignment.bat` 的能力。
    *   外部环境目录 `auto_alignment/` 仅要求保留 `env/` 与 `auto_alignment.bat`，原 `auto_alignment/app` 已可移除。

2.  **MFA 业务逻辑分层落位**:
    *   Core 层新增 `core/transcription/mfa_name_codec.py`，用于文件名编码/解码纯算法。
    *   Services 层新增 `services/pipelines/mfa_alignment_pipeline.py`，承接 MFA 对齐流程（拷贝、编码、对齐、解码回写）。
    *   Models 层新增 `MFAAlignmentRunResult`，用于承载运行态任务结果。

3.  **MFA 可视化界面整合**:
    *   GUI 层新增 `gui/dialogs/mfa_auto_alignment_dialog.py` 作为 MFA 参数输入与任务执行界面。
    *   新增 `gui/dialogs/mfa_auto_alignment_standalone.py` 作为外部批处理脚本入口。
    *   MFA 界面提供绿底白字小型“帮助”按钮，跳转 `Phonetic_Export/index.html` 的“8 自动标注”章节锚点。

4.  **外部批处理脚本规范**:
    *   `auto_alignment/auto_alignment.bat` 改为激活 `auto_alignment/env` 后启动仓库内 `phonetic_toolbox/gui/dialogs/mfa_auto_alignment_standalone.py`。
    *   该策略保证 MFA UI 与主项目风格、分层与代码规范一致，同时继续复用独立环境。

### 2.3 近期补充备注 (Updated 2026-03-18)

1.  **LPC 谱图接入方式**:
    *   主入口由 `gui/main_window.py` 的“LPC谱图”按钮进入 `gui/widgets/lpc_spectrum_widget.py`。
    *   GUI 层仅负责交互与绘图，不在界面层实现 LPC 核心算法。

2.  **LPC 分层落位**:
    *   Core 层新增 `core/acoustic/lpc.py`，提供 `compute_lpc_spectrum` 纯算法接口。
    *   Models 层新增 `models/lpc_models.py`，定义 `LPCSpectrumConfig` 与 `LPCSpectrumResult`。
    *   Services 层新增 `services/lpc_service.py`，承接音频加载、TextGrid 读取、标签提取与图片保存。

3.  **LPC 交互与导出约束**:
    *   波形交互采用“左键拖动平移、Shift+左键框选区间”的规则。
    *   若存在 TextGrid，导出图片文件名追加当前选区内标签文本（经文件名安全清洗）。
    *   导出图统一白底黑字，16:9 画幅，保存分辨率使用高 DPI。

### 2.4 近期补充备注 (Updated 2026-03-18)

1.  **语音合成页面链路补充**:
    *   主入口由 `gui/main_window.py` 的“语音合成”按钮进入 `gui/widgets/speech_synthesis_widget.py`。
    *   GUI 层负责参数曲线编辑与多视图交互，合成核心调用 `core/synthesis/klatt`。

2.  **Klatt 目录职责澄清**:
    *   `core/synthesis/klatt/` 仅保留核心算法与参数定义，不再存放 `*service.py` 命名文件。
    *   历史迁移产生的未被调用服务脚本已清理，避免无效模块滞留。

### 2.5 声学时间契约、资源定位与版本化打包 (Updated 2026-07-15)

1.  **显式声学时间轨迹**:
    *   Core 层使用 `models/acoustic_models.py::PitchTrack` 表达“采样时间 + 数值”，不得再假定第三方算法的第一个返回值位于 0 秒。
    *   Praat CC 必须调用 `Sound.to_pitch_cc()`；Praat AC 调用 `Sound.to_pitch()`，算法失败应显式上报，不得静默切换成另一算法。
    *   `AcousticAnalysisService` 负责把 Praat/REAPER 映射到 `np.arange(n_frames) * frameshift_ms / 1000` 的统一零基网格。Core 不承担跨算法轨迹对齐。
    *   网格外不外推；`NaN` 表示无声、无效或缺测。平滑不得跨越 `NaN` 间隙，也不得恢复被最终掩码排除的帧。

2.  **数值与导出边界**:
    *   `compute_rms` 返回线性时域 RMS；dB 音强由 `compute_energy` 表达，调用方不得混用量纲。
    *   频谱峰值细化使用经过解析测试的三点抛物线插值。
    *   Excel/CSV 写入错误必须传播到 Service 层；批处理由 Service 统一记录 `processed` 与 `failed`，IO 层不得吞异常。

3.  **资源可移植性**:
    *   禁止在 Python、HTML、CSS 或 JavaScript 中写入开发机盘符或仓库绝对路径。
    *   Python 根级资源使用 `get_resource_path()`、模块位置或 Service 的运行时目录解析器；PyInstaller onefile 优先支持 `sys._MEIPASS`。
    *   静态网页资源使用相对 URL。外部前端的目录发现、环境变量覆盖与打开动作属于 Services/API 职责。
    *   REAPER 二进制解析须覆盖显式路径、系统 PATH、模块目录和 `_MEIPASS`，同时保留纯 Python 回退。

4.  **版本与构建**:
    *   发布版本同时记录于 `pyproject.toml` 与 `phonetic_toolbox.__version__`，由自动测试保证一致。
    *   `run.spec` 必须以 `SPECPATH` 为项目根，禁止写死 checkout 路径；它从 `pyproject.toml` 读取版本并输出 `PhoneticToolbox_v<version>.exe`。
    *   打包前至少执行完整 pytest、`compileall`、GUI 主窗口构造冒烟和资源解析测试；打包后检查关键资源解包与 EXE 启动存活。

5.  **发音物理模拟器链路**:
    *   入口遵循 `MainWindow -> API -> ArticulatorySynthService -> ArticulatorySynthLaunchResult`。
    *   离线前端资源位于 `gui/resources/articulatory_synth/`；网页只处理交互与近似物理模型，不绕过 Service 实现平台路径发现。

## 3. models 文件夹的作用

`phonetic_toolbox/models/` 文件夹用于存放**数据模型 (Data Models)**。

### 为什么需要它？
在分层架构中，不同层之间需要传递数据。为了避免：
1.  传递散乱的字典 (`dict`)，导致键名拼写错误且无法自动补全。
2.  传递过多的单个参数 (`func(a, b, c, d, e...)`)。
3.  循环依赖 (Circular Imports) —— Core 层不能引用 GUI 定义的类，Services 层也不能引用 GUI。

我们使用 `models` 层作为所有层都可以引用的**公共数据定义层**。

### 应该写什么内容？
你应该在这里定义 `dataclass` 或 `Pydantic` 模型。例如：

1.  **配置对象**: 定义参数估计的配置结构。
2.  **结果对象**: 定义分析结果的数据结构。
3.  **显式轨迹对象**: 时间坐标不能可靠推导时，使用类似 `PitchTrack(times, values)` 的模型同时传递时间和值。

**示例 (`models/config.py`)**:
```python
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Optional

@dataclass
class AcousticConfig:
    min_f0: float = 60.0
    max_f0: float = 880.0
    frameshift_ms: float = 5.0
    # ... 其他配置

@dataclass
class AnalysisResult:
    time_axis: np.ndarray
    f0_praat: np.ndarray
    f0_reaper: Optional[np.ndarray] = None
    # ... 其他结果字段
    extras: Dict[str, np.ndarray] = field(default_factory=dict)
```

这样，GUI 层创建 `AcousticConfig` 对象传给 Services 层，Services 层调用 Core 层并返回 `AnalysisResult` 对象给 GUI 层显示。所有层都只依赖 `models`，解耦了逻辑。

## 4. 开发流程示例

假设我们要添加一个"计算音频时长"的功能：

1.  **Core 层**: 在 `core/signals/processing.py` 中编写 `get_duration(signal: np.ndarray, fs: int) -> float`。
2.  **Services 层**: 在 `services/io/audio.py` 中编写 `load_wav(path: str) -> (np.ndarray, int)`。
3.  **API 层**: 在 `api/facade.py` (或 `api/__init__.py`) 中暴露接口。
4.  **GUI 层**: 在按钮点击事件中，调用 `api.get_wav_duration(path)` 并显示结果。

## 5. 禁止事项

*   ❌ 禁止在 `core/` 中导入 `PyQt6`。
*   ❌ 禁止在 `core/` 函数中直接读取文件（应传入数据）。
*   ❌ 禁止在 `gui/` 中编写复杂的 `for` 循环进行数据处理。
*   ❌ 禁止在 `utils/` 中放入业务逻辑（Utils 应该是项目无关的通用工具）。
*   ❌ 禁止在源码或静态资源中写入开发机盘符、个人目录或仓库绝对路径。
*   ❌ 禁止通过截断/补齐数组替代具有真实时间坐标的跨算法对齐。

---
**违反上述规则的代码将被视为不合格代码，必须重构。**

## 6. 配置管理规则 (Updated 2026-03-07)

1.  **代码即配置 (Code as Configuration)**:
    *   所有声学参数配置必须通过 `phonetic_toolbox/models/config.py` 中定义的 `AcousticConfig` 类进行管理。
    *   **废弃** `settings.json` 文件。不再支持从外部 JSON 文件加载配置。
    *   修改默认参数必须直接修改 `phonetic_toolbox/models/config.py` 文件中的默认值。

2.  **运行时修改**:
    *   GUI 中的“设置”对话框仅修改**当前运行时**的内存配置对象。
    *   这些修改**不会**保存到磁盘，程序重启后将恢复为 `config.py` 中的代码默认值。
    *   `SettingsService` 现在是一个轻量级的配置持有者，不再负责文件 I/O。

3.  **禁止硬编码**:
    *   业务逻辑（Services/Core）不应包含硬编码的参数默认值。所有默认值应统一定义在 `models/config.py` 中。

4.  **单一数据源**:
    *   获取配置的唯一合法途径是调用 `SettingsService().get_config_object()` 或直接实例化 `AcousticConfig()`。
