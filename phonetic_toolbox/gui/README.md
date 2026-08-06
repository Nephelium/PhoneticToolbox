# 用户界面层 (User Interface Layer)

本目录 (`phonetic_toolbox/gui`) 包含了应用程序的**最外层 (Outer Layer)** 代码。该层基于 **PyQt6** 框架构建，负责与用户交互、显示数据以及调用下层服务 (`services`) 执行业务逻辑。

## 目录结构

| 模块/包 | 主要类/功能 | 说明 |
| :--- | :--- | :--- |
| **`main_window.py`** | `MainWindow` | **主窗口**。负责程序入口、主菜单导航、子窗口管理、版本检查以及全局主题切换；发音物理模拟等纯前端工具通过 API 启动。 |
| **`styles.py`** | - | **样式表定义**。包含 QSS (Qt Style Sheets) 代码，定义了深色 (`DARK`) 和浅色 (`LIGHT`) 主题的视觉风格。 |
| **`dialogs/`** | - | **对话框子模块**。 |
| ├── `settings_dialog.py` | `SettingsDialog` | **全局设置对话框**。提供图形化界面修改当前运行时的声学参数 (如 F0 范围、静音阈值等)。永久默认值需在 `models/config.py` 中修改。 |
| ├── `manipulation_dialogs.py` | `BatchProcessorDialog` 等 | **变调相关对话框**。包括批量处理器、基频序列导入、拐点编辑等对话框。 |
| **`widgets/`** | - | **UI 组件子模块**。包含可复用的功能性 Widget。 |
| ├── `parameter_estimation_widget.py` | `ParameterEstimationWidget` | **参数估计主界面**。支持批量处理、文件列表管理、Matplotlib 波形/语谱图绘制以及后台任务调度。 |
| ├── `acoustic_widget.py` | `AcousticWidget` | **单文件分析组件**。提供简单的单文件选择、参数调整、分析运行及结果表格显示。 |
| ├── `parameter_display_widget.py` | `ParameterDisplayWidget` | **参数显示组件**。用于可视化分析结果 (如绘制 F0 曲线、共振峰轨迹等)。 |
| ├── `speech_synthesis_widget.py` | `SpeechSynthesisWidget` | **语音合成实验室**。提供 Klatt 参数曲线编辑、实时试听、频谱/波形联动与参数导出。 |
| ├── `pitch_manipulation_widget.py` | `PitchManipulationWidget` | **基频实验室**。提供可视化的基频编辑、音频合成、播放及批量生成功能。 |
| ├── `egg_widget.py` | `EGGWidget` | **EGG 分析**。提供 EGG 波形/语谱图显示、事件检测可视化 (GCI/GOI)、逆滤波及批量处理。 |
| ├── `spec2wav_widget.py` | `Spec2WavWidget` | **语谱图转音频**。提供图像校正、Griffin-Lim 重建、试听与导出。 |
| └── `lpc_spectrum_widget.py` | `LPCSpectrumWidget` | **LPC 谱图**。提供波形缩放/平移、Shift 选区、TextGrid 层级切换、LPC 曲线显示与图片导出。 |
| **`resources/`** | - | **离线前端资源**。包含 IPA、感知实验和发音物理模拟器页面；打包后从 `_MEIPASS` 或运行目录解析。 |
| **`workers/`** | - | **后台工作线程**。 |
| └── `manipulation_workers.py` | `BatchProcessorWorker` | **批量变调线程**。在后台执行耗时的批量音频生成任务。 |

## 详细功能说明

### 1. MainWindow (主窗口)

`MainWindow` 是用户启动程序后看到的第一个界面。
*   **导航中心**: 采用网格布局 (Grid Layout) 显示功能入口按钮 (如"参数估计"、"参数显示"等)。
*   **窗口管理**: 维护 `self.sub_windows` 字典，防止子窗口被垃圾回收，并支持窗口激活/置顶。
*   **主题管理**: 监听主题切换按钮，调用 `apply_theme` 方法动态更新全局 `QApplication` 和当前窗口的样式表。
*   **LPC 入口**: 提供“LPC谱图”入口并管理 `LPCSpectrumWidget` 子窗口生命周期。
*   **版本显示**: 优先读取项目版本，打包环境缺少 `pyproject.toml` 时回退到 `phonetic_toolbox.__version__`。
*   **前端工具入口**: IPA、感知实验和发音物理模拟器经 API/Service 返回结构化启动结果，GUI 只负责显示失败信息。

### 2. ParameterEstimationWidget (参数估计)

这是本工具最常用的核心界面，用于批量处理音频数据。
*   **多线程架构**: 使用 `QThread` (`PEWorker`) 将耗时的声学分析任务放入后台线程执行，防止界面卡死 (ANR)。
*   **实时反馈**: 通过 `pyqtSignal` 信号机制，将后台处理进度和日志实时反馈到 UI 进度条和状态栏。
*   **集成绘图**: 嵌入了 `matplotlib` 的 `FigureCanvasQTAgg`，支持在 PyQt 界面中直接绘制波形图和语谱图。
*   **服务调用**: 内部实例化 `AcousticAnalysisService` 来执行实际的分析逻辑。

### 3. SpeechSynthesisWidget (语音合成实验室)

该组件用于参数化语音合成与可视化回读分析。
*   **参数轨迹编辑**: 支持 F0/F1-F5 及 AV/HNR/Jitter/Shimmer 等参数曲线编辑与局部重置。
*   **多视图联动**: 参数曲线、波形、语谱图共用时间轴联动缩放/拖拽。
*   **分段合成**: 支持按元音序列/空格分段进行合成并拼接，避免无效段能量突刺。
*   **架构落位**: GUI 负责交互；合成核心在 `core/synthesis/klatt`，外部程序与文件级处理由 Services 层承接。

### 4. PitchManipulationWidget (基频实验室)

这是一个功能丰富的音频编辑器组件，移植自 `changeF0` 项目。
*   **可视化编辑**: 支持通过鼠标直接在画布上修改基频曲线。
*   **实时合成**: 修改后可立即合成音频并播放。
*   **历史对比**: 自动记录并绘制多次修改的历史曲线，方便对比。
*   **批量工具**: 集成了批量生成线性变调序列的工具。

### 5. EGGWidget (EGG 分析)

提供了一套完整的 EGG 信号分析工作流。
*   **多视图联动**: 同时展示语谱图、CQ/SQ 轨迹、音频波形和 EGG 波形。
*   **交互式缩放**: 支持滚轮缩放查看微观波形 (Zoom View)，并可拖动查看不同时间段。
*   **事件可视化**: 在波形图上直观标记 GCI (声门闭合) 和 GOI (声门开放) 时刻。
*   **高级功能**: 集成了 **逆滤波 (Inverse Filtering)** 和 **批量处理** 功能。
*   **主题适配**: 完美支持日间/夜间模式切换，自动调整波形颜色。

### 6. LPCSpectrumWidget (LPC 谱图)

该组件用于单文件 LPC 谱包络分析与导出。
*   **交互方式**: 支持滚轮缩放、左键平移、`Shift + 左键` 选区，便于在局部时段进行分析。
*   **标注联动**: 可读取同名 TextGrid 并切换层级，仅显示当前层级标注内容。
*   **处理流程**: 调用 `LPCSpectrumService` 完成音频读取、LPC 计算、标签提取与导图保存。
*   **主题与导出**: 界面适配深浅主题；导出图使用白底黑字并以高 DPI 保存，便于论文与报告使用。

### 7. Spec2WavWidget (语谱图转音频)

这是一个创新的逆向工程工具，允许用户从论文或课件中的语谱图图像恢复出可听的音频。
*   **屏幕截图工具**: 内置截图与四点透视校正工具，可从歪斜的翻拍照片中提取标准语谱图。
*   **Griffin-Lim 重建**: 调用 Core 层的 Griffin-Lim 算法，迭代重建相位信息。
*   **实时对比**: 生成音频后自动绘制重构的语谱图，方便与原图对比验证效果。
*   **音频导出**: 支持将恢复的音频导出为 WAV 文件。

### 8. SettingsDialog (设置对话框)

提供了一个表单界面来修改全局配置。
*   **数据绑定**: 在打开时通过 `SettingsService` 加载当前配置到控件 (SpinBox, CheckBox)。
*   **运行时修改**: 点击保存时，将控件的值写回 `SettingsService` 并立即生效。**注意：GUI 修改仅在当前程序运行期间有效，不会保存到磁盘。** 如需修改永久默认值，请直接编辑 `phonetic_toolbox/models/config.py` 文件。
*   **参数校验**: 利用 UI 控件 (如 `QDoubleSpinBox`) 的 `Range` 属性限制用户输入的合法范围。

## 样式系统 (Styling)

本项目采用类似 CSS 的 **QSS (Qt Style Sheets)** 进行界面美化。`styles.py` 中定义了常量：

*   **`DARK_MAIN_STYLESHEET`**: 主页专用的深色样式 (大按钮、特殊背景)。
*   **`GLOBAL_DARK_STYLESHEET`**: 全局通用的深色样式 (适用于对话框、子窗口，配色更标准)。
*   **`LIGHT_MAIN_STYLESHEET`**: 主页专用的浅色样式。
*   **`GLOBAL_LIGHT_STYLESHEET`**: 全局通用的浅色样式。

## 资源路径与打包约定

1.  Python 界面打开项目根级资源（图标、`Phonetic_Export` 等）时，使用 `phonetic_toolbox.utils.get_resource_path()` 或明确的模块/运行时目录解析器。
2.  禁止把开发机盘符、仓库绝对路径或当前工作目录假定写入 GUI 代码。
3.  `resources/` 内的静态 HTML/CSS/JS 只使用相对 URL；需要寻找运行态目录的逻辑放入 Service，不在网页或 Widget 中复制平台判断。
4.  PyInstaller onefile 运行时优先解析 `sys._MEIPASS`，同时保留 EXE 同目录外置资源的兼容能力。
5.  帮助入口统一指向 `Phonetic_Export/index.html`，各模块只设置自己的章节锚点。

## 开发规范 (GUI Layer)

1.  **保持逻辑轻量**: GUI 层**严禁**包含复杂的声学计算逻辑。所有计算必须封装在 `core` 或 `services` 层，GUI 仅负责调用和显示。
2.  **防卡死**: 任何耗时超过 100ms 的操作 (如文件读写、分析计算) **必须** 放入 `QThread` 或使用 `QTimer` 异步执行。
3.  **异常处理**: 必须捕获 Service 层抛出的异常，并使用 `QMessageBox` 友好地提示用户，而不是让程序崩溃。
4.  **样式分离**: 尽量不要在 Python 代码中硬编码颜色 (`widget.setStyleSheet("color: red")`)，应统一在 `styles.py` 中定义或使用 ObjectName 选择器。
5.  **路径可移植**: 新增或修改资源入口时必须同时覆盖开发态、非项目工作目录和 PyInstaller `_MEIPASS` 场景，并补充对应测试。
