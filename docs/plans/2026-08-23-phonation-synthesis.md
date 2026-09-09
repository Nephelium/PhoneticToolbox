# Phonation Synthesis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** 将独立基频/发声类型重合成工具按 PhoneticToolbox 分层架构迁入主页“发声类型合成”模块，并交付经过测试、视觉检查和 onefile 打包验证的版本。

**Architecture:** Models 保存默认参数与跨层结果；Core 仅处理数组和 DSP；Service 编排 WAV/F0/导出；后台 Worker 隔离长任务；GUI 只负责交互与展示。复用现有 Praat、REAPER、WAV IO、主题和主窗口生命周期，不依赖源 EXE、论文数据或开发机绝对路径。

**Tech Stack:** Python 3.9–3.12、NumPy、SciPy、Parselmouth、PyQt6、Matplotlib、pytest、PyInstaller；构建环境固定为 conda `phonetic_311`。

---

## Scope

- 保留原工具全部面向用户的分析、F0 编辑、三类连续统、双向批量生成、CSV/WAV 导出和参数帮助。
- 不迁移论文统计分析、原论文语料、历史输出和独立 EXE/spec。
- 新增文件采用 PhoneticToolbox 命名、数据模型、Service、Worker、主题和测试约定。
- 版本由当前未提交的 2.1.5 升至 2.1.6，并同步运行时版本、说明书和构建产物名称。
- 保留当前工作区全部既有改动，不执行 Git commit、push 或发布。

### Task 1: 模型与纯算法契约

**Files:**
- Create: `phonetic_toolbox/models/phonation_synthesis_models.py`
- Modify: `phonetic_toolbox/models/__init__.py`
- Create: `phonetic_toolbox/core/manipulation/phonation_synthesis.py`
- Modify: `phonetic_toolbox/core/manipulation/__init__.py`
- Create: `phonetic_toolbox/tests/test_phonation_synthesis_core.py`

1. 写失败测试：默认参数、F0/帧/脉冲窗校验、静音裁剪和 LPC 残差输出形状。
2. 运行：`conda run -n phonetic_311 python -m pytest phonetic_toolbox/tests/test_phonation_synthesis_core.py -q --no-cov`；预期因模块不存在而失败。
3. 实现枚举、配置和结果 dataclass；默认值与源工具一致。
4. 实现纯数组函数：窗函数、裁剪、LPC、残差、边界、脉冲、F0 采样/编辑、残差连续统和重合成。
5. 写三类连续统的端点与有限值测试；再次运行定向测试，预期通过。

### Task 2: Service、F0 后端与输出

**Files:**
- Create: `phonetic_toolbox/services/phonation_synthesis_service.py`
- Modify: `phonetic_toolbox/services/__init__.py`
- Modify: `phonetic_toolbox/api/__init__.py`
- Create: `phonetic_toolbox/tests/test_phonation_synthesis_service.py`

1. 写失败测试：临时 WAV 分析、缺失文件、F0 毫秒网格映射、单连续统输出、六组双向输出和 CSV 列名。
2. 运行 Service 定向测试，确认先失败。
3. 实现单声道读取、11025 Hz 重采样和临时 WAV 生命周期。
4. 复用 `compute_praat_f0_track()` 与 `compute_reaper_f0()`；按真实时间映射到毫秒网格，不写绝对路径。
5. 实现 `analyze_file_pair`、`generate_selected`、`generate_all`、`save_f0_csv`，并支持进度和取消回调。
6. 再次运行 Core + Service 测试，预期通过。

### Task 3: 后台 Worker 与 GUI

**Files:**
- Create: `phonetic_toolbox/gui/workers/phonation_synthesis_workers.py`
- Create: `phonetic_toolbox/gui/widgets/phonation_synthesis_widget.py`
- Create: `phonetic_toolbox/tests/test_phonation_synthesis_widget.py`

1. 写离屏失败测试：窗口可构造、默认参数正确、任务期间按钮禁用、关闭时请求取消。
2. 运行：`conda run -n phonetic_311 python -m pytest phonetic_toolbox/tests/test_phonation_synthesis_widget.py -q --no-cov`；预期先失败。
3. 实现分析 Worker 与生成 Worker，信号包含进度、结果、错误和取消；不得从线程直接操作控件。
4. 将源 GUI 改写为 `QWidget`：顶部操作区、参数组、F0 表格、Matplotlib 波形/F0 图、状态栏和参数说明。
5. 实现深浅主题、中文 Matplotlib 字体、参数变化失效状态、控制点编辑和安全关闭。
6. 再次运行 Widget 测试，预期通过。

### Task 4: 主页入口与生命周期

**Files:**
- Modify: `phonetic_toolbox/gui/main_window.py`
- Modify: `phonetic_toolbox/tests/test_phonation_synthesis_widget.py`

1. 写失败测试或静态断言：首页存在“发声类型合成”，处理器创建单实例窗口并传播主题。
2. 在主页按钮网格加入入口，新增 `on_phonation_synthesis()`，窗口键使用 `phonation`。
3. 在 `apply_theme()` 中向可见窗口调用 `set_theme()`；关闭主窗口时让子窗口安全取消任务。
4. 运行 GUI 定向测试和主窗口离屏构造冒烟，预期通过。

### Task 5: 文档、版本与打包契约

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `phonetic_toolbox/core/manipulation/README.md`
- Modify: `phonetic_toolbox/models/README.md`
- Modify: `phonetic_toolbox/services/README.md`
- Modify: `phonetic_toolbox/gui/README.md`
- Modify: `Phonetic_Export/index.html`
- Modify: `Phonetic_Export/PhoneticToolboxDoc.html`
- Modify: `pyproject.toml`
- Modify: `phonetic_toolbox/__init__.py`
- Modify: `phonetic_toolbox/tests/test_version_consistency.py`

1. 将用户功能、算法边界、输入输出、参数说明和常见错误写入说明书的新章节，并保持编辑器源与导出页一致。
2. 同步各层 README 与 `ARCHITECTURE.md` 的新增文件和依赖链。
3. 更新 README 功能概览及版本；同步 `pyproject.toml` 与 `phonetic_toolbox.__version__` 为 2.1.6。
4. 扩展静态测试，确保新 GUI/Service/Core 无开发机绝对路径，版本一致性继续通过。
5. 运行文档链接、章节锚点、UTF-8/BOM 和 `git diff --check` 检查。

### Task 6: 完整验证、视觉 QA 与 EXE

**Files:**
- Modify only if validation finds a defect in the files above.
- Create: `image/screenshots/15_phonation_synthesis.png` if the screenshot set is retained.
- Output: `dist/PhoneticToolbox_v2.1.6.exe`

1. 运行：`conda run -n phonetic_311 python -m compileall -q run.py phonetic_toolbox`；预期退出码 0。
2. 运行：`conda run -n phonetic_311 python -m pytest -q`；预期全部通过。
3. 运行离屏主窗口和发声类型合成窗口构造冒烟。
4. 启动源码 GUI，在高 DPI 下检查暗色/亮色、中文字体、窗口缩放、参数区、表格和绘图；发现问题先修复再重截。
5. 用源项目示例 WAV 分别验证 Parselmouth、REAPER、三类当前连续统与全部六组输出；检查步数、采样率、峰值、有限值和 CSV。
6. 按项目规则执行：`conda run -n phonetic_311 python -m PyInstaller run.spec --noconfirm --log-level WARN`。
7. 从非项目目录启动 `dist/PhoneticToolbox_v2.1.6.exe`，确认主页入口及新窗口可用，并验证 `_MEIPASS` 中 REAPER、说明书和新模块资源。
8. 关闭本次启动的程序，只清理本任务拥有的进程；记录 EXE 大小、SHA-256、测试和运行结果。

