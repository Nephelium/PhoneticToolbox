# Acoustic Correctness and Packaging Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** 修复声学参数时间对齐、算法选择、数值计算、导出错误处理和 PyInstaller 资源定位问题，同时保持现有 Pickle 工作流不变。

**Architecture:** 保留现有 GUI → Services → Core/Models 分层。Core 层返回带真实时间坐标的轨迹，Service 层负责映射到统一的零起点固定帧移网格；GUI 只接收明确的成功或失败结果。资源路径统一通过模块位置或 `get_resource_path` 解析，不依赖当前工作目录。

**Tech Stack:** Python 3.9–3.12、NumPy、SciPy、pandas、Parselmouth、PyQt6、pytest、PyInstaller。

---

## Scope

- 修复 Praat CC 静默退回 AC。
- 对 Praat、REAPER、共振峰、频谱帧使用显式时间坐标。
- 修复抛物线峰值插值、RMS 和平滑后掩码丢失。
- Excel 写入失败必须传播到批处理结果。
- 修复 REAPER、帮助文件、图标和 PyInstaller spec 的资源路径。
- 将依赖本机音频的伪测试改为确定性测试。
- 不修改 Pickle 的读写或兼容行为。
- 不修改用户现有的发音物理模拟器等未提交功能。
- 不执行 Git commit、push 或发布。

### Task 1: Praat F0 算法与时间轨迹

**Files:**
- Create: `phonetic_toolbox/models/acoustic_models.py`
- Modify: `phonetic_toolbox/core/acoustic/f0_praat.py`
- Modify: `phonetic_toolbox/core/acoustic/__init__.py`
- Create: `phonetic_toolbox/tests/test_acoustic_time_alignment.py`

1. 写测试，断言 `method="cc"` 调用 `Sound.to_pitch_cc()`，而不是 `Sound.to_pitch()`。
2. 写测试，断言轨迹保留 `pitch.xs()` 返回的真实时间坐标。
3. 运行：`python -m pytest phonetic_toolbox/tests/test_acoustic_time_alignment.py -v`；预期先失败。
4. 新增 `PitchTrack(times, values)` 与 `compute_praat_f0_track()`；保留 `compute_praat_f0()` 的数组兼容接口。
5. 再次运行定向测试；预期通过。

### Task 2: 统一声学时间网格

**Files:**
- Modify: `phonetic_toolbox/core/acoustic/common.py`
- Modify: `phonetic_toolbox/services/acoustic_service.py`
- Modify: `phonetic_toolbox/tests/test_acoustic_time_alignment.py`

1. 写测试，使用已知源时间和值验证无外推插值。
2. 写测试，验证 F0 首个有限值出现在其真实时间对应的网格索引。
3. 写测试，验证 `segment_for_frame(..., k=5, frameshift_ms=5)` 的中心为 25 ms。
4. 运行定向测试确认失败。
5. Service 层建立 `np.arange(target_len) * frame_step` 的统一网格，对 Praat/REAPER 轨迹按时间插值，网格外填 `NaN`。
6. 将频谱片段中心统一为 `k * frame_shift`。
7. 运行定向测试确认通过。

### Task 3: 数值计算与掩码

**Files:**
- Modify: `phonetic_toolbox/core/acoustic/spectral_batch.py`
- Modify: `phonetic_toolbox/core/acoustic/energy.py`
- Modify: `phonetic_toolbox/services/acoustic_service.py`
- Create: `phonetic_toolbox/tests/test_acoustic_numerics.py`

1. 写解析抛物线测试：峰位置 1.25 必须恢复为 1.25。
2. 写单位正弦 RMS 测试：结果应接近 `1/sqrt(2)`。
3. 写平滑测试：原始 `NaN` 掩码在平滑后仍为 `NaN`，且不同连续区间互不污染。
4. 运行定向测试确认失败。
5. 修复插值公式；直接从时域窗口计算 RMS；提取保留缺失值的分段平滑帮助函数。
6. 运行定向测试确认通过。

### Task 4: 导出错误传播

**Files:**
- Modify: `phonetic_toolbox/services/io/excel.py`
- Modify: `phonetic_toolbox/services/acoustic_service.py`
- Modify: `phonetic_toolbox/tests/test_acoustic_batch_result.py`

1. 写测试，模拟 `DataFrame.to_excel()` 抛出文件占用错误。
2. 断言单文件保存抛出异常，批处理将文件计入 `failed` 而非 `processed`。
3. 运行定向测试确认失败。
4. 移除 `save_excel()` 中的吞异常逻辑，让 Service 统一记录并传播错误。
5. 运行定向测试确认通过。

### Task 5: EXE 资源与路径

**Files:**
- Modify: `phonetic_toolbox/core/acoustic/f0_reaper.py`
- Modify: `phonetic_toolbox/gui/widgets/parameter_estimation_widget.py`
- Modify: `phonetic_toolbox/gui/dialogs/parameter_tools_dialog.py`
- Modify: `run.spec`
- Create: `phonetic_toolbox/tests/test_resource_resolution.py`

1. 写测试，模拟非项目工作目录和 `_MEIPASS`，断言能够定位 REAPER。
2. 写测试或静态断言，确保帮助与图标通过 `get_resource_path()` 解析。
3. 运行定向测试确认失败。
4. REAPER 候选路径加入模块目录及 `_MEIPASS`；GUI 移除 D 盘硬编码。
5. spec 使用自身目录构造资源路径并去除重复资源项；保留完整 `phonetic_toolbox` data，因为 MFA 外部运行时当前依赖从 `_MEIPASS` 复制源码。
6. 运行定向测试确认通过。

### Task 6: 测试清理与完整验证

**Files:**
- Modify: `phonetic_toolbox/tests/test_fft_resolution.py`
- Modify: `phonetic_toolbox/tests/test_spectral_consistency.py`

1. 移除依赖个人桌面目录且无断言的测试路径。
2. 将可重复的解析测试并入数值测试；需要真实语料的脚本保留为 benchmark，但不冒充 pytest 成功。
3. 运行：`python -m compileall -q run.py phonetic_toolbox`。
4. 运行：`python -m pytest -q`。
5. 运行：`python -m pytest --cov=phonetic_toolbox --cov-report=term`，记录覆盖率，不在本次强设阈值。
6. 检查：`git diff --check` 与 `git status --short`，确认未覆盖既有用户修改。

---

## 实施结果（2026-07-15）

- 状态：Completed。
- Python 3.11.9（项目支持范围）：63 项测试全部通过，总覆盖率 15%。
- Python 3.13.12（额外兼容性检查）：60 项测试全部通过，总覆盖率 15%；该版本不在项目声明的支持范围内。
- `compileall`、`run.spec` 语法检查、GUI 入口导入和 `git diff --check` 均通过。
- 版本升级为 2.1.2，并生成 `dist/PhoneticToolbox_v2.1.2.exe`；旧版 EXE 未被覆盖。
- 新 EXE 已通过非项目工作目录启动存活测试，帮助页、REAPER、IPA 与感知实验资源均确认解包存在。
- 未修改 Pickle 工作流，未执行 Git commit、push 或发布。
