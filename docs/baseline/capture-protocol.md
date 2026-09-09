# P03 基线捕获协议

2026-09-09。范围为 M01 声学服务与 M03 EGG 服务的迁移前行为基准。生产包不能导入旧项目；只有本协议下的测试子进程可只读导入原 v2，见 [ADR-014](../decisions/ADR.md)。

## 独立来源与运行条件

调度器使用 v3 的独立 Python；算法子进程使用原 conda `phonetic_311`（Python 3.11.14）。实际依赖版本、所有已导入 `phonetic_toolbox` 模块的 SHA-256 都在每个捕获结果的 producer 中。输入 hash、采样率、样本数、声道数、数据类型、完整 dataclass 配置、调用集合、关键函数返回值和原生程序退出码一并保存。

子进程以 `-B -X utf8 -X faulthandler` 启动，工作目录、TEMP/TMP 指向当前输出目录；PATH 仅在该子进程前置原环境、Library/bin、Scripts。没有激活或修改系统环境，没有安装/升级原环境依赖。首次缺失 DLL 路径的 EGG 采集崩溃已留存，不能作为正常数值基准；最终全量重跑采用统一路径。

默认声学参数来自原代码 AcousticConfig，而非假定等于用户当前 GUI 设置。显式给出原环境 REAPER 路径，并在公开结果中规范为 `<v2>/phonetic_toolbox/core/acoustic/reaper.exe`。记录请求配置和实际执行分别是什么；进程返回失败、空结果、NaN 均原样保留。

## 输入、隐私与声道

- [清单](../../tests/fixtures/manifest.json)：8 个公开合成案例与 6 个私有案例的匿名 ID、hash、音频头和选择规则。
- [生成配方](../../scripts/baseline_support.py)：120 Hz 谐波和分段谐波、静音、440 Hz 正弦、双声道解析信号、单采样点、零采样点和损坏 WAV。`SYN-VOWEL` 只是用例名，其信号没有自然元音共振峰模型；`SYN-EGG` 也不是经验证的生理接触模型。
- 私有 WAV、TextGrid 内容、个人路径、实际数值曲线只在被忽略的本地证据/output 中。两个真实 TextGrid 案例各含两层，已走原解析与对齐流程。嘎裂分类仅来自候选来源，尚无逐帧专家金标准；尚缺专门确认的真实持续元音与唇形配套输入。
- 井井确认 `1–4.wav` 左 EGG/右音频，`牧歌.wav` 左音频/右 EGG。本轮选择 EGG-01（新 1.wav，FLOAT）与 EGG-05（牧歌，PCM16），分别设置 `flip_channels=false/true`。2–4.wav 完成双声道头检查，未作全量数值捕获。旧 LOCAL-04 单声道记录不参与 EGG 基线。
- 原文件不交换、不覆盖、不重存。额外探针将原服务选出的两个数组与独立按声道归一化的输入逐样本精确比较，检查的是通道选择，不能替代电极极性或 GCI 生理真值验证。

## 比较和冻结

每个案例启动两个独立原环境进程，比较 scientific 字段；校验源音频/关联 TextGrid 在两轮前后 hash 不变。状态 `returned` 表示旧服务返回，不能解释为指标全部有效。损坏 WAV 的 `error` 是预定基准行为。

[逐参数约定](../../tests/fixtures/parameter-contract.json) 覆盖目录中的 80 键和实际额外输出的两个 SOE 键。76 个目录键加 2 个 SOE 已捕获，4 个唇形键缺输入。78 个参数加 Time_s 共 79 列；有 TextGrid 时另外追加标签列。

- 每个有限指标采用 `abs(a-b) <= max(1e-10, 1e-7 * max(abs(a),abs(b)))`；这是同平台重复性检查阈值，尚不是跨平台或科学准确度容差。
- 字段、列表长度、shape、整数、布尔值、时间轴、事件时间、声道规则、实际后端身份、NaN/Inf 掩码精确一致。毫秒网格保留旧代码的 `i * 5 / 1000` 运算顺序，不用浮点乘法重排后的小差异扩大容差。
- 数值数组以 values 和 nonfinite 双字段表达；0=有限、1=NaN、2=正无穷、3=负无穷。单独的 NaN 原因无法从旧结果倒推，记录 legacy_unspecified；silence/voiced mask 另存，不伪造每点缺失原因。
- 大于 10000 点的 EGG 预处理数组保存 shape、dtype、有限计数、首末点和精确字节 SHA-256。这支持同 Windows/同依赖下的重复对照，不宣称支持跨平台近似比较。
- 只有显式 `--freeze-public` 才冻结公开数值；已有黄金文件内容不同即拒绝覆盖。常规 pytest 不运行 v3 算法生成 expected，也不自动更新黄金文件。未来行为修复需单独审阅和版本化基准。

## 复现命令

先确保忽略的 `docs/baseline/local-evidence.json` 与本机语料对应。使用 `desktop/experiments/capture_context.py` 的只读 `capture()` 函数，把新结果写到 `output/validation/p03/context-before.json`，不要覆盖 P01/P02 证据。确认声道文件 `output/validation/p03/confirmed-egg.json` 使用本机绝对路径，其 cases 项需含 case_id、input、flip_channels 和 channel_evidence。

在项目根目录使用 PowerShell：

```powershell
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/capture_v2_baseline.py --include-egg
& '.venv/v3-dev/Scripts/python.exe' -m pytest -c tests/pytest.ini tests/parity -q
& '.venv/p01-standalone/Scripts/python.exe' -X utf8 scripts/inspect_v2_executable.py
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/build_p03_catalog.py
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/validate_docs.py
```

第一次冻结需在审查双轮结果后添加 `--freeze-public`；复核现有捕获应使用保存的结果调用 `freeze()`，无需重复计算。EXE 探针 `scripts/probe_v2_baseline.py` 必须用原解释器 `-B` 运行，按调度器同样设置子进程 PATH/TEMP/TMP，并传入尚不存在的 `output/validation/p03/probe-*` 目录。它只向新目录导出合成结果（xlsx 与配套 ptb.sqlite）；不读取或迁移用户数据库。

## EXE、来源与验收界限

归档审计比较 EXE 内 30 个相关模块与原源码编译结果的标准化字节码，保留操作码、常量、参数和异常表，只忽略编译文件名及行号表。另执行 EXE 提取的配置/声学服务代码，与原服务结果对照；依赖仍来自原 conda 环境。这不是完整冻结 EXE 的 GUI、打包运行时或交互测试。

没有引入新第三方代码、数据或依赖；复用 [来源登记](../../third_party/source-registry.json) 中已有原算法与 P01 PyInstaller 工具。原始版本未明、许可未明的条目保持未决。新合成波形为本项目解析配方，不声称来自论文或自然语料。实际验证及缺陷边界见 [P03 报告](../testing/p03-baseline-report.md)。
