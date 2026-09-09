# P03 科研行为基线报告

2026-09-09。状态：verified，限定为 Windows 原 v2 独立捕获与重复性基线；v3 算法迁移和完整产品验证尚未开始。

## 实际完成

最终采集 `output/validation/p03/20260909-165413/capture-summary.json`：14 个案例各两次独立进程，28 次捕获的科学字段一致；13 类正常返回，1 类损坏 WAV 稳定报错。7 个早期声学公开结果与补齐 conda DLL 搜索路径后的重跑结果整个 JSON 相同；最终 EGG 结果来自统一启动条件，不采用早期崩溃结果。

公开 8 例覆盖持续谐波、分段谐波/静音、纯静音、纯正弦、双声道形态、极短/空/损坏输入。私有 6 例含 4 个声学输入（其中 2 个含 TextGrid）与 2 个用户确认声道顺序的 EGG 输入。全部 input hash、采样率、长度、声道、config、实际调用和结果均记录；公开清单不包含私有路径、标签正文或数值曲线。

EGG-01 为 44100 Hz FLOAT 双声道，646699 帧，左 EGG/右音频；EGG-05 为 44100 Hz PCM16 双声道，1006848 帧，左音频/右 EGG。两种选择均与原输入逐样本精确一致。只选择这两例作全量基线，2–4.wav 未全量分析。

声学默认全参数路径输出 79 列：时间列与 78 个参数。80 键目录中覆盖 76 键，缺 4 个唇形键；额外存在 SOE_pF0/SOE_rF0。真实 TextGrid 两例各追加 2 列。采样率覆盖 16000、22050、44100 Hz。

## 来源、后端与导出

- 原 v2 EXE SHA-256：`40c807cd9a58da11d1e87e805f9ea84cdb515b8998f99588e9ae788dde218cdd`。30 个相关归档模块标准化代码全部匹配原源码；EXE 内 REAPER 与原环境显式二进制完全相同。
- EXE 提取的 AcousticConfig/EGGConfig 默认值与原源码精确相同；提取的声学服务代码在原依赖环境对合成谐波产生与原服务逐值精确相同的 DataFrame。证据在 `output/validation/p03/executable/module-audit.json` 和本轮 `probe-*/probe-summary.json`。
- 非静音的 7 个声学案例中，REAPER 原生进程实际退出 0；其二进制 SHA 为 `279fecc82ed0a49b0277b114270771d7670299068e849058b392672825981824`。静音、空和单点输入的 REAPER 退出 1，原服务仍返回，不把它们标成原生后端成功。
- 7 个非静音声学案例的 WM jitter/shimmer 输入轨迹与 IRAPT 返回的有效点精确对应。静音/空/单点没有可用 WM 轨迹。本轮未覆盖“IRAPT 失败但 Praat 回退成功”分支。
- 合成样例实际导出 `.xlsx` 与 `.ptb.sqlite` 两个新文件，均回读 160 行 × 79 列；列名、时间、缺失 mask 精确一致，数值以 rtol=atol=1e-12 对照通过。没有操作用户现存数据库。

## 已知旧行为，未在本轮修正

| 编号 | 证据与影响 | 后续处理 |
| --- | --- | --- |
| P03-K01 | 44100/22050 Hz 输入的 AnalysisResult.sampling_rate 仍为默认 16000；真实 WAV 头另有记录。 | 迁移原样基线与元数据修正分开。 |
| P03-K02 | 440 Hz 正弦的 REAPER 最终有限 F0 中位数为 62.745098 Hz（当前默认 60–880 Hz）；本轮只记录算法输出。 | 不能把稳定重复当作正确基频或精度真值。 |
| P03-K03 | 空、单采样点 WAV 返回 0 行对象，而非顶层失败；内部 REAPER 失败不传播。 | 正式输入验证/错误协议另行明确。 |
| P03-K04 | EGG file_duration 取末采样点 `(N-1)/fs`，不是容器持续时间 `N/fs`。 | 秒与采样点终点语义单独统一。 |
| P03-K05 | CQ/SQ 请求 0–0.5 s 时，SYN-EGG 返回时间末点 0.5858503401360544 s，包含缓冲区；服务也对已滤波信号再做片段滤波。 | 记录现状，ROI 截取/滤波修正需单独审阅。 |
| P03-K06 | 旧输出无逐点缺失原因，目录参数与实际输出键数不同。 | 不编造 NaN 原因、不静默删除 SOE 或唇形功能。 |

首次 EGG 子进程异常码 `0xc06d007f` 位于 SciPy lstsq/detrend。仅补齐子进程原 conda DLL 路径即恢复；此为采集启动条件问题，没有修改旧环境或替换算法。FLOAT WAV 可读，其未知非数据 chunk 警告原样保留。

## 验证与限制

实际命令：`.venv/v3-dev/Scripts/python.exe -X utf8 scripts/capture_v2_baseline.py --include-egg`；`.venv/v3-dev/Scripts/python.exe -m pytest -c tests/pytest.ini tests/parity -q`：11 passed。覆盖黄金 hash、配方字节、字段覆盖、严格时间/缺失/shape 扰动、旧边界行为及双声道映射。配方毫秒网格初次断言因浮点运算顺序不同失败，按定义改为整数毫秒再除以 1000，没有扩大容差。

EXE 审计命令：`.venv/p01-standalone/Scripts/python.exe -X utf8 scripts/inspect_v2_executable.py`，30 模块匹配。`scripts/probe_v2_baseline.py` 在原 Python 使用 `-B -X utf8 -X faulthandler` 和原环境子进程 PATH 运行，退出 0。详细运行约束见 [协议](../baseline/capture-protocol.md)。

原 v2 保存性检查：427 个基线文件 hash、HEAD、暂存区 hash、Git 状态、环境位置和已安装包版本元数据均与本轮开始一致。每个实际输入及关联 TextGrid 的采集前后 hash 一致。检查仅写入本轮 context-after，不覆盖 P01/P02 证据；没有新增第三方依赖或改动来源许可结论。

`.venv/v3-dev/Scripts/python.exe -X utf8 scripts/validate_docs.py`：122 个文件、282 条来源、32 个任务，当前错误 0。另报告 36 个原封保留历史文档中的旧链接，不混入本轮通过项；`git diff --check` 通过。

尚未验证：完整原 EXE 的 GUI/冻结运行时、v3 算法 parity、跨平台数值、专家标注的生理事件真值、确认的真实持续元音、唇形数据、EGG 其他高级分析、用户现存导出表的配置来源。黄金基线冻结的是旧行为，不能证明所有科学指标正确。后续 P04 可建设公共 UI；大规模算法迁移须依此基线逐模块检查，涉及已知问题的修正单列。
