# M06 来源与操作映射

**2026-10-04 M06-R5：** 已按井井要求移除本模块 WORLD/PSOLA/Harvest，恢复 Klatt 单一路径。源文件选择移到左侧首项，加载与提取分开；元音表紧凑化，复制接公共剪贴板。当前范围见 [R5 报告](../../testing/2026-10-04-m06-r5-report.md)。下文旧检查点按日期读取。

**2026-10-04 M06-R4：** 新增可选 WORLD/PSOLA 重合成及 Harvest F0，保留 Klatt。分别记 `m06-world/1`、`m06-psola/1`、`m06-natural-extract/1`。新增 `SRC-PYWORLD`/`SRC-WORLD`，保留独立 `SRC-PRAAT` 和 `SRC-REAPER` 来源。源哈希、种子、谱/非周期单位、数字零暂停恢复和四文件导出见 [ADR](../../decisions/ADR-M06-R4.md)；Windows 开发态及 WSL 纯核心的限定验证见 [R4 报告](../../testing/2026-10-04-m06-r4-report.md)。以下旧版本的算法及迁移结论按日期读取，不作为新路径的计算说明。

**2026-10-04 M06-R3：** F03/F04改用横排窗长和模块底部完整共享播放条，F05增加已有Praat CC/AC及原生REAPER选择，保留元音规则和生成操作。提取改用真实帧时间并修复APQ5百分数/内部比例，单列 `m06-extract/2`；非旧V2逐位等价。历史文件保留，旧m06/2缺少算法字段补CC。见[核查](../../references/m06-r3-resynthesis-audit.md)和[ADR](../../decisions/ADR-M06-R3.md)。

**2026-10-02 更新：** [M06-R2](../../testing/2026-10-02-m06-r2-report.md) 已获授权直接修复科研行为。新增源校准与 AH、AV 0–80、20 kHz 内部率、声源关闭/噪声 FIR，移除归一化/AGC 与 HNR→AH 映射，预设保持 F0 形状。配置 m06/2，计算 klatt/2；旧逐位断言保留作为差异证据。以下 2026-09-27 记录是原样迁移的历史状态，不能作为当前计算说明。

日期：2026-09-27。当前直接来源为相邻 `../PhoneticToolbox_v2`，未修改原文件或研究语料。对应原说明书 `Phonetic_Export/index.html` 4.1–4.5。状态按 [验收报告](../../testing/m06-report.md) 的平台范围读取。

## 六组功能逐项对应

| 组 | V2 实际入口 | V3 入口与正常证据 | 错误/边界证据 |
| --- | --- | --- | --- |
| F01 基础输入 | widget 时长/淡入淡出/平滑/F0 范围，`generate_vowels`，input_parser | 页面基础设置；23 参数默认目录；独立六场景生成/数组/合成比较 | 空/非法 IPA 定位，0/NaN 时长，反向 F0 范围；未应用输入禁止提交 |
| F02 曲线编辑 | `ParameterCurve`、绘图鼠标事件、`_apply_param_input`、`reset_view_ranges`、`clear_all_params` | CurveEditor 与 state：Shift 绘制、Ctrl 擦除、平移/缩放；常数覆盖、逗号插值、分号分段、重置范围、清空确认 | Override 锁定编辑；非法值拒绝；图外绘制限于时域；原曲线保留在完整 CSV |
| F03 视图 | `_on_spectrogram_button_clicked`、AudioPanel scipy spectrogram | 共享波形；合成/源音频独立预览；5/10/20/40 ms；曲线与预览共用时间窗 | 空结果显示空态；切换不提交任务；旧结果明确标记 |
| F04 操作 | `generate_vowels:913`、`synthesize:1105`、`play_audio:1171`、`export_audio:1181` | 元音只改曲线；合成提交正式持久任务；公共播放器；实际 WAV、CSV、JSON 导出 | 无结果禁用试听/导出；真实取消；失败不清除旧结果；异步结果按 revision/epoch 拒绝 |
| F05 提取/规则 | `_extract_loaded_audio_params:1342`、`_on_load_audio_placeholder:1499`、`_apply_voice_preset:1548` | 只读音频句柄→资源 ID；23 项提取复用 M01 核心；五类原预设、元音表、辅音说明 | 坏 WAV 失败；覆盖前确认；超过 10 s/480000 样本拒绝；无声/非有限轨迹沿用 V2 fallback |
| F06 参数/帮助 | `export_params:1191`、`import_params:1208`、`_on_help_placeholder:1561` | 旧四列 CSV + 完整 `__PTB_CONFIG__` 快照；JSON 导入；公共关闭保护与本机草稿 | 错误表头/超限文件/未知参数拒绝；保存失败不关闭；往返保留 Override 下原曲线与内部单位 |

科学方法、代码来源与许可证分开：tdklatt 实现对应 SRC-TDKLATT，MIT Copyright (c) 2017 Adrian Y. Cho and Daniel R Guest。方法论文为 Dennis H. Klatt (1980), *Software for a cascade/parallel formant synthesizer*, JASA 67, 971–995，DOI 10.1121/1.383940。提取继承已迁移 acoustic 核心的 Praat 及各方法来源标记，不声称本模块原创或感知等价。

## 原样迁移与边界修正

- klatt_config、input_parser、spectral_filter、smoothing_utils 四文件逐字节迁移。tdklatt 只去掉设备播放、文件保存、sounddevice 导入失败 sys.exit 和 main 演示，原数值函数、版权说明保留。widget 的曲线/数组编排抽成 Engine，不依赖 Qt、HTTP、DB、账号或声卡。
- 总时长 2 s、输出 16000 Hz、GUI 淡入 50 ms/淡出 100 ms、平滑 5、F0 50–500 Hz 不变。Klatt 内部 10000 Hz，resample_poly 保留。曲线 linspace 含终点，音频 arange(N)/fs，两种时间轴不混写。
- 合法 IPA 的 +/−/*/÷ 所对应 ASCII `+-*/` 乘数为 1.1/0.9/2/0.5。原解析器静默忽略非法字符，V3 按本轮要求明确拒绝。原预设以实际源码为准：耳语 AV120、气声 AV190。说明书部分乘数示例与 AV 值不同，未照抄。
- 原说明书锁定共振峰说法未对应当前源码控件。原辅音入口只是“已移除辅音合成”的说明，V3 保留这一真实语义，不引入辅音算法。
- 参数提取保留源采样率、float32 通道均值及原时间插值、clipping/fallback。V2 加载时沿用旧 silence，V3 清除旧 silence/boundaries，防止后续错误静音。缩放时长会同步缩放这两个标记，单列为状态一致性修正。
- 标量 Override 的有效值原样保留，越界/NaN 拒绝。分段曲线仍由原插值与裁剪计算，不改默认值。旧 CSV 丢失 Override 下原曲线和部分设置，V3 增加完整快照行；V2 会忽略该行，因此旧 V2 往返仍非无损。
- 原合成 peak .95 不变，WAV 继续采用 V2 soundfile float32→PCM16。手写 floor 量化在近边界样本曾差 1 PCM 单位，现已撤换并用独立 V2 数据验证。
- V2 play_audio 会额外归一到 .99；V3 公共播放器直接播放实际 PCM WAV，音量由公共播放器控制。这是显示/设备适配差异，避免试听与导出音频不一致，未改变合成数组。
- 语谱沿用 scipy Tukey .25、75% overlap、spectrum scaling、5/99 百分位显示，最高 5500 Hz；预览栅格上限 256×1000，仅抽点显示。V2 将图像从 0 铺开，V3 使用真实首尾帧中心时间。
- V2 np.random 序列保留，不在纯核心内改种子。每个正式子进程获得独立 32-bit seed，结果 JSON 记录该 seed；固定 seed 只用于测试。常规用户重复点击仍产生原有随机变化。

## 独立基准

`tests/support/m06_baseline.py` 只加载 V2 klatt 与 V2 acoustic 数值实现，并直接编译原 widget 方法 AST。UI 空实现仅消除控件调用，未调用 V3 生成 expected。公开正弦合成源音频，六个合成配置，167 数组、991816 个值。第二次独立捕获所有数组哈希及来源哈希一致。

| 直接来源文件 | SHA-256 |
| --- | --- |
| `phonetic_toolbox\gui\widgets\speech_synthesis_widget.py` | `989c9e94003af647f3fcea34b0792f5d54936b2303ba757efb7b9c92421b7a14` |
| `Phonetic_Export\index.html` | `a46c984929bcd8073ff1daf4e6b382a6685d4ca19e0fee0e29de3ecf0fb39ad5` |

完整 acoustic 源文件 hash 与数组 shape/hash 在 `tests/fixtures/m06/v2.json`。MIT 许可随核心 wheel 打包为 `synthesis/klatt/LICENSE`，未把上游观测 commit 误作 V2 实际引入版本。

## 2026-10-02 M06-R1 更新

当前 UI 操作以 [M06-R1 报告](../../testing/2026-10-02-m06-r1-report.md) 和 [最新版说明](../../manual/speech-synthesis.md) 为准：辅音规则与右侧重复参数导出入口按用户要求移除，完整结果参数仍随音频目录导出。曲线覆盖中文化、参考共振峰、统一时间域与固定高度均属于交互改动。

[AV 核查](../../references/m06-av-audit-2026-10-02.md) 已证实旧版噪声前级与所核查 tdklatt 上游不同，AV/HNR 映射、0 dB 声源语义和能量提取存在科学解释问题。本轮科研核心不变，原样迁移证据不等于标准 Klatt 校准或感知有效性证明。
