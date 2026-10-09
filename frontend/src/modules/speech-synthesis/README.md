# 声学参数合成 · M06

本目录提供 Klatt 参数曲线编辑、元音规则生成、短 WAV 手动提取和音频合成。详细的逐控件操作在[说明书工程 M06 章节](../../../../manual/chapters/m06.json)，原有[简版说明](../../../../docs/manual/speech-synthesis.md)用于旧文档入口。模块名称与持久 ID 见[模块注册表](../../app/registry.ts)。以下事实按当前源码及 2026-10-04 M06-R5 的范围整理。

## 当前功能与边界

- 编辑 24 条参数曲线，支持 Shift 手绘、Ctrl 局部恢复、观察范围平移/缩放、常量和逗号/分号组合覆盖。单个常量保留底层点，序列覆盖重建点，清除覆盖不能撤销已替换的序列。
- 30 条工具内 IPA 规则建立 F1–F3 起点。生成同时重建 AV、静音与边界，将 F4/F5 恢复默认，保留 F0 及其余声源控制。平滑点数作用于生成的 F1–F3，修改后须重新生成。
- 五类发声预设覆盖 AV、AH、HNR、Slope、H1H2、SHR、Jitter、Shimmer。假声/嘎裂只给基础 F0 整体平移，重复同类预设不叠加偏移，切回其他类别撤去偏移并保留期间编辑。
- 选择 WAV 只加载源波形和试听，提取参数须显式点击。提取使用 Praat CC、Praat AC 或原生 REAPER，固定 10 ms 帧移与已应用 F0 范围。提取没有二次覆盖确认，先保存需要保留的配置。
- 当前仅保留 Klatt。WORLD、PSOLA、Harvest 已移除，当前没有合成后端、锁定共振峰或辅音合成控件。独立的[变速变调模块](../pitch-manipulation/README.md)另有对应原信号处理。
- 预览、控制曲线和测量值语义不同。提取的可编辑 F0 会填补缺值，无声帧 AV=0、AH 初始为 0，不能据图面连续曲线宣称各时刻有效测得 F0，也不能完整重建清音噪声。

## 输入、预算与输出

| 项目 | 实际规则 |
| --- | --- |
| 可选源 WAV | 桌面准入不超过 8000000 字节。正式提取限 0.1–10 秒、480000 采样点、最多 8 声道；所有声道算术平均用于提取，源试听默认第 1 声道 |
| 编辑配置 | 时长 0.1–100 秒，完整文件采样率 8000–192000 Hz；页面没有采样率输入，新配置初始 16000 Hz，提取后使用源采样率 |
| 正式任务 | 当前配置不超过 10 秒且时长×采样率不超过 480000，源自身另检；播放选区不会自动切短提取输入 |
| 参数文件 | 导入 CSV/JSON 不超过 8000000 字节；任务提交完整 CSV 不超过 2000000 字节；每曲线 1–10001 点，IPA 最长 2048 字符 |
| 桌面新合成 | `synthesis.wav`、`m06.ptb.json`、`parameters.csv` 三件套，绑定该次任务快照；WAV 单声道 PCM16 |
| 网页音频导出 | 当前按钮单独下载 WAV，未提供三件套打包入口；下载位置与重名规则由浏览器管理 |
| 顶部导出参数 | `声学参数合成参数.csv`，绑定当前编辑配置，可能与上次合成快照不同 |
| 本机草稿 | 保存完整编辑配置，不恢复源音资源或合成 WAV；独立文件归档须另行导出 |

桌面保存同名且字节/哈希相同的文件会复用，内容不同时加任务标识前缀，仍冲突时再改前缀，已有文件保留。取消保存或失败不清空已生成工件。

## 最短操作

1. 应用总时长，输入 `a` 并点击生成元音。生成只更新配置，尚无新 WAV。
2. 选择参数手绘或应用覆盖，按需要确认应用发声预设。默认选项常态浊声不表示已自动应用该预设。
3. 需要提取时先选择 WAV，等加载完成，应用 F0 范围、选择算法，再点击提取参数。缺失后端报错，不静默回退。
4. 完成时长、范围和覆盖应用，点击合成音频。核对安全幅度衰减提示，在底栏试听源音或结果的全段/选区。
5. 导出实际音频三件套，另导出当前参数或保存草稿。编辑后原音频成为旧结果，重新合成才更新 WAV。

任务固定提交时配置。运行期间改变配置或源关联，迟到结果不自动应用，原结果保留。当前页面只保留最后一次采用的结果，没有多个历史结果切换列表。

## 文件兼容与科学单位

新配置为 `m06/2`。完整 CSV 的 `__PTB_CONFIG__` 是导入权威快照，普通参数行不能覆盖它。旧 `m06/2` 缺 F0 算法时补 CC，R4 Klatt 的 `render` 字段只在内存移除。退役 WORLD/PSOLA/Harvest 配置、旧 `m06/1` AV 标尺或缺完整快照的旧 CSV 明确拒绝，原文件和旧结果不重写。历史四文件清单可读，当前新结果不生成 `analysis.npz`。

AV/AH 范围 0–80 dB，0 关闭相应声源，使用本工具数字参考而非校准 SPL。HNR、H1H2、Slope 是项目频域扩展，不保证输入等于输出重测值。Shimmer 界面为百分数，文件内部为比例，3% 对应 0.03。A1–A5 为并联幅度参数，默认周期源采用串联路由，不能直接承诺独立谱峰幅度控制。五类预设和复制合成效果需试听与独立分析。

## 源码与依据

| 文件 | 职责 |
| --- | --- |
| [SpeechSynthesisPage.vue](SpeechSynthesisPage.vue)、[port.ts](port.ts) | 加载、编辑、提取、任务与旧结果归属 |
| [CurveEditor.vue](CurveEditor.vue)、[SynthesisSpectrum.vue](SynthesisSpectrum.vue) | 曲线手势和语谱显示 |
| [state.ts](state.ts)、[catalog.json](catalog.json) | 24 参数、30 规则、预设、完整配置与兼容校验 |
| [平台适配](../../platform/m06.ts)、[桌面桥接](../../../../desktop/src/ptb_desktop/m06_bridge.py) | 参数准入、任务、下载与原生保存 |
| [Klatt 核心](../../../../packages/phonetic_core/src/phonetic_core/synthesis/klatt/) | 声源、生成、提取、频域扩展与合成 |

来源 ID 为 `REF-KLATT`、`SRC-TDKLATT`、`SRC-PRAAT`、`SRC-REAPER`。论文方法见 [Klatt 1980 DOI](https://doi.org/10.1121/1.383940)，具体实现见 [tdklatt 官方项目](https://github.com/guestdaniel/tdklatt)，提取依赖见 [Praat](https://www.praat.org/)、[Parselmouth](https://parselmouth.readthedocs.io/en/stable/) 与 [REAPER](https://github.com/google/REAPER)。详见[统一登记](../../../../third_party/source-registry.json)、[来源映射](../../../../docs/modules/evidence/M06-source-map.md)、[AV 核查](../../../../docs/references/m06-av-audit-2026-10-02.md)、[提取核查](../../../../docs/references/m06-r3-resynthesis-audit.md)及[R3 ADR](../../../../docs/decisions/ADR-M06-R3.md)。旧 R4 ADR 已被 R5 替代，不能当作当前功能清单。

## 验证入口与范围

从仓库根目录使用[源码启动器](../../../../scripts/Start-M06-Workbench.ps1)。产品代码检查使用 `npm --prefix frontend run typecheck`、`npm --prefix frontend test`、`npm --prefix frontend run build`；文档结构检查使用 `python scripts/manual/validate.py --project manual --strict`。

定向检查入口为[加载交互](../../../../tests/e2e/m06-r5-loading.cjs)及[Qt 检查](../../../../scripts/verify_m06_r5_qt.py)。[R5 报告](../../../../docs/testing/2026-10-04-m06-r5-report.md)记录 Windows 源码、Chrome/实际 Qt 及 WSL 纯核心范围，[统一名称和入口报告](../../../../docs/testing/2026-10-05-p19-r14-r16-report.md)记录后续界面整合。科学任务、自然度听辨、实体声卡及跨平台 GUI 分别验收，文档结构和链接检查不能替代。说明书图音应来自当前实现并注明生成条件。
