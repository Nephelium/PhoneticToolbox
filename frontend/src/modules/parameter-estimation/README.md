# 参数估计 · M01

本目录实现共享参数估计工作台。模块名称与稳定 ID 见[注册表](../../app/registry.ts)。完整用户说明的可编辑正文为[说明书 M01 章](../../../../manual/chapters/m01.json)，历史 Markdown 入口为[参数估计操作说明](../../../../docs/manual/parameter-estimation.md)。以下按当前源码整理接口、科学解释和开发边界。

## 工作台与输入

顶部选择 WAV、TextGrid、唇形及结果目录。左栏选择文件、80 项输出参数、14 项设置及可选 EGG 角色，中央预览波形、Praat 语谱图和标签，右栏提交批次并显示单文件进度、处理记录和逐文件状态。底部公共播放条只控制试听范围。

顶部右侧“帮助”打开应用内 `m01` 章，沿用已登记的 37 个小节锚点；“方法与引用”打开独立来源窗口。说明书的 16 张四列表均有独立表题，23 张最大化 V3 图片按参数选择、设置校验、分析范围、关联、切分、EGG 与保存恢复步骤放置，章节和图表编号由阅读器统一生成。

点击文件名加载预览；复选框决定“分析选中文件”及切分范围。“开始全列表分析”始终处理全部 WAV。“分析选中文件”只提交勾选项，按列表顺序并带入各项关联及声道覆盖。当前试听选区不会缩短完整音频的计算范围。

| 输入 | 当前行为 |
| --- | --- |
| WAV | Windows 本机支持顶层／递归目录；新有界分析最长 1800 秒、2 GB、8–192 kHz、1–8 声道 |
| TextGrid | 默认同名关联，可独立目录或逐文件改选；批量切分按层名匹配 |
| 唇形 | `.lip.json` 优先；本机受限读取 V2 数值 `.pkl` 及时间戳伴随文件，网页使用安全 JSON |
| 历史参数 | 指定旧单表 XLSX／`.ptb.sqlite`／`.ptb.sqlite3` 仅用于明确关联的参数切分 |

TextGrid／唇形目录默认跟随音频，显式选择后保留，重置恢复跟随。递归按相对路径匹配。批量唇形关联可替换已找到同名项的手动选择，未找到时保留手动项。草稿保存已应用参数、14 项设置及默认 extended 设置，不保存文件授权、目录、音频或逐文件声道覆盖。

## 设置与科学解释

[契约设置](../../../../contracts/schemas/acousticrequest.json)给出新草稿默认值与范围。默认帧移 5 ms、能量窗 40 ms、分析窗 40 ms、参数平滑 10 帧、唇形平滑 0、F0 范围 30–800 Hz。旧草稿恢复有效的已保存值。

- 最小／最大 F0 位于 REAPER 页签，但同时用于 Praat、REAPER、WM 链及部分派生指标。
- 仅有声帧最终由 Praat／REAPER 有效正基频并集判定。新 `acoustic-bounded/1` 同时使用全段强度静音参考，避免低振幅块自行改变阈值。
- 分析窗不统一控制全部算法。WM Jitter／Shimmer 采用 `max(160 ms, windowsize_ms)`；Burg 共振峰固定 25 ms 窗；谐波、CPP 与 HNR 使用 `n_periods` 周期窗；SHR 固定 40 ms；Slope 固定 5 周期。
- 共振峰数量传入 Burg 候选数，但目录仅输出 F1–F4／B1–B4，沿当前分槽阈值；不能推断为无筛选 Praat 同序候选。
- 数字谱幅为 dB 标度，Intensity 为本实现均方强度标度，均不直接当作校准声压级。Jitter／Shimmer 为百分数；SHR 当前为截断到 1 的幅值比；Slope 为 dB 对 log10(Hz) 的斜率。80 项完整单位和实现解释见说明书。
- 唇形四项读取已关联指标并插值。本应用记录使用归一化比例；M01 不重新计算图像关键点、不执行新的同步校准，也不能假定外部历史记录沿同一尺度。

## 联合 EGG 与长文件

默认 EGG 关闭，音频全部声道混合。开启后默认音频2／EGG1，可逐文件交换。试听声道独立；单声道跳过 EGG 并继续声学分析。当前界面角色选择提供双声道1／2。

两种形式均保留原周期：插值到主表并保留原始周期表，或原始周期独立表。默认插值后平滑 20 ms、最大插值间隔 50 ms；无效周期、长缺口和范围外不插值。gF0 派生默认关闭，开启后在音频声道上添加 `(gF0)` 谐波、幅值差、CPP、HNR、SHR 和 Slope，pF0／rF0 保留，Jitter／Shimmer 不改变为 GCI 扰动。

CQ 为接触时长／周期时长；SQ 沿项目定义 `(去接触时长−接触建立时长)/接触时长`，不解释为普通两时长之比。CQ／SQ 位于周期起点，gF0 位于相邻 GCI 中点。

有界分析采用 20 秒输出块加上下文，全局帧编号与全段静音参考明确记录。基频路径和滤波端点依赖上下文，不宣称与 V2 整段调用逐位相同。旧请求／网页仍限 200 万采样值、240 秒、64 MB 结果和 20 万单元格，新长文件／EGG 尚未开放远程。波形预览另限原源 8–96 kHz，轻量预览不替换科研输入。

## 保存与切分

桌面提交时固定自动保存目标，默认回到各 WAV 所在目录，也可独立目录。用户产物为 XLSX 和 `.ptb.sqlite`，新 `m01-bundle/2` 含主表 `params`、可选 `egg_cycles` 和 `ptb_metadata`。超过 100 万数据行时 XLSX 分续表；NaN 为空／NULL，无穷值为显式文本。内部 `.ptb.json` 保存来源与产物摘要，长表由 SQLite 承载。

同名同内容复用，不同内容整组以相同末尾数字保存，例如 `声音 (2).xlsx` 和 `声音 (2).ptb.sqlite`，不覆盖已有文件。重开后须重新授权目录，历史递归批次选独立结果目录。“保存已完成结果”可再次导出成功缓存。

TextGrid 切分有勾选时处理勾选项，无勾选时只处理当前文件。参数切分默认找同一任务存储中原 WAV 哈希一致的最近完整结果，只截原帧，`Time_s` 减去实际首样本时间，并保留 `Source_Time_s`。新多表同时切主表和周期表，周期按 GCI 起点归属，跨结尾周期保留原终点。指定历史表仅支持旧单表，明确记录 `user_associated_unverified`；缺关联阻止提交，缺同源父结果提示并仅切 WAV。

## 源码结构

| 文件 | 职责 |
| --- | --- |
| [ParameterEstimationPage.vue](ParameterEstimationPage.vue) | 目录、关联、预览、列表顺序提交与切分 |
| [state.ts](state.ts)、[store.ts](store.ts) | 契约默认值、校验、关联和草稿 |
| [JointEggSettings.vue](JointEggSettings.vue) | 本机联合 EGG 控件与角色 |
| [BatchResults.vue](BatchResults.vue) | 常驻单文件进度、批次、取消、重试和保存 |
| [Acoustic 服务](../../../../packages/phonetic_core/src/phonetic_core/services/acoustic.py) | 共享纯科学编排、掩码、平滑及后端记录 |
| [流式服务](../../../../packages/phonetic_core/src/phonetic_core/services/acoustic_stream.py) | 有界块计算、全局参考、周期与时间归属 |
| [参数目录](../../../../packages/phonetic_core/src/phonetic_core/catalog.py) | 80 项稳定键和显示名 |

## 开发与验证证据

从根目录使用[源码启动器](../../../../scripts/Start-M01-M02-Workbench.ps1)，复用既有项目环境。前端必需检查为 `npm --prefix frontend run typecheck`、`npm --prefix frontend test` 和 `npm --prefix frontend run build`。文档工作不据此重新声称产品任务验证。

[前端定向测试](../../../tests/m01-r4.test.ts)、[联合 Qt 检查](../../../../scripts/verify_m01_r4_qt.py)、[R4 报告](../../../../docs/testing/2026-10-04-m01-r4-report.md)和[选中文件 R5 报告](../../../../docs/testing/2026-10-04-m01-r5-selected-report.md)记录限定 Windows 源码证据。30 分钟选定参数实测不代表 30 分钟全部参数、自然语料准确率、实体音频、DPI 或 Linux GUI 全部通过；对应历史报告的旧 EXE 未更新状态按具体包范围解释。

说明书操作取证保留在本机 `output/manual-work/chapter-audit-m01.json` 及对应运行报告，README 不复述截图轮次和测试数量。正式研究数据与用于取证的临时副本分开管理。

方法涉及 Praat／Parselmouth、REAPER、IRAPT、WMPC、VoiceSauce 及修正方法。参见[来源映射](../../../../docs/modules/evidence/M01-source-map.md)、[联合分析 ADR](../../../../docs/decisions/ADR-M01-R4.md)与[统一来源登记](../../../../third_party/source-registry.json)。论文引用、代码来源和分发许可分别记录，研究复现读取实际任务元数据。
