# TextGrid标注 · M12

本目录实现 WAV / TextGrid 的人工标注、词典填充、词表顺序定界及唇形时间对齐。模块 ID `M12` 和名称来自[注册表](../../app/registry.ts)，挂载于[统一外壳](../../app/AppShell.vue)。用户操作正文位于[完整手册章](../../../../manual/chapters/m12.json)，手册工程和图音规范见[编写约定](../../../../manual/AUTHORING.md)与[V2 对照规范](../../../../manual/V2_STYLE_GUIDE.md)。旧[操作记录](../../../../docs/manual/annotation.md)用于历史查验，当前行为以源码与完整手册为准。

## 入口与职责

页面通过 `ResearchContext.files` / `AnnotationPort` 接入文件能力。桌面显示“打开音频目录”，具备文件导入能力的浏览器预览显示“打开语料文件”；服务器项目扫描已归属资源。文件读取、预览、目标准备与版本保存属于平台适配层，前端不直接访问任意本地路径或调用独立服务器。

| 入口 | 职责 |
| --- | --- |
| [AnnotationPage.vue](AnnotationPage.vue) | 文件关联、层角色、顺序模式、搜索、保存与未保存保护，桥接共用波形 / 试听 |
| [AnnotationTracks.vue](AnnotationTracks.vue) | Hann 语谱、相对强度与唇形显示、网格事件及共享选区 |
| [editor.mjs](editor.mjs) | 实例化编辑状态、边界 / 文字 / 参考复用、强度贴合、50 步撤销 |
| [layers.ts](layers.ts)、[format.ts](format.ts) | 可编辑层角色、层创建、TextGrid 长短文本解析与六位秒数序列化 |
| [sequence.ts](sequence.ts) | 空白区间人工定界、词表进度、声母 / 完整韵母首次切分 |
| [movement.mjs](movement.mjs)、[clipboard.mjs](clipboard.mjs) | 音节与对应音素整体平移、完整区间剪贴及越界 / 重叠前置拒绝 |
| [shortcuts.ts](shortcuts.ts) | 图面与输入框快捷键分流，Ctrl / Meta＋S 优先提交字段 |
| [spectrum.ts](spectrum.ts)、[fft.mjs](fft.mjs)、[display.ts](display.ts) | Hann / FFT、30 ms 显示 RMS、唇形可见窗缩放 |
| [default.dict](default.dict) | 内置普通话标签到音素对应，上传词典仅替换当前页资源 |
| [annotation.ts](../../platform/annotation.ts)、[research.ts](../../platform/research.ts) | 类型化文件与标注能力、安全唇形 JSON、浏览器 / 服务器适配 |
| [桌面 annotation.py](../../../../desktop/src/ptb_desktop/annotation.py)、[annotation_audio.py](../../../../desktop/src/ptb_desktop/annotation_audio.py) | 受控递归目录、版本保护与原子写入、长 WAV 轻量预览 |

源码工作台使用既有[统一启动器](../../../../scripts/Start-M16-M17-Workbench.ps1)，实际环境与统一入口见[仓库 README](../../../../README.md)。本页不建立独立 HTTP 服务，也不导入相邻 V2 源码运行。

## 输入、状态与输出

| 类型 | 内容与约束 |
| --- | --- |
| 录音 | WAV；桌面长文件入口最多 2 GB、8–96 kHz、8 声道，必要时第一声道 PCM16 预览。预览预算 64 MB / 3200 万采样值，最低 8 kHz，不升采样，原字节与原时长保留。浏览器入口沿用 64 MB 读取预算 |
| 标注 | Praat 长 / 短文本 TextGrid，文件域必须与原录音相同；最多 64 层、100000 个区间或点、2 MB 文本。词 / 音素角色需不同的全域 IntervalTier，其他层与点层保留 |
| 词典 | `.dict` / `.txt`，每行 `标签 音素1 音素2 …`，标签大小写无关。词典均分是编辑辅助 |
| 词表 | `.lab` / `.txt` 或弹窗粘贴，空白分条目，同名 LAB 自动关联；最多 10000 项、每项 200 字符、2 MB 文本 |
| 唇形 | `ptb.lip/1` 安全 `.lip.json`；桌面可受控解析数值 PKL 及伴随时间戳，拒绝任意 Python 对象。两曲线共用一个独立 `metadata.lip_manual_offset` |
| TextGrid 输出 | 默认 WAV 基本名加 `_自动保存.TextGrid`，支持自定后缀。保存所有保留层与六位小数秒时间；桌面空后缀确认覆盖 WAV 同名原始文件，网页另存不可变版本 |
| 唇形输出 | 保存唇偏独立更新记录元数据，保留其他字段 / 数组；下载安全 JSON 可携带当前偏移，下载不清未保存状态 |

新工作区图窗初值 3.2 s、短音频全段、语谱 Hann 20 ms、共用音量初始 70%、微调步长 1 ms、强度内收 10 ms、参考模式区间之外、顺序生成关闭。已存有效层名、后缀、首次音素切分和右侧开合可恢复。直接搜索词层做包含匹配，替换整段区间标签并重建音素。

原始同名 TextGrid 优先，随后为 `_自动保存`、`_webedit`、`_post`、`_auto`。多个其他候选要求显式选择；不会因恢复版较新就自动抢占原始版。标注剪贴板保留文字、时长、内部音素边界及组间距，不递增声调。

## 保存和恢复

手动保存遵循右侧目标后缀。每 60 秒、切换文件、扫描、离开项目及保存后关闭会调用未保存保护，自动目标固定为 `_自动保存`，唇偏独立保存。待提交字段和输入法组字纳入保护。浏览器预览不触发后台定时下载。

来源 / 目标版本冲突、写入失败或保存期间出现新编辑时保留当前文档并阻止直接切换。下载当前内容可归档，但不会代替原目标保存。关闭标签提供取消、保存修改并关闭与放弃修改并关闭；后者丢弃未保存内存内容。网页额度、启用状态和到期以服务返回为准，参见[现行存储政策](../../../../docs/manual/storage-policy.md)。

## 方法、来源与许可

- 编辑器、Hann / FFT 规则及内置词典来自本项目自有 V2 历史实现 / 整理，来源标识 `ORIGIN-WEBEDITOR`、`PENDING-DICTIONARY` 保留。2026-10-05 已确认自有归属，旧未决字段为历史证据，详见[来源映射](../../../../docs/modules/evidence/M12-source-map.md)、[分类核查](../../../../docs/references/p19-license-classification-audit.md)与[统一登记](../../../../third_party/source-registry.json)。
- `SRC-PRAAT` 在 M12 用于 TextGrid 格式参考，格式说明见 [Praat TextGrid](https://www.fon.hum.uva.nl/praat/manual/TextGrid.html) 与[标注入门](https://www.fon.hum.uva.nl/praat/manual/Intro_7__Annotation.html)。本页客户端语谱 / 强度没有调用 Praat 或 Parselmouth。其他模块的 Parselmouth / Praat 依赖许可另记，不能把共享来源条目当成本页算法。
- 长 WAV 预览复用 SoundFile / SciPy，登记分别为 `PKG-SOUNDFILE` 与 `M01-PY-SCIPY`。两者采用 BSD-3-Clause，libsndfile、wheel 内原生组件与发行材料按单独条款审查。自有编辑器归属不替代整个发行包许可证核验。
- 显示 RMS 为约 30 ms 窗相对强度，强度贴合内部使用 25 ms / 5 ms RMS 与原平滑规则；两者不可混写。唇形曲线各自缩放只供时序比较，原数组不变。轻量预览、词典均分和候选贴合不等于声学测量真值或强制对齐。

## 验证与限度

文档改动使用 `.venv/m09-ui/Scripts/python.exe scripts/manual/validate.py --project manual --strict`。行为修改再按影响选择公共前端检查 `npm --prefix frontend run typecheck`、`npm --prefix frontend test`、`npm --prefix frontend run build` 与定向编辑测试，文档核查不额外声称这些产品测试已重跑。

- [R6 报告](../../../../docs/testing/m12-r6-report.md)记录整段标注剪贴、删除、撤销、保留点层及指定 Chrome / 历史冻结成品保存链。
- [P17 报告](../../../../docs/testing/p17/M12-report.md)记录后续 Windows 源码的指定真实录音编辑、保存下载、图窗与入口，完整复合手势仍 `in_progress`。
- 定向入口包括[剪贴测试](../../../tests/annotation-r6.test.ts)、[顺序测试](../../../tests/annotation-sequence.test.ts)、[Qt 检查](../../../../scripts/verify_m12_qt.py)和[长预览检查](../../../../scripts/verify_m12_long_qt.py)。历史通过范围不能推及当前全部成品、物理 IME / DPI、声卡延迟、自然标注准确率、远程账号链或 Linux / macOS 图形界面。

说明书操作取证见本机 `output/manual-work/chapter-audit-m12.json`。界面截图与 TextGrid 回读不代替真实唇形、实体听辨或科研准确率验证。
