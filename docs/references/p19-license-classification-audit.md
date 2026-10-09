# PhoneticToolbox v3 来源与许可分类核查

核查日期：2026-10-05。当前工程来源登记 391 条，逐项结果与本机许可文件摘要见 [audit.json](../../third_party/evidence/p19-license-classification/audit.json) 和 [依赖许可清单](../../third_party/evidence/p19-license-classification/local-dependency-licenses.json)。这些条目包含历史环境、开发工具、可选依赖和重复版本登记，不能当作 391 个实际随 EXE 分发的独立组件。

## 结论与本轮处理

| 处理 | 登记条数 | 结论 |
| --- | ---: | --- |
| 纯论文、方法、格式或文档参考 | 43 | 已隐藏界面许可说明，保留引用、作者和来源链接。无需为独立实现的方法本身申请著作，如图2权许可。 |
| 有现有开放许可的库、代码或资源 | 292 | 按许可原文执行，通常无需额外邮件。含本轮补齐 CC BY 4.0 的 VoQS 中文译表。仍须核对最终实际发行文件及其原生组件，不能把本表当作全包发行验收。 |
| 载瓦语发声类型合成 | 2 | 1 条代码来源保留 2026-09-10 邮件授权，1 条论文保留关联许可上下文，无新增邮件需求。 |
| VoiceSauce / SoE 代码关系 | 2 | 两条登记涉及同一个主要缺口。已找到 OpenSauce Octave 的 BSD 两条款许可，未找到覆盖所述 SoE MATLAB 函数的具体许可。 |
| MFA / FFmpeg 聚合记录 | 2 | 先核对实际构建、模型、词典、许可原文与源码义务，当前不等于需要另求作者许可。 |
| 配色适配 | 29 | 少量基本颜色与完整主题实现分别判断。公开许可原文继续保留，缺少同名主题元数据不自动成为群发邮件理由。 |
| 项目自有或生成材料 | 11 | 含本轮用户确认的 EGG、TextGrid、词典、11 套 IPA 映射及音系归纳 5 项，已移出外部致谢。另有两个当前可见的自有或生成记录隐藏第三方许可提示，第三方依赖另列。 |
| 当前未使用的历史依赖或退役组件 | 10 | 4 条原已退役。新增隐藏 html2canvas、React、Babel、Tailwind、Lucide 和 PySide6 候选宿主 6 条当前致谢记录，工程历史保留。 |

合计 391 条。当前界面 372 条来源记录；45 条当前可见记录的许可说明为空，其中纯引用 43 条、自有或生成来源 2 条。本轮保留原来的实际版本及代码来源证据，未删除工程记录或论文引用。五项自有来源依用户确认更新作者与当前状态，其此前未决字段另存历史，不把用户确认扩展为全部第三方材料均无权利限制。

## 独立实现与调用库的区别

按论文中的思想、数学定义、方法或流程自行编写代码，通常无须取得这些方法的著作权许可。中国《计算机软件保护条例》第六条明确区分软件表达与思想、处理过程、操作方法、数学概念；纯书目引用和链接也不能视为复制论文全文的许可需求。[最高人民法院知识产权法庭公布的条例](https://ipc.court.gov.cn/zh-cn/news/view-407.html)、[WIPO 著作权说明](https://www.wipo.int/en/web/copyright/protection)支持这个区分。

直接复制或改写别人代码、导入 Python 包、随产品带入二进制库、字体、模型、词表或图像，应按相应材料的现有许可执行。有明确开放许可且使用方式符合条件时，通常不需要额外邮件。免费发布、学术用途或加了致谢，都不能替代已经适用的许可条件。

本轮隐藏许可说明只影响界面展示。代码改写记录、许可原文和最早来源未决项继续留在工程资料中，便于之后按实际发行物执行许可。

## 按现有许可执行的重点

### GPL 与 LGPL 组件

- **Parselmouth 0.4.7**：官方版本 README 与本机 `praat_parselmouth-0.4.7.dist-info` 许可明确为 GPLv3 或更高版本。该版本内嵌 Praat 6.1.38 的 GPLv2 或更高版本声明另列。旧的 license pending 标签已更正。[对应版本官方说明](https://github.com/YannickJadoul/Parselmouth/blob/v0.4.7/README.md)。
- **PyQt6 / PyQt6-WebEngine**：本机开源包使用 GPLv3；PyQt 提供 GPLv3 与商业许可两种渠道，不能把 Qt 的 LGPL 当成 PyQt 的许可。按 GPL 组合和分发应用时，需要项目许可证兼容，并落实对应源码、许可和通知。若选择不兼容 GPL 的发行方式，应使用适用商业许可。[Riverbank 产品许可](https://en.riverbankcomputing.com/software/pyqt)、[商业许可 FAQ](https://en.riverbankcomputing.com/commercial/license-faq)。
- **VocalTractLab / IRAPT**：使用了原生实现、适配代码或实际移植的代码/数据，继续按其 GPL 许可履行相应源码和通知义务。不能因为科学方法来自论文而把这些具体代码许可也隐藏。
- **Qt、libsndfile、部分 FFmpeg 及原生依赖**：按实际文件适用的 LGPL/GPL、例外或其他许可分别处理。LGPL 是否涉及源码、修改披露及用户替换/重新链接能力，取决于实际组合方式。[Qt 的 LGPL 义务](https://www.qt.io/development/open-source-lgpl-obligations)、[FFmpeg 官方法律说明](https://ffmpeg.org/legal.html)。

**当前项目根目录尚未找到项目总 LICENSE 文件。** 免费、开源的发行意图已明确，但应在公开发行前由项目作者选择兼容的总许可证，并准备应提供的对应源码及组件通知。本轮没有替井井给整个项目更换许可证，也没有核验新 EXE 的完整对应源码交付。

### MIT、BSD、Apache、MPL 与构建例外

| 实际使用 | 现有许可与处理 |
| --- | --- |
| tdklatt / TrackDraw、Three.js、Vue、许多工具库 | MIT。保留版权与许可原文。tdklatt 是实际改写来源，科研公式后续修改不消除原代码通知义务。 |
| OpenSauce Octave 对应文件、NumPy、SciPy、Pandas 等 | BSD。保留各自版权与许可证，wheel 内 OpenBLAS/LAPACK 等组件不能被顶层 BSD 标签覆盖。 |
| REAPER、OpenSauce Python、MediaPipe 代码、SheetJS CE 0.20.3、MFA 程序、Kaldi | Apache-2.0 或各自登记的 MIT。Apache 组件保留许可、适用 NOTICE 和修改说明。MFA/Kaldi/模型/词典分开处理。 |
| Troparion / WM-PC jitter 与 shimmer | MPL-2.0，保留适用文件及改写来源的版权与源码义务，按文件范围判断。 |
| PyInstaller 与 hooks | 构建工具许可及 bootloader 分发例外另列；随包 runtime hooks 使用 Apache-2.0。不能仅因构建工具有 GPL 字样就推导整个应用必须采用相同许可。 |
| Matplotlib、Pillow、dateutil、PyWavelets、OpenCV 等 | 保留实际发行文件中的完整许可。dateutil 有按贡献时间区分的 Apache/BSD 范围；OpenCV 的第三方文件包含独立原生组件许可。 |

本轮读取既有环境中的 491 条 Python 安装元数据，得到 99 个不同的包名/版本组合和 522 个许可文件记录；另核对 170 条 npm 锁定项。可选平台条目、开发依赖及历史探针环境并不因此进入当前 EXE。旧登记版本与现有环境不同的条目，只更正许可名称，不擅自改写为当前打包版本。

### 字体、模型与研究素材

| 材料 | 结论与必要保留内容 |
| --- | --- |
| Doulos SIL 7.000、JetBrains Mono 2.304、PTB IPA Plus 1.000 及派生字形的 Noto 来源 | OFL-1.1，有明确开放许可，无需额外邮件。保留版权与完整原文；派生字体遵守保留名称条件。系统宋体/Times New Roman 使用已安装字体，不随本项目新增分发这些商业字体文件。[OFL 官方 FAQ](https://openfontlicense.org/ofl-faq/)、[JetBrains 对应版本 OFL](https://github.com/JetBrains/JetBrainsMono/blob/v2.304/OFL.txt)。 |
| 平均鼻腔网格 | 本轮补齐官方 Figshare v4 的 CC BY 4.0 证据，原始 file ID、MD5 和 SHA-256 与现有 `sources.lock.json` 一致。保留作者、v4 DOI、许可链接及坐标/几何修改说明，无需邮件。[v4 官方元数据](https://api.figshare.com/v2/articles/9585410/versions/4)、[数据 DOI](https://doi.org/10.6084/m9.figshare.9585410.v4)。 |
| FACE2 头部资产 | 原创建者 byzmod3d 在 OpenGameArt 以 CC0 发布，无需另求许可。保持素材来源及非 MRI 个体模型的说明。[原创建者资源页](https://opengameart.org/content/3d-human-parts-pack)。 |
| Web MediaPipe face_landmarker float16/1 | 原始下载 SHA-256 与现有 `resources/m05/resources.json` 一致。官方 FaceMesh V2、BlazeFace 短距和 Blendshape V2 模型卡明确 Apache-2.0，当前 Web 模型按现有许可处理。legacy wheel 模型与原生组件仍须按各自实际文件核对，不能拿网页页脚的内容许可替代模型许可。 |
| 官方 IPA 图表、自绘交互表 | 官方图表为 CC BY-SA 4.0，保留作者、版本和许可链接，适用的衍生图表按相同许可；通用符号本身与自绘图表的表达范围分开判断。[官方许可页](https://www.internationalphoneticassociation.org/content/ipa-chart)。 |
| extIPA 2025 | ICPLA 官方页明确 CC BY-SA 3.0，另说明原表复制不作变更。当前没有分发或修改原 PDF，符号与自主说明不自动需要邮件。未来若直接改原图，应先澄清这段复制说明与 CC 条款的范围。[ICPLA 官方资源](https://www.icpla.org.uk/resources)。 |

模型卡原始链接：[FaceMesh V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)、[Blendshape V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf)、[BlazeFace Short Range](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20%28Short%20Range%29.pdf)。资源摘要记录见 [鼻腔原始网格身份](../../third_party/evidence/p19-license-classification/nasal_original-identity.json) 和 [Face Landmarker 身份](../../third_party/evidence/p19-license-classification/face_landmarker_original-identity.json)。

## 需要确认的具体对象

### 优先确认 SoE 对应代码的许可范围

`packages/phonetic_core/src/phonetic_core/acoustic/soe.py` 明确写有依据 Soo Jin Park 的 `func_getSoE.m`，保留了 MATLAB 来源线索。登记也将它列为实现链。仅凭当前材料，无法将此项判定为完全从论文独立编写。

本轮取得 [OpenSauce Octave 官方 BSD 两条款原文](https://github.com/voicesauce/opensauce/blob/master/LICENSE)，保存到 [许可证据](../../third_party/licenses/SRC-VOICESAUCE/OPENSAUCE-OCTAVE-BSD.txt)。观测提交为 `c81b15d727e990236197291923fb05de7f6d2a81`，它证明本轮看到的仓库状态，不能倒推为项目最初改写版本。[OpenSauce Python](https://github.com/voicesauce/opensauce-python) 已有 Apache-2.0。

两仓库当前文件树未找到 `func_getSoE.m`。Octave 仓库 CPP/HNR 等文件的 BSD 许可不自动覆盖仓库外的 SoE 或其他 VoiceSauce 发行版。原 VoiceSauce 官方站点访问受到证书/连接故障影响，未绕过证书校验，也未把站点论文或数据的许可扩大为全部代码的许可。

因此，如果实际保留了该 MATLAB 函数受保护的代码表达或改写，应该先向 VoiceSauce 维护者或对应代码权利人确认免费开源改写与分发条件。若井井能确认本地实现仅依据数学方法独立编写，并补足这段来源说明，则无需为方法本身申请许可。当前列为需要解决的具体代码授权线索，不是要求给 SoE 论文作者申请方法版权。

CPP/HNR 等明确参考代码的部分，先核对实际改写来源是否对应已获 BSD/Apache 许可的实现。能绑定对应文件即可按现有许可执行；仍找不到覆盖许可的具体改写文件，才针对该文件求确认。

### VoQS 中文译表已补齐明确许可

按井井指定，将引用更新为：王天恒. VoQS：音质符号（2016 中英双语版）[EB/OL]. Zenodo(2020-08-30)[2023-12-26]. [https://doi.org/10.5281/zenodo.10206204](https://doi.org/10.5281/zenodo.10206204)。

[Zenodo 官方记录](https://zenodo.org/records/10206204) 与 [API 元数据](https://zenodo.org/api/records/10206204) 明确作者 Wang, Tianheng、发表日期 2020-08-30、CC BY 4.0，并把旧知乎文章列为 identical-to。已保存 [记录证据](../../third_party/evidence/p19-license-classification/voqs-zenodo-record.json)。引用中的 2023-12-26 访问日期按井井提供的书目保留，本轮查验日期另记为 2026-10-05。

56 个既有译名保持。按 CC BY 4.0 保留译者、DOI、许可链接以及自行绘制交互界面的修改说明，**无须额外邮件**。原表作者 Ball、Esling、Dickson 的归属与原论文参考继续保留，原 PDF 未随产品新增分发。稳定 ID `M17-VOQS-ZH-UNTPHESOCA`、`voqs-zhihu-203037479` 保留，防止破坏既有条目引用；界面和说明书改用新的正式作者与 DOI。

### 五项已由项目作者确认

| ID | 项目 | 当前结论 |
| --- | --- | --- |
| PENDING-EGG | EGG 事件分析与简化逆滤波 | 用户确认的自有历史实现。聚焦检查未发现另有明确外部代码改写声明，Parselmouth 等依赖保留各自许可。方法文献与科学有效性仍分别记录。 |
| ORIGIN-WEBEDITOR | TextGrid / web_praat_editor 迁移链 | 用户确认的自有历史编辑器。格式、Hann/FFT 和通用方法参考保留。 |
| PENDING-DICTIONARY | 内置普通话发音词典 | 用户确认的自有整理成果，不再向外部词典作者求许可。当前简短拼音与音素映射不据此声明为官方权威词典。 |
| PENDING-IPA | 11 套汉字转 IPA 规则与映射 | 用户确认的自有代码与整理成果，补入其另一账号的 IPA-lab 仓库证据。各标准的完整书目信息仍需作为引用质量工作核实。 |
| PENDING-PHONOLOGY | 音系归纳规则与帮助材料 | 用户确认的自有历史实现。独立方法和通用语言事实无需为方法本身额外求许可。 |

井井在本轮逐项确认这五项是自己的历史工作，其中部分曾有 AI 协助。已更新来源登记，保存之前的未决字段，并从外部致谢中隐藏。AI 协助编写本身不产生向所有参考论文作者求许可的需求，具体复用的第三方代码与材料仍按各自许可执行。

[IPA-lab](https://github.com/Nephelium-chryseum/IPA-lab) 的官方 API 显示仓库创建于 2020-03-14，六次公开提交在 2020-03-14 至 15 日。观测 HEAD 为 `6c25e53b4954e6fbabef1e26fc483a48bed348b2`，包含 MATLAB 程序、CSV 字表和 Excel 文件。账号归属来自用户直接确认，仓库日期及文件支持历史来源。该证据不宣称 2020 文件与现有 V3 数据逐字一致，也不把没有独立 LICENSE 的自有仓库当成需要作者向自己求许可的缺口。见 [仓库身份摘要](../../third_party/evidence/p19-license-classification/ipa-lab-identity.json)。

### 聚合发行物和配色还需完成的工作

- MFA 本体、Kaldi 的现有许可明确。当前可选工具环境还包括模型、词典及约 193 个 Conda 包，必须绑定实际材料与对应许可。缺少模型下载身份或完整原生通知是发行资料缺口，不能自动改写为论文作者未授权。
- FFmpeg 依据实际 GPL/LGPL 构建判断。现有 MFA 环境的 FFmpeg 是 GPL 构建，不能用另一套 LGPL 构建的说明代替。程序内 OpenCV 自带视频组件也应保留其实际第三方通知。
- 29 套配色使用项目独立界面与少量基本颜色值，未复制 Codex 的 JS/CSS、图标或字体。基本配色和具备独创表达的完整主题文件、品牌标识须分开判断。11 套完整 MIT 原文、Gruvbox 声明及 Monokai 公开实现的证据继续保留；16 套同名来源/许可资料缺口保留在 [此前逐主题核查](p19-theme-license-audit.md)，不将这些缺口一概变成必须给所有主题作者发邮件的结论。未来引入完整主题文件或素材时，再按确切来源核对。

## 已有邮件授权

`SRC-ZAIWA` 保留 2026-09-10 作者团队邮件，范围包括相关 MATLAB 代码改写、集成以及免费开源发布。`REF-ZAIWA` 保留论文引用与这一代码授权的上下文。原论文全文、录音和统计数据不会因为这封代码许可自动获得再分发许可，本轮也未新增分发这些材料。

## 验证与边界

本轮只调整来源登记、许可原文证据与公共致谢显示，科研计算代码未因本次分类变更。验证结果见 [本轮报告](../testing/2026-10-05-p19-r13-license-classification-report.md)。未对外发送邮件、替整个项目选择许可证、公开发布或重新打包 EXE。

这份核查完成了 391 条现有登记的分类、现有本机许可元数据读取以及重点未决来源的官方资料核实。它不代表已证明全部历史代码均为原创，也不代表完整 Qt/Chromium、所有 wheel/Conda 原生组件及最终发行包的源码义务已经逐文件验收。具体未决对象已逐项保留，避免把所有论文统一标成待授权。

## 全部登记逐项索引

下表使用当前登记版本，不以本轮上游观测替换历史版本。每项使用方式、判断依据、后续动作、许可文件与摘要在机器可读清单中完整保留。

### 纯引用，隐藏许可说明（43 条）

| ID | 名称 | 当前登记许可或范围 | 界面处理 |
| --- | --- | --- | --- |
| REF-CPP | CPP: Hillenbrand, Cleveland & Erickson | not-established | 许可说明隐藏 |
| REF-HNR | HNR: de Krom | not-established | 许可说明隐藏 |
| REF-SHR | SHR: Xuejing Sun | not-established | 许可说明隐藏 |
| REF-ISELI | 谐波幅度校正: Iseli & Alwan | not-established | 许可说明隐藏 |
| REF-HAWKS | 共振峰带宽估计: Hawks & Miller | not-established | 许可说明隐藏 |
| REF-KLATT | Klatt formant synthesizer | not-established | 许可说明隐藏 |
| REF-GRIFFINLIM | Griffin–Lim 重建 | not-established | 许可说明隐藏 |
| DOC-VTL24 | VocalTractLab 2.4 官方手册 | not-established | 许可说明隐藏 |
| DOC-VTL23 | VocalTractLab 2.3 官方手册（补充） | not-established | 许可说明隐藏 |
| REF-VTL2006 | 三维声道模型 | not-established | 许可说明隐藏 |
| SRC-VTLWRAPPER | VocalTractLab-Python | not assessed for redistribution; no bundled code claimed | 许可说明隐藏 |
| REF-MFA | Montreal Forced Aligner 论文 | not-established | 许可说明隐藏 |
| P05-REF-OWASP-SESSION | OWASP Session Management Cheat Sheet | No text/code redistributed; document license not independently audited | 许可说明隐藏 |
| P05-REF-OWASP-CSRF | OWASP CSRF Prevention Cheat Sheet | No text/code redistributed; document license not independently audited | 许可说明隐藏 |
| P06-PG-LOCKS | PostgreSQL 17 SELECT and advisory locking documentation | Reference only; no documentation redistributed | 许可说明隐藏 |
| P07-PYTHON-FILES | Python 3.11 file durability and process file locking documentation | Reference only; no documentation redistributed | 许可说明隐藏 |
| P07-STARLETTE-RESPONSES | Starlette response ASGI interface documentation | Reference only; no documentation redistributed | 许可说明隐藏 |
| M01-WIN32 | Windows Job Objects and named pipes | Documentation referenced, no third-party document/code redistributed | 许可说明隐藏 |
| M01-WAVE | RIFF WAVE / WAVEFORMATEXTENSIBLE | Documentation referenced, no third-party document/code redistributed | 许可说明隐藏 |
| M01-PICKLE | Python 3.11 pickle format and restricted numeric conversion | Documentation referenced, no third-party document/code redistributed | 许可说明隐藏 |
| REF-WEBCODECS | WebCodecs specification | Specification reference only; no specification text or external muxer source code redistributed | 许可说明隐藏 |
| REF-WEBM | WebM Container Guidelines | Specification reference only; no specification text or external muxer source code redistributed | 许可说明隐藏 |
| REF-PNG | Portable Network Graphics (PNG) Specification (Third Edition) | Specification reference only; no specification text or sample implementation redistributed | 许可说明隐藏 |
| REF-FONT-RENDERING | CSS font loading, Qt font catalogue and Matplotlib font resolution | API and specification reference only; no upstream example code copied | 许可说明隐藏 |
| REF-YIN-EGG-THESIS | 汉语韵律的嗓音发声研究（旧版手册引用） | 引用与原站链接；论文全文未收录 | 许可说明隐藏 |
| REF-HENRICH-2004-DEGG | On the use of the derivative of electroglottographic signals for characterization of nonpathological phonation | © 2004 Acoustical Society of America；引用与原站链接，全文未收录 | 许可说明隐藏 |
| REF-MAKHOUL-1975-LPC | Linear Prediction: A Tutorial Review | Citation and original link only; paper not bundled | 许可说明隐藏 |
| M17-VOQS-BALL-2018 | Revisions to the VoQS system for the transcription of voice quality | Article/chart copyright retained by rights holders; no redistribution grant inferred | 许可说明隐藏 |
| M17-UNICODE-PHONETIC-ENCODING | Unicode phonetic marks, VoQS glyph correction and cartouche encoding discussion | References only; no Unicode documents bundled | 许可说明隐藏 |
| M17-CIN-REFERENCE | User-supplied ipa.cin lookup aliases | Original file provenance/license not established; original not bundled | 许可说明隐藏 |
| M17-INTERACTIVE-LAYOUT-REFERENCE | Interactive IPA chart — layout reference | No third-party code or assets copied | 许可说明隐藏 |
| M17-EXTIPA-LV-JIANG-2013 | 国际音标扩展表的分类、命名与功能 | Copyright retained; no redistribution permission inferred | 许可说明隐藏 |
| M17-IOS-IPA-PHONETICS-REFERENCE | User-supplied iPA Phonetics Pro screenshot — visible symbol inventory reference | Reference only; third-party screenshot, artwork, code and assets not copied | 许可说明隐藏 |
| M17-IPA-CHART-ZH-2007 | 中国语言学会语音学分会 · 国际音标中文版（2007；修订至2005年） | 中文名称据用户指定图核对；原图与文章不随包分发 | 许可说明隐藏 |
| M17-IPA-HANDBOOK-JIANG-2008 | 国际语音学会编，江荻译（2008）· 国际语音学会手册：国际音标使用指南 | 书籍版权保留；只使用自主短解释、书目信息和页码，原书与扫描图不随包 | 许可说明隐藏 |
| M17-VOICE-QUALITY-ESLING-2019 | Esling, Moisik, Benner & Crevier-Buchman (2019) · Voice Quality: The Laryngeal Articulator Model | 书籍版权保留；自主概述并列准确章节页码，原PDF及图像不随包 | 许可说明隐藏 |
| M17-EXTIPA-BALL-2018 | Ball, Howard & Miller (2018) · Revisions to the extIPA chart, pp.155–164 | 原论文与图表不随包，条目含义自主概述 | 许可说明隐藏 |
| M17-EXTIPA-BALL-2024 | Ball (2024) · Changes to certain extIPA diacritics, pp.692–695 | 原文CC BY-NC-ND 4.0；仅引用与自主解释，未复制或修改原图 | 许可说明隐藏 |
| M17-EXTIPA-BALL-2025-REPLY | Ball (2025) · Reply to letter concerning the revisions to some diacritics of the extensions to the IPA (extIPA), Ball (2024), pp.911–912 | 原文版权与许可保留；仅书目和版本差异说明，原PDF不随包 | 许可说明隐藏 |
| SRC-CODEX-PALETTE-REFERENCE | Codex themes: visual reference | 历史视觉参考记录；逐主题公开上游及许可另见 SRC-PALETTE-*。不声明 Codex 桌面应用或所有方案统一开源。 | 许可说明隐藏 |
| REF-DRUGMAN-GLOTTAL-COMPARISON | A Comparative Study of Glottal Source Estimation Techniques | Reference and original-site link only; no paper or code redistribution | 许可说明隐藏 |
| REF-VOCODER-COMPARISON-2018 | Sound quality comparison among high-quality vocoders by using re-synthesized speech | paper citation only; no redistribution licensed here | 许可说明隐藏 |
| REF-GLOTTDNN-2016 | GlottDNN — A Full-Band Glottal Vocoder for Statistical Parametric Speech Synthesis | paper citation only; no redistribution licensed here | 许可说明隐藏 |

### 执行现有许可（292 条）

| ID | 名称 | 当前登记许可或范围 | 界面处理 |
| --- | --- | --- | --- |
| SRC-PRAAT | Praat / Parselmouth | Parselmouth 0.4.7：GPL-3.0-or-later；内嵌 Praat 6.1.38：GPL-2.0-or-later。按组合发行适用条款提供许可与对应源码；依赖另列。 | 保留许可/范围说明 |
| SRC-REAPER | REAPER (Robust Epoch And Pitch EstimatoR) | Apache-2.0 | 保留许可/范围说明 |
| SRC-IRAPT | IRAPT | GPL-3.0 | 保留许可/范围说明 |
| SRC-WMPC | Troparion / WM-PC jitter and shimmer | MPL-2.0 (Troparion repository) | 保留许可/范围说明 |
| SRC-OPENSAUCE | OpenSauce Python | Apache-2.0 | 保留许可/范围说明 |
| SRC-TDKLATT | tdklatt / TrackDraw | MIT | 保留许可/范围说明 |
| SRC-VTL | VocalTractLab API 2.4 | GPL-3.0-or-later | 保留许可/范围说明 |
| SRC-THREE | Three.js r180 / OrbitControls | MIT | 保留许可/范围说明 |
| ASSET-HEAD | 3D human parts pack / FACE2 | CC0-1.0 | 保留许可/范围说明 |
| ASSET-NASAL | Healthy nasal cavities — averaged geometry | CC BY 4.0（官方 v4 元数据与原始网格摘要已核对）。保留作者、来源、许可链接和修改说明。 | 保留许可/范围说明 |
| SRC-MEDIAPIPE | MediaPipe legacy Face Mesh / Web Face Landmarker | Apache-2.0 for code; bundled model terms require artifact audit | 保留许可/范围说明 |
| ASSET-DOULOS | Doulos SIL 字体 | SIL Open Font License 1.1; embedded license extracted without modification | 保留许可/范围说明 |
| SRC-SHEETJS | SheetJS | Apache-2.0 | 保留许可/范围说明 |
| PKG-NUMPY | numpy | BSD-3-Clause；实际 wheel 的原生组件与运行库许可另列 | 保留许可/范围说明 |
| PKG-SCIPY | scipy | BSD-3-Clause；实际 wheel 的原生组件与运行库许可另列 | 保留许可/范围说明 |
| PKG-PANDAS | pandas | BSD-3-Clause | 保留许可/范围说明 |
| PKG-MATPLOTLIB | matplotlib | Matplotlib 自有许可及历史组件许可，见原文；不能将 PSF 风格概括当成全部组件的单一许可 | 保留许可/范围说明 |
| PKG-PYQT6 | PyQt6 | GPLv3 / Riverbank 商业许可；PyPI 开源版本须遵守 GPL，Qt 与 Chromium 等组件许可另列 | 保留许可/范围说明 |
| PKG-PYQTGRAPH | pyqtgraph | MIT | 保留许可/范围说明 |
| PKG-PRAAT-PARSELMOUTH | praat-parselmouth | GPL-3.0-or-later；Praat 及其依赖另列 | 保留许可/范围说明 |
| PKG-PYWAVELETS | PyWavelets | MIT AND BSD-3-Clause | 保留许可/范围说明 |
| PKG-SOUNDDEVICE | sounddevice | MIT；PortAudio 等原生组件许可另列 | 保留许可/范围说明 |
| PKG-SOUNDFILE | soundfile | BSD-3-Clause；libsndfile 等原生组件许可另列 | 保留许可/范围说明 |
| PKG-OPENPYXL | openpyxl | MIT | 保留许可/范围说明 |
| PKG-PYTHON-DOCX | python-docx | MIT | 保留许可/范围说明 |
| PKG-OPENCV-CONTRIB-PYTHON | opencv-contrib-python | Apache-2.0；打包的第三方组件见 LICENSE-3RD-PARTY.txt | 保留许可/范围说明 |
| PKG-MEDIAPIPE | mediapipe | Apache-2.0；模型和原生依赖按具体资源分别核对 | 保留许可/范围说明 |
| PKG-PILLOW | Pillow | MIT-CMU | 保留许可/范围说明 |
| PKG-PYINSTALLER | PyInstaller | GPL-2.0-or-later，含 bootloader 分发例外及 Apache-2.0 runtime hooks；构建工具许可不自动赋予应用相同许可 | 保留许可/范围说明 |
| P01-PY-ALTGRAPH | altgraph | MIT | 保留许可/范围说明 |
| P01-PY-COLORAMA | colorama | BSD-3-Clause | 保留许可/范围说明 |
| P01-PY-INICONFIG | iniconfig | MIT | 保留许可/范围说明 |
| P01-PY-NUMPY | numpy | BSD-3-Clause；wheel 内原生组件与运行库许可另列。 | 保留许可/范围说明 |
| P01-PY-PACKAGING | packaging | Apache-2.0 OR BSD-2-Clause | 保留许可/范围说明 |
| P01-PY-PEFILE | pefile | MIT | 保留许可/范围说明 |
| P01-PY-PLUGGY | pluggy | MIT | 保留许可/范围说明 |
| P01-PY-PSUTIL | psutil | BSD-3-Clause | 保留许可/范围说明 |
| P01-PY-PYAUDIOWPATCH | PyAudioWPatch | Apache-2.0 license | 保留许可/范围说明 |
| P01-PY-PYGMENTS | Pygments | BSD-2-Clause | 保留许可/范围说明 |
| P01-PY-PYINSTALLER | pyinstaller | GPLv2-or-later with a special exception which allows to use PyInstaller to build and distribute non-free programs (including commercial ones) | 保留许可/范围说明 |
| P01-PY-PYINSTALLER-HOOKS-CONTRIB | pyinstaller-hooks-contrib | 普通构建 hooks：GPL-2.0-or-later；随应用打包的 runtime hooks：Apache-2.0。 | 保留许可/范围说明 |
| P01-PY-PYQT6 | PyQt6 | GPL-3.0-only | 保留许可/范围说明 |
| P01-PY-PYQT6-QT6 | PyQt6-Qt6 | LGPL v3 | 保留许可/范围说明 |
| P01-PY-PYQT6-WEBENGINE | PyQt6-WebEngine | GPL-3.0-only | 保留许可/范围说明 |
| P01-PY-PYQT6-WEBENGINE-QT6 | PyQt6-WebEngine-Qt6 | LGPL v3 | 保留许可/范围说明 |
| P01-PY-PYQT6-SIP | PyQt6_sip | BSD-2-Clause | 保留许可/范围说明 |
| P01-PY-PYTEST | pytest | MIT | 保留许可/范围说明 |
| P01-PY-PYWIN32-CTYPES | pywin32-ctypes | BSD-3-Clause | 保留许可/范围说明 |
| P01-PY-SETUPTOOLS | setuptools | MIT | 保留许可/范围说明 |
| P01-NPM-BABEL-HELPER-STRING-PARSER | @babel/helper-string-parser | MIT | 保留许可/范围说明 |
| P01-NPM-BABEL-HELPER-VALIDATOR-IDENTIFIER | @babel/helper-validator-identifier | MIT | 保留许可/范围说明 |
| P01-NPM-BABEL-PARSER | @babel/parser | MIT | 保留许可/范围说明 |
| P01-NPM-BABEL-TYPES | @babel/types | MIT | 保留许可/范围说明 |
| P01-NPM-JRIDGEWELL-SOURCEMAP-CODEC | @jridgewell/sourcemap-codec | MIT | 保留许可/范围说明 |
| P01-NPM-OXC-PROJECT-TYPES | @oxc-project/types | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-ANDROID-ARM-EABI | @rolldown/binding-android-arm-eabi | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-ANDROID-ARM64 | @rolldown/binding-android-arm64 | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-DARWIN-ARM64 | @rolldown/binding-darwin-arm64 | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-DARWIN-X64 | @rolldown/binding-darwin-x64 | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-FREEBSD-X64 | @rolldown/binding-freebsd-x64 | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-LINUX-ARM-GNUEABIHF | @rolldown/binding-linux-arm-gnueabihf | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-LINUX-ARM64-GNU | @rolldown/binding-linux-arm64-gnu | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-LINUX-ARM64-MUSL | @rolldown/binding-linux-arm64-musl | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-LINUX-PPC64-GNU | @rolldown/binding-linux-ppc64-gnu | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-LINUX-S390X-GNU | @rolldown/binding-linux-s390x-gnu | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-LINUX-X64-GNU | @rolldown/binding-linux-x64-gnu | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-LINUX-X64-MUSL | @rolldown/binding-linux-x64-musl | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-OPENHARMONY-ARM64 | @rolldown/binding-openharmony-arm64 | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-WIN32-ARM64-MSVC | @rolldown/binding-win32-arm64-msvc | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-BINDING-WIN32-X64-MSVC | @rolldown/binding-win32-x64-msvc | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN-PLUGINUTILS | @rolldown/pluginutils | MIT | 保留许可/范围说明 |
| P01-NPM-VITEJS-PLUGIN-VUE | @vitejs/plugin-vue | MIT | 保留许可/范围说明 |
| P01-NPM-VOLAR-LANGUAGE-CORE | @volar/language-core | MIT | 保留许可/范围说明 |
| P01-NPM-VOLAR-SOURCE-MAP | @volar/source-map | MIT | 保留许可/范围说明 |
| P01-NPM-VOLAR-TYPESCRIPT | @volar/typescript | MIT | 保留许可/范围说明 |
| P01-NPM-VUE-COMPILER-CORE | @vue/compiler-core | MIT | 保留许可/范围说明 |
| P01-NPM-VUE-COMPILER-DOM | @vue/compiler-dom | MIT | 保留许可/范围说明 |
| P01-NPM-VUE-COMPILER-SFC | @vue/compiler-sfc | MIT | 保留许可/范围说明 |
| P01-NPM-VUE-COMPILER-SSR | @vue/compiler-ssr | MIT | 保留许可/范围说明 |
| P01-NPM-VUE-LANGUAGE-CORE | @vue/language-core | MIT | 保留许可/范围说明 |
| P01-NPM-VUE-REACTIVITY | @vue/reactivity | MIT | 保留许可/范围说明 |
| P01-NPM-VUE-RUNTIME-CORE | @vue/runtime-core | MIT | 保留许可/范围说明 |
| P01-NPM-VUE-RUNTIME-DOM | @vue/runtime-dom | MIT | 保留许可/范围说明 |
| P01-NPM-VUE-SERVER-RENDERER | @vue/server-renderer | MIT | 保留许可/范围说明 |
| P01-NPM-VUE-SHARED | @vue/shared | MIT | 保留许可/范围说明 |
| P01-NPM-ALIEN-SIGNALS | alien-signals | MIT | 保留许可/范围说明 |
| P01-NPM-CSSTYPE | csstype | MIT | 保留许可/范围说明 |
| P01-NPM-DETECT-LIBC | detect-libc | Apache-2.0 | 保留许可/范围说明 |
| P01-NPM-ENTITIES | entities | BSD-2-Clause | 保留许可/范围说明 |
| P01-NPM-ESTREE-WALKER | estree-walker | MIT | 保留许可/范围说明 |
| P01-NPM-FDIR | fdir | MIT | 保留许可/范围说明 |
| P01-NPM-FSEVENTS | fsevents | MIT | 保留许可/范围说明 |
| P01-NPM-LIGHTNINGCSS | lightningcss | MPL-2.0 | 保留许可/范围说明 |
| P01-NPM-LIGHTNINGCSS-ANDROID-ARM64 | lightningcss-android-arm64 | MPL-2.0 | 保留许可/范围说明 |
| P01-NPM-LIGHTNINGCSS-DARWIN-ARM64 | lightningcss-darwin-arm64 | MPL-2.0 | 保留许可/范围说明 |
| P01-NPM-LIGHTNINGCSS-DARWIN-X64 | lightningcss-darwin-x64 | MPL-2.0 | 保留许可/范围说明 |
| P01-NPM-LIGHTNINGCSS-FREEBSD-X64 | lightningcss-freebsd-x64 | MPL-2.0 | 保留许可/范围说明 |
| P01-NPM-LIGHTNINGCSS-LINUX-ARM-GNUEABIHF | lightningcss-linux-arm-gnueabihf | MPL-2.0 | 保留许可/范围说明 |
| P01-NPM-LIGHTNINGCSS-LINUX-ARM64-GNU | lightningcss-linux-arm64-gnu | MPL-2.0 | 保留许可/范围说明 |
| P01-NPM-LIGHTNINGCSS-LINUX-ARM64-MUSL | lightningcss-linux-arm64-musl | MPL-2.0 | 保留许可/范围说明 |
| P01-NPM-LIGHTNINGCSS-LINUX-X64-GNU | lightningcss-linux-x64-gnu | MPL-2.0 | 保留许可/范围说明 |
| P01-NPM-LIGHTNINGCSS-LINUX-X64-MUSL | lightningcss-linux-x64-musl | MPL-2.0 | 保留许可/范围说明 |
| P01-NPM-LIGHTNINGCSS-WIN32-ARM64-MSVC | lightningcss-win32-arm64-msvc | MPL-2.0 | 保留许可/范围说明 |
| P01-NPM-LIGHTNINGCSS-WIN32-X64-MSVC | lightningcss-win32-x64-msvc | MPL-2.0 | 保留许可/范围说明 |
| P01-NPM-MAGIC-STRING | magic-string | MIT | 保留许可/范围说明 |
| P01-NPM-MUGGLE-STRING | muggle-string | MIT | 保留许可/范围说明 |
| P01-NPM-NANOID | nanoid | MIT | 保留许可/范围说明 |
| P01-NPM-PATH-BROWSERIFY | path-browserify | MIT | 保留许可/范围说明 |
| P01-NPM-PICOCOLORS | picocolors | ISC | 保留许可/范围说明 |
| P01-NPM-PICOMATCH | picomatch | MIT | 保留许可/范围说明 |
| P01-NPM-POSTCSS | postcss | MIT | 保留许可/范围说明 |
| P01-NPM-ROLLDOWN | rolldown | MIT | 保留许可/范围说明 |
| P01-NPM-SOURCE-MAP-JS | source-map-js | BSD-3-Clause | 保留许可/范围说明 |
| P01-NPM-TINYGLOBBY | tinyglobby | MIT | 保留许可/范围说明 |
| P01-NPM-TYPESCRIPT | typescript | Apache-2.0 | 保留许可/范围说明 |
| P01-NPM-VITE | vite | MIT | 保留许可/范围说明 |
| P01-NPM-VSCODE-URI | vscode-uri | MIT | 保留许可/范围说明 |
| P01-NPM-VUE | vue | MIT | 保留许可/范围说明 |
| P01-NPM-VUE-TSC | vue-tsc | MIT | 保留许可/范围说明 |
| P01-CPYTHON | CPython via python-build-standalone | PSF-2.0 and bundled component terms; runtime distribution audit pending | 保留许可/范围说明 |
| P02-PY-ANNOTATED-DOC | annotated-doc | MIT | 保留许可/范围说明 |
| P02-PY-ANNOTATED-TYPES | annotated-types | MIT | 保留许可/范围说明 |
| P02-PY-ANYIO | anyio | MIT | 保留许可/范围说明 |
| P02-PY-BUILD | build | MIT | 保留许可/范围说明 |
| P02-PY-CERTIFI | certifi | MPL-2.0 | 保留许可/范围说明 |
| P02-PY-CLICK | click | BSD-3-Clause | 保留许可/范围说明 |
| P02-PY-COLORAMA | colorama | BSD-3-Clause | 保留许可/范围说明 |
| P02-PY-FASTAPI | fastapi | MIT | 保留许可/范围说明 |
| P02-PY-H11 | h11 | MIT | 保留许可/范围说明 |
| P02-PY-HTTPCORE | httpcore | BSD-3-Clause | 保留许可/范围说明 |
| P02-PY-HTTPX | httpx | BSD-3-Clause | 保留许可/范围说明 |
| P02-PY-IDNA | idna | BSD-3-Clause | 保留许可/范围说明 |
| P02-PY-INICONFIG | iniconfig | MIT | 保留许可/范围说明 |
| P02-PY-PACKAGING | packaging | Apache-2.0 OR BSD-2-Clause | 保留许可/范围说明 |
| P02-PY-PLUGGY | pluggy | MIT | 保留许可/范围说明 |
| P02-PY-PYDANTIC | pydantic | MIT | 保留许可/范围说明 |
| P02-PY-PYDANTIC-CORE | pydantic_core | MIT | 保留许可/范围说明 |
| P02-PY-PYGMENTS | Pygments | BSD-2-Clause | 保留许可/范围说明 |
| P02-PY-PYPROJECT-HOOKS | pyproject_hooks | MIT | 保留许可/范围说明 |
| P02-PY-PYQT6 | PyQt6 | GPL-3.0-only | 保留许可/范围说明 |
| P02-PY-PYQT6-QT6 | PyQt6-Qt6 | LGPL v3 | 保留许可/范围说明 |
| P02-PY-PYQT6-WEBENGINE | PyQt6-WebEngine | GPL-3.0-only | 保留许可/范围说明 |
| P02-PY-PYQT6-WEBENGINE-QT6 | PyQt6-WebEngine-Qt6 | LGPL v3 | 保留许可/范围说明 |
| P02-PY-PYQT6-SIP | PyQt6_sip | BSD-2-Clause | 保留许可/范围说明 |
| P02-PY-PYTEST | pytest | MIT | 保留许可/范围说明 |
| P02-PY-SETUPTOOLS | setuptools | MIT | 保留许可/范围说明 |
| P02-PY-STARLETTE | starlette | BSD-3-Clause | 保留许可/范围说明 |
| P02-PY-TYPING-INSPECTION | typing-inspection | MIT | 保留许可/范围说明 |
| P02-PY-TYPING-EXTENSIONS | typing_extensions | PSF-2.0 | 保留许可/范围说明 |
| P02-PY-UVICORN | uvicorn | BSD-3-Clause | 保留许可/范围说明 |
| P02-PY-WHEEL | wheel | MIT | 保留许可/范围说明 |
| P02-NPM-BABEL-CODE-FRAME-7-29-7 | @babel/code-frame | MIT | 保留许可/范围说明 |
| P02-NPM-BABEL-HELPER-STRING-PARSER-7-29-7 | @babel/helper-string-parser | MIT | 保留许可/范围说明 |
| P02-NPM-BABEL-HELPER-VALIDATOR-IDENTIFIER-7-29-7 | @babel/helper-validator-identifier | MIT | 保留许可/范围说明 |
| P02-NPM-BABEL-PARSER-7-29-8 | @babel/parser | MIT | 保留许可/范围说明 |
| P02-NPM-BABEL-TYPES-7-29-8 | @babel/types | MIT | 保留许可/范围说明 |
| P02-NPM-JRIDGEWELL-SOURCEMAP-CODEC-1-6-0 | @jridgewell/sourcemap-codec | MIT | 保留许可/范围说明 |
| P02-NPM-OXC-PROJECT-TYPES-0-148-0 | @oxc-project/types | MIT | 保留许可/范围说明 |
| P02-NPM-REDOCLY-AJV-8-11-2 | @redocly/ajv | MIT | 保留许可/范围说明 |
| P02-NPM-REDOCLY-CONFIG-0-22-0 | @redocly/config | MIT | 保留许可/范围说明 |
| P02-NPM-REDOCLY-OPENAPI-CORE-1-34-19 | @redocly/openapi-core | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-ANDROID-ARM-EABI-1-2-7 | @rolldown/binding-android-arm-eabi | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-ANDROID-ARM64-1-2-7 | @rolldown/binding-android-arm64 | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-DARWIN-ARM64-1-2-7 | @rolldown/binding-darwin-arm64 | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-DARWIN-X64-1-2-7 | @rolldown/binding-darwin-x64 | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-FREEBSD-X64-1-2-7 | @rolldown/binding-freebsd-x64 | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-LINUX-ARM-GNUEABIHF-1-2-7 | @rolldown/binding-linux-arm-gnueabihf | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-LINUX-ARM64-GNU-1-2-7 | @rolldown/binding-linux-arm64-gnu | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-LINUX-ARM64-MUSL-1-2-7 | @rolldown/binding-linux-arm64-musl | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-LINUX-PPC64-GNU-1-2-7 | @rolldown/binding-linux-ppc64-gnu | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-LINUX-S390X-GNU-1-2-7 | @rolldown/binding-linux-s390x-gnu | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-LINUX-X64-GNU-1-2-7 | @rolldown/binding-linux-x64-gnu | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-LINUX-X64-MUSL-1-2-7 | @rolldown/binding-linux-x64-musl | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-OPENHARMONY-ARM64-1-2-7 | @rolldown/binding-openharmony-arm64 | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-WIN32-ARM64-MSVC-1-2-7 | @rolldown/binding-win32-arm64-msvc | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-BINDING-WIN32-X64-MSVC-1-2-7 | @rolldown/binding-win32-x64-msvc | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-PLUGINUTILS-1-0-1 | @rolldown/pluginutils | MIT | 保留许可/范围说明 |
| P02-NPM-VITEJS-PLUGIN-VUE-6-0-8 | @vitejs/plugin-vue | MIT | 保留许可/范围说明 |
| P02-NPM-VOLAR-LANGUAGE-CORE-2-4-28 | @volar/language-core | MIT | 保留许可/范围说明 |
| P02-NPM-VOLAR-SOURCE-MAP-2-4-28 | @volar/source-map | MIT | 保留许可/范围说明 |
| P02-NPM-VOLAR-TYPESCRIPT-2-4-28 | @volar/typescript | MIT | 保留许可/范围说明 |
| P02-NPM-VUE-COMPILER-CORE-3-5-42 | @vue/compiler-core | MIT | 保留许可/范围说明 |
| P02-NPM-VUE-COMPILER-DOM-3-5-42 | @vue/compiler-dom | MIT | 保留许可/范围说明 |
| P02-NPM-VUE-COMPILER-SFC-3-5-42 | @vue/compiler-sfc | MIT | 保留许可/范围说明 |
| P02-NPM-VUE-COMPILER-SSR-3-5-42 | @vue/compiler-ssr | MIT | 保留许可/范围说明 |
| P02-NPM-VUE-LANGUAGE-CORE-3-3-11 | @vue/language-core | MIT | 保留许可/范围说明 |
| P02-NPM-VUE-REACTIVITY-3-5-42 | @vue/reactivity | MIT | 保留许可/范围说明 |
| P02-NPM-VUE-RUNTIME-CORE-3-5-42 | @vue/runtime-core | MIT | 保留许可/范围说明 |
| P02-NPM-VUE-RUNTIME-DOM-3-5-42 | @vue/runtime-dom | MIT | 保留许可/范围说明 |
| P02-NPM-VUE-SERVER-RENDERER-3-5-42 | @vue/server-renderer | MIT | 保留许可/范围说明 |
| P02-NPM-VUE-SHARED-3-5-42 | @vue/shared | MIT | 保留许可/范围说明 |
| P02-NPM-AGENT-BASE-7-1-4 | agent-base | MIT | 保留许可/范围说明 |
| P02-NPM-ALIEN-SIGNALS-3-2-1 | alien-signals | MIT | 保留许可/范围说明 |
| P02-NPM-ANSI-COLORS-4-1-3 | ansi-colors | MIT | 保留许可/范围说明 |
| P02-NPM-ARGPARSE-2-0-1 | argparse | Python-2.0 | 保留许可/范围说明 |
| P02-NPM-BALANCED-MATCH-1-0-2 | balanced-match | MIT | 保留许可/范围说明 |
| P02-NPM-BRACE-EXPANSION-2-1-4 | brace-expansion | MIT | 保留许可/范围说明 |
| P02-NPM-CHANGE-CASE-5-4-4 | change-case | MIT | 保留许可/范围说明 |
| P02-NPM-COLORETTE-1-4-0 | colorette | MIT | 保留许可/范围说明 |
| P02-NPM-CSSTYPE-3-2-3 | csstype | MIT | 保留许可/范围说明 |
| P02-NPM-DEBUG-4-4-3 | debug | MIT | 保留许可/范围说明 |
| P02-NPM-DETECT-LIBC-2-1-2 | detect-libc | Apache-2.0 | 保留许可/范围说明 |
| P02-NPM-ENTITIES-7-0-1 | entities | BSD-2-Clause | 保留许可/范围说明 |
| P02-NPM-ESTREE-WALKER-2-0-2 | estree-walker | MIT | 保留许可/范围说明 |
| P02-NPM-FAST-DEEP-EQUAL-3-1-3 | fast-deep-equal | MIT | 保留许可/范围说明 |
| P02-NPM-FDIR-6-5-0 | fdir | MIT | 保留许可/范围说明 |
| P02-NPM-FSEVENTS-2-3-3 | fsevents | MIT | 保留许可/范围说明 |
| P02-NPM-HTTPS-PROXY-AGENT-7-0-6 | https-proxy-agent | MIT | 保留许可/范围说明 |
| P02-NPM-INDEX-TO-POSITION-1-2-0 | index-to-position | MIT | 保留许可/范围说明 |
| P02-NPM-JS-LEVENSHTEIN-1-1-6 | js-levenshtein | MIT | 保留许可/范围说明 |
| P02-NPM-JS-TOKENS-4-0-0 | js-tokens | MIT | 保留许可/范围说明 |
| P02-NPM-JS-YAML-4-3-2 | js-yaml | MIT | 保留许可/范围说明 |
| P02-NPM-JSON-SCHEMA-TRAVERSE-1-0-0 | json-schema-traverse | MIT | 保留许可/范围说明 |
| P02-NPM-LIGHTNINGCSS-1-33-0 | lightningcss | MPL-2.0 | 保留许可/范围说明 |
| P02-NPM-LIGHTNINGCSS-ANDROID-ARM64-1-33-0 | lightningcss-android-arm64 | MPL-2.0 | 保留许可/范围说明 |
| P02-NPM-LIGHTNINGCSS-DARWIN-ARM64-1-33-0 | lightningcss-darwin-arm64 | MPL-2.0 | 保留许可/范围说明 |
| P02-NPM-LIGHTNINGCSS-DARWIN-X64-1-33-0 | lightningcss-darwin-x64 | MPL-2.0 | 保留许可/范围说明 |
| P02-NPM-LIGHTNINGCSS-FREEBSD-X64-1-33-0 | lightningcss-freebsd-x64 | MPL-2.0 | 保留许可/范围说明 |
| P02-NPM-LIGHTNINGCSS-LINUX-ARM-GNUEABIHF-1-33-0 | lightningcss-linux-arm-gnueabihf | MPL-2.0 | 保留许可/范围说明 |
| P02-NPM-LIGHTNINGCSS-LINUX-ARM64-GNU-1-33-0 | lightningcss-linux-arm64-gnu | MPL-2.0 | 保留许可/范围说明 |
| P02-NPM-LIGHTNINGCSS-LINUX-ARM64-MUSL-1-33-0 | lightningcss-linux-arm64-musl | MPL-2.0 | 保留许可/范围说明 |
| P02-NPM-LIGHTNINGCSS-LINUX-X64-GNU-1-33-0 | lightningcss-linux-x64-gnu | MPL-2.0 | 保留许可/范围说明 |
| P02-NPM-LIGHTNINGCSS-LINUX-X64-MUSL-1-33-0 | lightningcss-linux-x64-musl | MPL-2.0 | 保留许可/范围说明 |
| P02-NPM-LIGHTNINGCSS-WIN32-ARM64-MSVC-1-33-0 | lightningcss-win32-arm64-msvc | MPL-2.0 | 保留许可/范围说明 |
| P02-NPM-LIGHTNINGCSS-WIN32-X64-MSVC-1-33-0 | lightningcss-win32-x64-msvc | MPL-2.0 | 保留许可/范围说明 |
| P02-NPM-MAGIC-STRING-0-30-21 | magic-string | MIT | 保留许可/范围说明 |
| P02-NPM-MINIMATCH-5-1-9 | minimatch | ISC | 保留许可/范围说明 |
| P02-NPM-MS-2-1-3 | ms | MIT | 保留许可/范围说明 |
| P02-NPM-MUGGLE-STRING-0-4-1 | muggle-string | MIT | 保留许可/范围说明 |
| P02-NPM-NANOID-3-3-18 | nanoid | MIT | 保留许可/范围说明 |
| P02-NPM-OPENAPI-TYPESCRIPT-7-13-0 | openapi-typescript | MIT | 保留许可/范围说明 |
| P02-NPM-PARSE-JSON-8-3-0 | parse-json | MIT | 保留许可/范围说明 |
| P02-NPM-PATH-BROWSERIFY-1-0-1 | path-browserify | MIT | 保留许可/范围说明 |
| P02-NPM-PICOCOLORS-1-1-1 | picocolors | ISC | 保留许可/范围说明 |
| P02-NPM-PICOMATCH-4-0-7 | picomatch | MIT | 保留许可/范围说明 |
| P02-NPM-PLURALIZE-8-0-0 | pluralize | MIT | 保留许可/范围说明 |
| P02-NPM-POSTCSS-8-5-28 | postcss | MIT | 保留许可/范围说明 |
| P02-NPM-REQUIRE-FROM-STRING-2-0-2 | require-from-string | MIT | 保留许可/范围说明 |
| P02-NPM-ROLLDOWN-1-2-7 | rolldown | MIT | 保留许可/范围说明 |
| P02-NPM-SOURCE-MAP-JS-1-2-1 | source-map-js | BSD-3-Clause | 保留许可/范围说明 |
| P02-NPM-SUPPORTS-COLOR-10-2-2 | supports-color | MIT | 保留许可/范围说明 |
| P02-NPM-TINYGLOBBY-0-2-17 | tinyglobby | MIT | 保留许可/范围说明 |
| P02-NPM-TYPE-FEST-4-41-0 | type-fest | (MIT OR CC0-1.0) | 保留许可/范围说明 |
| P02-NPM-TYPESCRIPT-5-9-3 | typescript | Apache-2.0 | 保留许可/范围说明 |
| P02-NPM-URI-JS-REPLACE-1-0-1 | uri-js-replace | MIT | 保留许可/范围说明 |
| P02-NPM-VITE-8-2-2 | vite | MIT | 保留许可/范围说明 |
| P02-NPM-VSCODE-URI-3-2-0 | vscode-uri | MIT | 保留许可/范围说明 |
| P02-NPM-VUE-3-5-42 | vue | MIT | 保留许可/范围说明 |
| P02-NPM-VUE-TSC-3-3-11 | vue-tsc | MIT | 保留许可/范围说明 |
| P02-NPM-YAML-AST-PARSER-0-0-43 | yaml-ast-parser | Apache-2.0 | 保留许可/范围说明 |
| P02-NPM-YARGS-PARSER-21-1-1 | yargs-parser | ISC | 保留许可/范围说明 |
| P02-CPYTHON | CPython via python-build-standalone | PSF-2.0 and bundled component terms; runtime distribution audit pending | 保留许可/范围说明 |
| P05-PY-ARGON2-CFFI | argon2-cffi | MIT | 保留许可/范围说明 |
| P05-PY-ARGON2-CFFI-BINDINGS | argon2-cffi-bindings | MIT | 保留许可/范围说明 |
| P05-PY-CFFI | cffi | MIT-0 | 保留许可/范围说明 |
| P05-PY-ITSDANGEROUS | itsdangerous | BSD-3-Clause | 保留许可/范围说明 |
| P05-PY-PSYCOPG | psycopg | LGPL-3.0-only | 保留许可/范围说明 |
| P05-PY-PSYCOPG-BINARY | psycopg-binary | LGPL-3.0-only | 保留许可/范围说明 |
| P05-PY-PYCPARSER | pycparser | BSD-3-Clause | 保留许可/范围说明 |
| P05-PY-TZDATA | tzdata | Apache-2.0 | 保留许可/范围说明 |
| P05-POSTGRESQL-TEST | PostgreSQL (P05 isolated database validation) | PostgreSQL; bundled native dependencies have separate licenses | 保留许可/范围说明 |
| P06-SQLITE | SQLite (CPython bundled local task store) | Public domain; bundled with existing CPython runtime | 保留许可/范围说明 |
| P07-PLAYWRIGHT-TEST | Playwright independent browser verification | Apache-2.0 | 保留许可/范围说明 |
| P07-PYTHON-ZIP | Python zipfile standard library and ZIP API documentation | PSF-2.0 (existing CPython runtime) | 保留许可/范围说明 |
| M01-PY-BUILD | build | MIT | 保留许可/范围说明 |
| M01-PY-COLORAMA | colorama | BSD-3-Clause | 保留许可/范围说明 |
| M01-PY-INICONFIG | iniconfig | MIT | 保留许可/范围说明 |
| M01-PY-NUMPY | numpy | BSD-3-Clause；wheel 内 OpenBLAS/LAPACK/运行库等独立许可与例外须一并保留。 | 保留许可/范围说明 |
| M01-PY-PACKAGING | packaging | Apache-2.0 OR BSD-2-Clause | 保留许可/范围说明 |
| M01-PY-PANDAS | pandas | BSD-3-Clause；保留实际发行文件的版权和第三方通知。 | 保留许可/范围说明 |
| M01-PY-PLUGGY | pluggy | MIT | 保留许可/范围说明 |
| M01-PY-PRAAT-PARSELMOUTH | praat-parselmouth | GPL-3.0-or-later（Parselmouth 0.4.7）；内嵌 Praat 与依赖许可另列。 | 保留许可/范围说明 |
| M01-PY-PYGMENTS | Pygments | BSD-2-Clause | 保留许可/范围说明 |
| M01-PY-PYPROJECT-HOOKS | pyproject_hooks | MIT | 保留许可/范围说明 |
| M01-PY-PYTEST | pytest | MIT | 保留许可/范围说明 |
| M01-PY-PYTHON-DATEUTIL | python-dateutil | Apache-2.0（2017-12-01 后及已重新许可的贡献）与 BSD-3-Clause（早期贡献），按文件与原文范围保留。 | 保留许可/范围说明 |
| M01-PY-PYTZ | pytz | MIT | 保留许可/范围说明 |
| M01-PY-SCIPY | scipy | BSD-3-Clause；wheel 内原生组件、运行库许可与例外须一并保留。 | 保留许可/范围说明 |
| M01-PY-SETUPTOOLS | setuptools | MIT | 保留许可/范围说明 |
| M01-PY-SIX | six | MIT | 保留许可/范围说明 |
| M01-PY-TZDATA | tzdata | Apache-2.0 | 保留许可/范围说明 |
| M01-PY-WHEEL | wheel | MIT | 保留许可/范围说明 |
| M01-PY-OPENPYXL | openpyxl | MIT | 保留许可/范围说明 |
| M01-PY-ET-XMLFILE | et-xmlfile | MIT | 保留许可/范围说明 |
| M04-PY-NUMPY | NumPy | See existing installed distribution licenses; no new library redistribution in this stage | 保留许可/范围说明 |
| M04-PY-SCIPY | SciPy | See existing installed distribution licenses; no new library redistribution in this stage | 保留许可/范围说明 |
| ASSET-M05-NASA | NASA Eileen Collins portrait via scikit-image | NASA public-domain image; no endorsement or commercial likeness claim | 保留许可/范围说明 |
| M17-IPA-CHART | International Phonetic Alphabet chart — 2026 reprint | CC BY-SA 4.0 | 保留许可/范围说明 |
| M17-EXTIPA-2025 | Extensions to the IPA — 2025 chart | CC BY-SA 3.0 as identified on ICPLA resources; original chart reproduction wording also recorded | 保留许可/范围说明 |
| M17-VOQS-ZH-UNTPHESOCA | VoQS：音质符号（2016 中英双语版） | CC BY 4.0（Zenodo 对应中文译表记录）；保留作者、来源、许可链接及交互改写说明，原表作者另列，原 PDF 不随包。 | 保留许可/范围说明 |
| M17-PTB-IPA-PLUS-FONT | PTB IPA Plus 1.000 — module font derived from Doulos SIL 7.000 | SIL Open Font License 1.1 | 保留许可/范围说明 |
| SRC-JETBRAINS-MONO | JetBrains Mono | SIL Open Font License 1.1 | 保留许可/范围说明 |

### 已获邮件授权（1 条）

| ID | 名称 | 当前登记许可或范围 | 界面处理 |
| --- | --- | --- | --- |
| SRC-ZAIWA | 载瓦语 F0 与发声类型研究代码/数据 | （2026-09-10）作者邮件许可，允许相关 MATLAB 代码改写为 Python，集成至 PhoneticToolbox v3 免费开源发布，并按约定引用、署名致谢和说明改写。 | 保留许可/范围说明 |

### 保留关联代码授权说明（1 条）

| ID | 名称 | 当前登记许可或范围 | 界面处理 |
| --- | --- | --- | --- |
| REF-ZAIWA | F0 与发声类型对载瓦语声调感知的贡献 | article license is separate from code/data | 保留许可/范围说明 |

### 代码适用许可待确认（2 条）

| ID | 名称 | 当前登记许可或范围 | 界面处理 |
| --- | --- | --- | --- |
| SRC-VOICESAUCE | VoiceSauce | 已核对的 OpenSauce Octave 实现采用 BSD 两条款许可；原 VoiceSauce MATLAB 与 SoE 的具体适用许可仍需按函数核对。 | 保留许可/范围说明 |
| REF-SOE | Strength of Excitation / epoch extraction | SoE 实现明确参考 Soo Jin Park 的 func_getSoE.m；尚未找到覆盖该实现改写及分发的明确许可，需核对或向权利人确认。 | 保留许可/范围说明 |

### 具体发行物待核对（2 条）

| ID | 名称 | 当前登记许可或范围 | 界面处理 |
| --- | --- | --- | --- |
| SRC-FFMPEG | FFmpeg | build-dependent; not yet verified | 保留许可/范围说明 |
| SRC-MFA | Montreal Forced Aligner / Kaldi / models | each executable/model/dictionary must be checked | 保留许可/范围说明 |

### 自有或生成材料（11 条）

| ID | 名称 | 当前登记许可或范围 | 界面处理 |
| --- | --- | --- | --- |
| PENDING-EGG | EGG 事件分析/简化逆滤波方法 | 本项目自有实现或整理成果；项目总许可证由作者确定，实际第三方依赖及复用材料按各自登记许可执行。 | 本条隐藏 |
| PENDING-DICTIONARY | 内置普通话发音词典 | 本项目自有实现或整理成果；项目总许可证由作者确定，实际第三方依赖及复用材料按各自登记许可执行。 | 本条隐藏 |
| PENDING-IPA | 11 套普通话转换规则及映射数据 | 本项目自有实现或整理成果；项目总许可证由作者确定，实际第三方依赖及复用材料按各自登记许可执行。 | 本条隐藏 |
| PENDING-PHONOLOGY | 音系归纳规则和帮助材料 | 本项目自有实现或整理成果；项目总许可证由作者确定，实际第三方依赖及复用材料按各自登记许可执行。 | 本条隐藏 |
| ORIGIN-WEBEDITOR | 本地 pitch_perception 的 web_praat_editor 来源链 | 本项目自有实现或整理成果；项目总许可证由作者确定，实际第三方依赖及复用材料按各自登记许可执行。 | 本条隐藏 |
| ASSET-MASCOT | K2 蓝色波形团子与 U2 设计参考 | not-established | 许可说明隐藏 |
| PROJECT-SYNTHETIC-WAV | PhoneticToolbox 公开解析测试波形 | Project-authored test signal; no third-party recording | 本条隐藏 |
| PROJECT-M10 | PhoneticToolbox 声道界面与平台适配 | Project integration; native-derived adapter GPL-3.0-or-later; third-party components separately listed | 本条隐藏 |
| PROJECT-M05 | PhoneticToolbox 唇形指标迁移与本地采集适配 | Project integration; third-party components separately listed | 本条隐藏 |
| SRC-M16-VOICEVISTA | VoiceVista egg_recorder 录音安全机制 | User-authored local source; public redistribution license not established | 许可说明隐藏 |
| METHOD-M16-SPECTRAL-SUBTRACTION | M16 噪声样本功率谱减法（自有实现） | Project implementation; SciPy runtime license is separate | 本条隐藏 |

### 当前不使用（10 条）

| ID | 名称 | 当前登记许可或范围 | 界面处理 |
| --- | --- | --- | --- |
| SRC-HTML2CANVAS | html2canvas | exact-version-license-pending | 本条隐藏 |
| SRC-REACT | React | exact-version-license-pending | 本条隐藏 |
| SRC-BABEL | Babel standalone | exact-version-license-pending | 本条隐藏 |
| SRC-TAILWIND | Tailwind CSS | exact-version-license-pending | 本条隐藏 |
| SRC-LUCIDE | Lucide | exact-version-license-pending | 本条隐藏 |
| P01-PYSIDE6 | PySide6 candidate host | LGPL/GPL/commercial and component-specific terms; see Qt license inventory | 本条隐藏 |
| REF-WORLD-2016 | WORLD: A Vocoder-Based High-Quality Speech Synthesis System for Real-Time Applications | paper citation only; no redistribution licensed here | 本条隐藏 |
| REF-D4C-2016 | D4C, a band-aperiodicity estimator for high-quality speech synthesis | paper citation only; no redistribution licensed here | 本条隐藏 |
| SRC-PYWORLD | PyWORLD | MIT | 本条隐藏 |
| SRC-WORLD | WORLD (bundled with PyWORLD) | BSD-3-Clause | 本条隐藏 |

### 基本配色与主题文件分开判断（29 条）

| ID | 名称 | 当前登记许可或范围 | 界面处理 |
| --- | --- | --- | --- |
| SRC-PALETTE-ABSOLUTELY | 配色 · Absolutely（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-AYU | 配色 · Ayu（PTB 独立适配） | MIT（已核对公开上游许可原文）。保留作者与许可声明；PTB 独立适配，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-CATPPUCCIN | 配色 · Catppuccin（PTB 独立适配） | MIT（已核对公开上游许可原文）。保留作者与许可声明；PTB 独立适配，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-CODEX | 配色 · Codex（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-DRACULA | 配色 · Dracula（PTB 独立适配） | MIT（已核对公开上游许可原文）。保留作者与许可声明；PTB 独立适配，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-EVERFOREST | 配色 · Everforest（PTB 独立适配） | MIT（已核对公开上游许可原文）。保留作者与许可声明；PTB 独立适配，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-GITHUB | 配色 · GitHub（PTB 独立适配） | MIT（已核对公开上游许可原文）。保留作者与许可声明；PTB 独立适配，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-GRUVBOX | 配色 · Gruvbox（PTB 独立适配） | MIT/X11（上游 README 声明）。完整版权/许可通知尚未取得；PTB 独立适配，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-LINEAR | 配色 · Linear（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-LOBSTER | 配色 · Lobster（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-MATERIAL | 配色 · Material（PTB 独立适配） | 待确认：历史 Material Theme 上游已转向 Vira，当前未取得许可原文。未套用历史标签或其他同名项目许可。 | 保留许可/范围说明 |
| SRC-PALETTE-MATRIX | 配色 · Matrix（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-MONOKAI | 配色 · Monokai（PTB 独立适配） | VS Code 公开实现为 MIT；经典 Monokai 原设计及 Codex 此方案许可链待确认。此项为 PTB 适配，不是 Monokai Pro。 | 保留许可/范围说明 |
| SRC-PALETTE-NIGHT-OWL | 配色 · Night Owl（PTB 独立适配） | MIT（已核对公开上游许可原文）。保留作者与许可声明；PTB 独立适配，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-NORD | 配色 · Nord（PTB 独立适配） | MIT（已核对公开上游许可原文）。保留作者与许可声明；PTB 独立适配，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-NOTION | 配色 · Notion（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-OG | 配色 · OG（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-OSCURANGE | 配色 · Oscurange（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-ONE | 配色 · One（PTB 独立适配） | MIT（已核对公开上游许可原文）。保留作者与许可声明；PTB 独立适配，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-PROOF | 配色 · Proof（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-RAYCAST | 配色 · Raycast（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-ROSE-PINE | 配色 · Rose Pine（PTB 独立适配） | MIT（已核对公开上游许可原文）。保留作者与许可声明；PTB 独立适配，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-SENTRY | 配色 · Sentry（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-SOLARIZED | 配色 · Solarized（PTB 独立适配） | MIT（已核对公开上游许可原文）。保留作者与许可声明；PTB 独立适配，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-TEMPLE | 配色 · Temple（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-TOKYO-NIGHT | 配色 · Tokyo Night（PTB 独立适配） | MIT（已核对公开上游许可原文）。保留作者与许可声明；PTB 独立适配，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-VERCEL | 配色 · Vercel（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-VSCODE-PLUS | 配色 · VS Code Plus（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |
| SRC-PALETTE-XCODE | 配色 · Xcode（PTB 独立适配） | 待确认：未取得可对应此主题的公开再分发许可。PTB 独立适配，名称仅说明视觉参考，不表示官方授权或背书。 | 保留许可/范围说明 |

