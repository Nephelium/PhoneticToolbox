# PhoneticToolbox 3.0 · 开发工作台

**2026-10-04 M01 基频默认值：** 参数估计新草稿默认范围改为30–800 Hz，REAPER/Praat/WM共用，已有草稿与历史设置保留。266项前端、84项后端/契约及Chrome定向检查通过，见[报告](docs/testing/2026-10-04-m01-f0-default-report.md)。[源码入口](scripts/Start-M01-M02-Workbench.ps1)，旧EXE未更新。

**2026-10-04 唇形 M05-R3：** 统一实时预览、实时录制、高帧率先录后算三种模式，结束后另存可选视频/面部动画，实时音频和唇形始终保留。视频置右栏顶部并与放大面部同步回放，新增真实波形与参数共轴的偏移检查弹窗，支持 1 ms 连续按键调整及保存。Windows 开发态、Chrome/实际 Qt 合成输入与保存组合验证通过，见[报告](docs/testing/2026-10-04-m05-r3-report.md)。[时间审计](docs/references/m05-r3-timing-audit.md)确认旧零点叠加问题已修复，默认 0 不代表设备已校准。使用[源码工作台入口](scripts/Start-M05-Workbench.ps1)，旧 EXE 未更新，实体设备同步与 Linux 媒体链未验。下方 M05-R2 的停止自动保存流程由本轮取代。

**2026-10-04 M16 录音打磨：** 主录制/播放按钮移到提示卡，任务可直接删除并保存，导入表格提供中文说明和 CSV 示例，剪辑/降噪工具收紧。语谱图覆盖完整可见时间窗并限制为 0–5000 Hz，与波形同步拖选。Windows 开发态、实际 Qt/Chrome 合成设备验证通过，见 [报告](docs/testing/2026-10-04-m16-r3-report.md)。使用 [源码工作台入口](scripts/Start-M16-M17-Workbench.ps1)，旧 EXE 未重新打包。

**2026-10-03 M02 参数显示修复：** 参数勾选采用随栏宽变化的2/3/4列，按钮和列表收紧；波形标注保持正常字形，波形、语谱和单/双纵轴图窗横轴对齐，各图面都能直接拖选时间范围。前端246项、Chrome专项/导出/字体及实际Qt原生读取与PNG通过，见[验证报告](docs/testing/2026-10-03-m02-r2-report.md)。使用[当前源码入口](scripts/Start-M01-M02-Workbench.ps1)，本轮未生成新EXE。

**2026-10-03 M01-R2：** TextGrid切分移至左栏分析设置下方，输出参数弹窗采用四列。顶部新增全列表唇形关联，桌面默认读取新JSON与旧PKL，单条改选/取消刷新保留。前端、Chrome、实际Qt混合格式任务与原文件哈希检查通过，见[报告](docs/testing/2026-10-03-m01-r2-report.md)。使用[源码入口](scripts/Start-M01-M02-Workbench.ps1)，下列EXE尚未包含本轮修改。

**2026-10-03 P20 最新 EXE：** [当前本机试用包](dist/PhoneticToolbox-v3-Latest-20261003-R1/PhoneticToolbox-v3-Latest-20261003-R1.exe) 已包含P19/M05-R2及音标/桌面后续修复。296.52 MiB，420包内文件、10任务/15页、录音与音标11组/14布局、26子进程退出及非法参数拒绝通过，见[成品报告](docs/testing/2026-10-03-p20-exe-report.md)。仍依赖本机既有科学环境。已整理[18.48 GiB待删除清单](docs/testing/2026-10-03-p20-cleanup-plan.md)，保留实际依赖和验证报告，尚未删除。

**2026-10-03 P19 外观：** 设置改为紧凑左右两栏，新增 29 套 Codex 同名适配配色（连同原主题共30套），全部深浅配对与跟随系统。默认宋体/Times New Roman，代码内置 JetBrains Mono，IPA保持Doulos。任务栏图标可见占用由约79%提高到约94%，原图保留。242项前端、Chrome主题/布局/字体与真实导出、实际Qt离线字体/60主题组合/M10联动、2项多尺寸图标检查通过，见[报告](docs/testing/2026-10-03-p19-appearance-report.md)。[源码入口](scripts/Start-M16-M17-Workbench.ps1)，旧EXE未更新，实体任务栏/DPI与其他平台GUI未验。

**2026-10-03 唇形 M05-R2：** 默认优先麦克风，显示实际输入与电平；停止后直接保存 MP4、WAV 和可自动关联的实时唇形，保存后仍可另存，面部动画居中放大。无需先离线分析，实时模型来源保留。源码入口 [Start-M05-Workbench.ps1](scripts/Start-M05-Workbench.ps1)，见[验证报告](docs/testing/2026-10-03-m05-r2-report.md)与[浏览器计算分工说明](docs/specs/2026-10-03-client-compute-assessment.md)。本轮未打新 EXE，实体麦克风人声/同步仍待手验。

**2026-10-03 P18 精简 EXE：** [新本机试用包](dist/PhoneticToolbox-v3-P18-20261003/PhoneticToolbox-v3-P18-20261003.exe) 已包含统一布局与最新音标多列/滚动/悬浮说明。342.77 → 296.38 MiB，缩减13.53%，排除Qt调试、未使用QML及额外WebEngine语言资源，保留中文/英文、科研库和字体。419包内文件哈希、合成/自然输入各10任务、完整15页、录音/音标八组及18分区、M10原生引擎与两次各25子进程退出通过。见[成品及体积审计报告](docs/testing/2026-10-03-p18-exe-report.md)。旧包保留，仍依赖本机既有独立科学环境。

**2026-10-03 P18 / M17-R1：** 15个模块统一默认300px侧栏、工具栏、按钮与整高外框，EGG操作栏移左，用户已保存的栏宽保留。国际音标Plus新增125个组合/例示入口，总计625个；附加符号与韵律采用紧凑多列，默认仅名称/符号，解释悬浮显示，表区上下滚动且编辑框固定。VoQS指定译名保留，声道工作台未改造。231项前端、Chrome专项、Qt60组布局/真实录音预览及音标18分区/原生保存通过，WSL目录生成一致。详见[联合报告](docs/testing/2026-10-03-visual-and-ipa-report.md)。使用[源码工作台入口](scripts/Start-M16-M17-Workbench.ps1)，**旧EXE未更新**。

**2026-10-02 M16/M17 R2 EXE：** [本机试用启动文件](dist/PhoneticToolbox-v3-M16-M17-20261002-R2/PhoneticToolbox-v3-M16-M17-20261002-R2.exe) 已生成并通过成品检查：10项计算任务、15个既有页面、两新增模块8组及六布局、声道引擎真实初始化、419文件哈希、30子进程退出。录音默认双音频、EGG手选，音标页长例句码位已精简；截图中的启动弹窗原因已修正并复验。[成品报告](docs/testing/2026-10-02-m16-m17-exe-report.md)。先关闭旧版再启动；依赖本机既有独立科学环境，实体音频设备留待手验。

**2026-10-02 M16 录音 / M17 国际音标 Plus：** 新增本地录音工程、可选任务与表格导入、实时波形/语谱、可恢复剪辑/降噪、重录及批量保存；默认双声道音频，EGG 手动指定。音标页三表横向重排、固定内置字体、500个输入入口、底部编辑与本机草稿，VoQS 的56个中文名采用井井指定的 UntPhesoca 译表并纳入引用。使用 [源码工作台入口](scripts/Start-M16-M17-Workbench.ps1)，见 [联合验证](docs/testing/2026-10-02-m16-m17-integration-report.md)、[录音手册](docs/manual/recording.md)、[音标手册](docs/manual/ipa-plus.md)。无需服务器处理音频或输入文本。Windows 开发态功能/实际Qt/Chrome 已分项验证，实体声卡/EGG与完整跨平台验收仍待；追加R2 EXE交付见上方。

**2026-10-02 唇形提取 M05-R1：** 采集状态移至左栏，画面下均匀显示四曲线；离线完整帧回放、连续录制、MP4 + WAV 保存及 V2/下游参数读取已修复。见 [验证报告](docs/testing/2026-10-02-m05-r1-report.md) 与 [操作说明](docs/manual/lip-extraction.md)。从 [Start-M05-Workbench.ps1](scripts/Start-M05-Workbench.ps1) 使用最新源码，旧 EXE 未更新；物理设备同步和 Linux 媒体链仍待验证。

**2026-10-02 M06-R2：** AV/AH 标尺与源开关已修复，五类预设保留 F0 曲线并支持整体平移/编辑/回切，详见[修复报告](docs/testing/2026-10-02-m06-r2-report.md)。新配置 m06/2，旧参数标尺明确拒绝；源码入口 Start-M06-Workbench，现有 EXE 未重打。

**2026-10-02 语音合成 M06-R1：** 图窗同页、时间轴/元音边界对齐、参考共振峰与即时应用时长已修复，见 [验证报告](docs/testing/2026-10-02-m06-r1-report.md)。[AV 专项核查](docs/references/m06-av-audit-2026-10-02.md)明确历史标尺与标准 Klatt 的差异。开发入口 `scripts/Start-M06-Workbench.ps1`，EXE 未更新。

2026-10-01 M03-R2：井井要求实时交互、四图手势/布局、总览显示和IF图窗，并追加暂时移除声门活动、高低通数值输入及默认低通2000 Hz。已完成限定Windows真实EGG录音、Chrome/实际Qt验证，见 [报告](docs/testing/2026-10-01-m03-r2-report.md) 与 [ADR](docs/decisions/ADR-M03-R2.md)。交互改为有界内存会话，正式导出保留任务；12组数值/字节对照、7项会话/HTTP、28项架构、15项前端状态、14组Chrome及Qt PNG保存通过。连续更新约0.13–0.22秒，首次仍需加载，未承诺固定延迟。Linux实时准入未开放，旧EXE未更新，未DDL/push/改V2或原音频。


**2026-10-01 M01/M02 试用修复：** 两页列表内部滚动，批次轮询不再造成按钮抖动，支持可选递归目录；修复TextGrid空白尾段导致的切分失败，新增图窗清空/删除，当前图默认PNG，桌面长WAV采用有界完整时长预览。真实48文件桌面切分、274片段采样核对及实际参数图操作通过。使用[当前源码入口](scripts/Start-M01-M02-Workbench.ps1)，[验证范围与限制](docs/testing/2026-10-01-m01-m02-repairs-report.md)。下方已交付EXE未包含本轮后续修复，真实长录音仍未验。

**2026-10-01 Windows EXE 已更新：** [本机试用 EXE](dist/PhoneticToolbox-v3-LocalPreview-20261001/PhoneticToolbox-v3-LocalPreview-20261001.exe) 包含当前代码与蓝线定位修复。成品 10 项任务、15 页和退出清理通过，仍依赖本机独立科学环境；先关闭旧版再运行。原试用数据和旧包保留，详见[交付记录](docs/testing/2026-10-01-exe-update-report.md)。后续根据井井人工试用反馈逐项修复。

**2026-10-01 P16 审查修复：** 已修复声学窗口累计偏移、共振峰数量失效、异常误报成功、REAPER 策略、MFA 桥接大小、导出字体、能力宣告、目录保存及 Command 快捷键，并修复侧栏边界遮挡波形。新声学结果标记 acoustic/2，旧结果仍可读。Windows 定向测试与前端构建完成，Linux 新 POSIX 保存/macOS 原生与完整发行待验，旧 EXE 未更新。见[逐项报告](docs/testing/2026-10-01-review-repairs-report.md)和[桌面运行时清单](docs/specs/desktop-bundle.md)。下方带日期条目为各阶段历史证据；当前网页政策代码为 1 GB/3 天，本轮未迁移现存数据库。

**2026-09-29 EGG 修复：** 加载和参数变化自动绘图，修复运行中换选区后的空白、历史恢复与取消竞争，并去除管道及临时写入瓶颈。实际 77 秒录音已通过 Chrome/Qt，数值输出逐字节不变。使用 [M03 源码入口](scripts/Start-M03-Workbench.ps1)，旧 EXE 未更新，具体性能与边界见 [修复及审查报告](docs/testing/2026-09-29-m03-realtime-report.md)。

**2026-09-29 P04-RESIZE / M02-DEFAULT：** 全部现有模块侧栏和导航栏统一边界拖动、按模块/账号记忆，M01 默认拓宽，M02 首次空图且移除宽度滑块。剩余重复模块页首/关闭入口已移除，设置与使用说明进入工作台标签。Windows Chrome/实际 Qt 开发态限定验证及 WSL 静态核对见[报告](docs/testing/p04-resize-report.md)，[操作说明](docs/manual/settings.md)。旧 EXE 未更新；Linux GUI、触屏与生产服务器未验。

2026-09-27 M05：基线、浏览器候选、正式离线任务/文件和 Windows Chrome/实际 Qt 本机三模式录制已有分项证据。已按 V2 修复完整网格、同帧叠加、停止尺寸及 Qt 权限错误提示。浏览器新模型未通过等价门，正式结果保留 legacy；完整模块 in_progress，物理同步、Linux/远程准入及剩余设备矩阵待验。入口 `scripts/Start-M05-Workbench.ps1`，见 [M05 报告](docs/testing/m05-report.md) 与 [说明](docs/manual/lip-extraction.md)。未 push、DDL、部署或生成 EXE。

M11 本轮开发入口：[MFA 使用与组件安装](docs/manual/mfa.md)。Windows 本地正式任务和离线候选已限定验证，完整迁移仍 in_progress；[报告与各平台边界](docs/testing/m11-report.md)、[远程接线要求](docs/specs/m11-remote-handoff.md)。主 EXE 未更新。

**2026-09-27 M15 感知实验：** 已接入统一工作台 → 标注与实验 → 感知实验。纯客户端，无需登录，刺激/问卷/结果只保存在本机。四范式、配置/XLSX、恢复与三格式结果导出已完成限定 Windows Chrome/实际 Qt 验收。离线 A 与 Qt C 定向通过，B 保留已批准方案；Linux 浏览器与物理时延未验。[操作说明](docs/manual/perception.md) · [验收报告](docs/testing/m15-report.md) · [统筹摘要](docs/testing/m15-coordination-summary.md)。没有 M15 服务器 API/数据库前置依赖，未打包 EXE。

**2026-09-26 当前统筹入口：** [Linux/小型服务器/统一 UI 任务计划](docs/plans/2026-09-26-server-coordination.md) · [可信外接计算节点设计](docs/specs/remote-compute.md) · [只读代码审查](docs/testing/2026-09-26-planning-audit.md)。新需求为阿里云 Ubuntu 24.04.2/x86_64、2 vCPU、套餐 4 GiB/50 GiB（系统可见内存约 3.4 GiB）、每人 **1 GB/3 天**、每模块追加 Linux 验证与资源测量、取消模块内重复页首/关闭按钮。当前只更新规划，代码/数据库仍用旧政策，Linux/远程节点/UI 改造均待井井指派 agent。已有模块的 Windows 验收范围不变。

**2026-09-19 M12-R6：** 图窗置顶、资源与搜索下移、整段标注删除/剪切/粘贴完成。119 项前端、6 组新增及 8 组 R5 Chrome、类型/构建通过；实际冻结 EXE 11 步保存下载链路通过。临时入口 `dist/m12-preview-r6/PhoneticToolbox-v3-M12-R6.exe`，旧包保留。[R6 报告](docs/testing/m12-r6-report.md)。深色沿用全局主题，旧灰色截图为过渡中间态。

**2026-09-19 M12-R5：** 图窗编辑、选区同步、毫秒语谱窗及原始 TextGrid 优先已完成限定开发态验证，见 [R5 报告](docs/testing/m12-r5-report.md)。前端已构建，旧 EXE 未更新。

2026-09-15：按用户要求完成 M12-R4 长 WAV 轻量读取和标注层下方秒数刻度，已生成 `dist/m12-preview-r4/PhoneticToolbox-v3-M12-R4.exe`。本轮明确不做验证，故仅记录实现与构建完成，未验证运行行为。见[构建记录](docs/plans/2026-09-15-m12-r4-long-audio.md)。

**M12 语音标注对齐（2026-09-14）：** 已完成限定 Windows 功能迁移及 R3 试用修复。手工边界取消毫秒级间距限制，清空/合并使用键盘；已加载 TextGrid 支持空白处双击新增，自动保存改为 `_自动保存.TextGrid`。波形/语谱图等高，波形细节按原采样点连线，三图支持 Shift 平移，语谱图支持左键拖选；首次音素切分可选点击位置或等分。保留 R1/R2 的实际层名、顺序标注、整体移动及共享选区试听。整个 v3 在设置中调整页面大小，禁止 Ctrl＋滚轮整页缩放。临时单文件入口 `dist/m12-preview-r3/PhoneticToolbox-v3-M12-R3.exe`，实际冻结 EXE 50 步通过，旧包保留。[操作说明](docs/manual/annotation.md) · [R3 验证](docs/testing/m12-r3-report.md) · [迁移验收](docs/testing/m12-report.md) · [R1 范围](docs/testing/m12-r1-report.md) · [R2 验证](docs/testing/m12-r2-report.md)。M04 和其他模块保持原迁移检查点，临时包不包含 M03/M04 独立兼容运行环境。

**M04 LPC谱图（2026-09-13）：** D页面已限定Windows开发态verified，接入统一波形/标注/频谱/试听/任务、草稿与保存下载。24组真实Chrome、69项前端、13项Python通过，完整M04仍in_progress，下一项托管网页/自然录音及20项验收收口。见[页面报告](docs/testing/m04-ui-report.md)、[操作说明](docs/manual/lpc-spectrum.md)与[实施步骤](docs/plans/2026-09-13-m04-implementation.md)。

**EGG开发态功能收口（2026-09-13）：** 六功能组与最后草稿/连续操作检查完成，开发态功能阶段verified。补齐LP阶数保存、重开及失败反馈，65项前端与23组Chrome通过。入口为`scripts/Start-Research-Workbench.ps1`，范围及限制见[收口报告](docs/testing/m03-dev-closeout-report.md)。完整模块的设备/生产/跨平台验收仍单列。

**历史范围（2026-09-13）：** 当时只推进开发版，EXE封装、打包和相关探针暂停。2026-09-14 井井已明确要求 M12 临时包并继续反馈修复，当前以页首 M12-R3 为准；其他 EXE 专项仍暂停。

**当前优先级（2026-09-13）：** EGG代码引用后续核查按井井要求暂停；LPC学术引用已补，代码出处有界查找不阻挡功能。已修复切换文件后旧预览错误串入新文件的问题，[Chrome五项回归](docs/testing/m03-preview-switch-report.md)通过。

**EGG来源复核（2026-09-13）：** 已确认直接迁移来源为V2，8份原文件哈希一致。使用说明补充简化逆滤波固定取GCI后3ms、未按GOI确认闭相的实际行为；方法文献对应与许可缺口单列，见[复核记录](docs/testing/m03-provenance-report.md)。

**EGG功能复核（2026-09-13）：** 六功能组已重新映射，补齐总览单击保留选区定位、提交期间取消。[开发态复核报告](docs/testing/m03-function-review.md)记录全部功能及差异，完整M03仍in_progress，方法/许可与设备证据单列。

**EGG结果反馈（2026-09-13）：** 结果读取失败在当前窗口提示，可重新读取或返回分析；迟到的读取、保存和下载反馈不影响新窗口。[开发态Chrome限定验收](docs/testing/m03-result-feedback-report.md)通过。

**EGG批次反馈（2026-09-13）：** 参数错误在弹窗内提示并保留选择；全部提交被拒绝时保留上一批保存入口，部分成功时列出未提交原因。[Chrome限定验收](docs/testing/m03-batch-feedback-report.md)通过。

**EGG试听修复（2026-09-13）：** IF原音频/估计按文件角色绑定播放，修正清单排序导致的标签对调；两条试听可直接切换，旧设备请求失败不会打断新播放。[限定开发态验收](docs/testing/m03-playback-report.md)通过，[EXE候选范围](docs/plans/2026-09-13-m03-f-candidate-review.md)仍planned。

**EGG导出字体（2026-09-12）：** 单文件三图与批次带图提交前检查实际计算环境字体，缺失时保留分析/批次选择并提示；纯CSV与逆滤波不受字体缺失阻断。[限定Windows开发态验收](docs/testing/m03-font-preflight-report.md)已通过。

**EGG长录音（2026-09-12）：** 当前开发版支持最长120秒且576万帧，保留全文件处理与末尾定位、整段导出。[限定Windows验收](docs/testing/m03-long-report.md)通过，剩余交互/来源收口及冻结EXE仍待。

**公共滚动修复（2026-09-12）：** 内容超出窗口可滚动，EGG/参数图改用 Ctrl＋滚轮缩放，弹窗底部操作保持可见；桌面最小尺寸随屏幕可用区域限制。[开发态验收](docs/testing/p04-scroll-report.md)已通过，历史 EXE 未重新打包。

**全局字体（2026-09-12）：** 设置中可统一调整中文、英文与数字、等宽字体，IPA始终固定Doulos SIL。M01/M02/M09/M10开发页面与新导出图像已接入，M03后台任务使用提交时的字体快照。整幅PNG刻度、音标和图例统一基础字号。操作见[字体设置](docs/manual/settings.md)，证据与限制见[字体报告](docs/testing/p04-fonts-report.md)。历史EXE未重新打包。

**M03最新进度（2026-09-12）：** D已接入统一EGG页面与分组参数区，限定Windows开发态Qt/独立Chrome的四图、单文件/IF/批次保存与重开操作已验证，见[页面报告](docs/testing/m03-ui-report.md)及[使用说明](docs/manual/egg-analysis.md)。开发入口`scripts/Start-Research-Workbench.ps1`可打开新页。完整M03仍in_progress，下一项E联合收口；旧EXE未重新打包，不能视为包含新页面。以下为前序历史记录。

**2026-09-12 收尾更新：** 既有成果已建立本地源码检查点，M02新增包含波形/标注/可选语谱图的白底300dpi整幅PNG，限定开发态Windows Qt/Chrome验收见[收尾报告](docs/testing/m02-png-closeout-report.md)。使用`scripts/Start-Research-Workbench.ps1`可读取本轮前端构建；下面历史Research-Fix1.exe未重新打包。M03-A/B已完成独立基准及[Windows纯核心验证](docs/testing/m03-core-report.md)：80项检查、11样例31761项精确比较。完整M03仍in_progress，任务/导出/页面未接入；后续布局按[v2位置与v3组件约束](docs/design/m03-v2-layout.md)执行。P08继续in_progress。

**2026-09-12：M01/M02/M09 Windows 联合修复已通过限定单文件验收。** 本机修复入口：`dist/research-repair/PhoneticToolbox-v3-Research-Fix1.exe`。双击自动持有本地任务服务，首次新建专用任务库，不需要登录或手动启服务器。已修复冻结子进程误开窗口、参数/语谱图读取、TextGrid 时间比例与截图隐藏，并纳入深色下拉修正。具体证据和未测范围见[修复报告](docs/testing/desktop-repair-report.md)。原 M10-R5 EXE 保留，以下为历史交付记录。仍未正式发行。

**2026-09-11：M01完成39项收口，M02同图叠加与批量多图窗与M09语谱图重建已完成限定Windows双端迁移。本任务未改动M10录制EXE。按井井要求停止在这几个任务，当前尚未正式发行。**

当前 [联合报告](docs/testing/m02-m09-report.md)；[参数显示说明](docs/manual/parameter-display.md)；[语谱图重建说明](docs/manual/spectrogram-to-audio.md)。本次开发入口：[Start-Research-Workbench.ps1](scripts/Start-Research-Workbench.ps1)，使用独立 `.venv/m09-ui`。原M10录制EXE保持不变。以下保留各阶段历史证据。

P05 当前进展：[账号与项目报告](docs/testing/p05-accounts-report.md)、[迁移审阅](docs/testing/p05-migration-review.md)。P05 已获专属空库授权并通过真实 PostgreSQL 定向验收；P06 已完成 Windows 持久任务流程验收及闪退后复验，见 [任务验收与限制](docs/testing/p06-jobs-report.md)及 [建表审阅](docs/testing/p06-migration-review.md)。P07 已获 003 存储表与专属测试文件清理授权，Windows 单文件的真实数据库/磁盘、并发额度、TCP 到期和独立浏览器定向验收已通过；004 已获“好，允许”的具体授权并执行，P07 受控任务文件与有界 ZIP 的 Windows 联合验收也通过；科学算法、原生目录输出和生产/跨平台能力仍未验收，见 [P07 单文件报告](docs/testing/p07-storage-report.md)、[联合验收与限制](docs/testing/p07-job-files-report.md)和[具体操作审阅](docs/testing/p07-migration-review.md)。两次 Codex 退出与内置测试页关闭存在直接时间关联，排查及绕行约定见 [恢复记录](docs/testing/p06-recovery-and-codex-exit.md)。

P08 已完成的 M01 阶段：[M01 参数估计实施计划](docs/plans/2026-09-09-m01-implementation.md)与[源码审阅报告](docs/testing/m01-planning-report.md)已形成，80参数/14设置和39项验收已映射；[M01-A基准](docs/testing/m01-baseline-report.md)现已完成28例双轮捕获和23项测试，[M01-B科学核心](docs/testing/m01-core-report.md)现已通过独立wheel的149项Windows定向测试，[M01-C适配](docs/testing/m01-io-report.md)已通过222项Windows wheel测试及实际双产物回读，[M01-D契约](docs/testing/m01-contract-report.md)已通过限定Windows协议与真实结果往返验收，[M01-F2](docs/testing/m01-persistent-report.md)现已接入持久计算、取消/重试、结果保存与TextGrid同步切分，并通过列明的Windows真实双端验证；[M01-G联合审阅](docs/testing/m01-report.md)已完成本轮Windows真实上传/双格式回读、自然录音对照、错误与响应式验证；[操作说明](docs/manual/parameter-estimation.md)已更新。[旧格式入口](docs/testing/m01-legacy-report.md)已补齐本机PKL图形转换及显式关联历史XLSX/SQLite同步切分。M01/G已在[最终审阅](docs/testing/m01-final-review.md)关闭39项验收门；M02/M09当前范围见上方联合报告。

公共界面入口：[P04 工作台试用报告](docs/testing/p04-workbench-report.md)（公共界面已审阅，学术优先分组已修订；P04 限定范围 verified，P05 账号/项目范围 verified）。试用启动方法见 [开发说明](docs/development.md)。

前阶段：[P03 基线报告](docs/testing/p03-baseline-report.md)、[捕获协议](docs/baseline/capture-protocol.md)。P01 单文件探针与试用见 [原型报告](docs/testing/p01-host-probe-report.md)；P02 环境、包与契约见 [开发说明](docs/development.md) 和 [P02 报告](docs/testing/p02-scaffold-report.md)。P03 冻结旧服务行为用于后续回归，不表示已验证 v3 算法或所有旧指标的科学准确度。目前M01已完成限定Windows交付，其余模块按各自状态记录。

沿用 U2 紧凑工作台、浅深色主题和 K2 波形团子。一个自有仓库共用前端与科学核心，分别交付网页版、Windows 单文件直用版/安装版、macOS；Linux 桌面作为独立平台验收项保留。已有可用代码优先迁移，不重复实现同一算法。

| 建议阅读顺序 | 文档 |
| --- | --- |
| 1. 审阅入口 | [规划摘要](docs/REVIEW.md) |
| 2. 全部阶段与任务 | [详细总计划](docs/plans/2026-09-09-v3-master-plan.md) |
| 3. 总体边界 | [架构文档](ARCHITECTURE.md) |
| 4. 外观与操作 | [UI 规范](docs/design/UI_SPEC.md) |
| 5. 全部功能如何迁移 | [15 模块迁移规格](docs/modules/module-migration.md) |
| 6. 登录、1 GB、3 天目标与迁移 | [账号与存储规格](docs/specs/accounts-storage-jobs.md) |
| 7. 来源与论文 | [查验报告](docs/references/source-audit.md)、[引用与第三方清单](third_party/README.md) |
| 8. 测试与发布 | [验证策略](docs/testing/verification-plan.md)、[平台与发行](docs/deployment/platform-release.md) |
| 9. 开发约束 | [AGENTS.md](AGENTS.md)、[架构决策](docs/decisions/ADR.md) |

## 已完成和未完成
- 建立同一仓库的 codex/v3-rebuild 工作分支与独立本地 worktree。
- 从当前 v2 工作状态继承 427 个源码/资源/文档文件，原目录文件、HEAD、暂存区保持原样。
- 本地源码基线为 ccf4ff73c355e8d955c2a6b9605b32ac2c7255a6；这是迁移依据，不表示该源码已通过完整功能验收。
- D0.1 的功能与视觉资料已在本工作区归档；其架构部分以本轮文档为准。
- 新目录中的 AGENTS.md / ARCHITECTURE.md 定义将来的实现边界；并没有伪造空壳业务实现。
- 继承的 phonetic_toolbox、run.py、run.spec、pyproject.toml 仍是 v2 过渡代码。运行它们不会得到统一前端的 v3。完成迁移前不得对外称已经实现 v3。
- P01 已在项目内建立隔离运行时与探针依赖，并构建单文件技术原型；未创建数据库、部署服务器、公开发布或删除旧网页目录。
- 学术/代码来源已按证据登记；没有明确许可证的材料仍须在发行前解决。

2026-09-11：[M10 声道工作台](docs/testing/m10-report.md)及追加的 [R4 录制增强](docs/testing/m10-recording-features-report.md) **已迁移 / verified（限定 Windows 本机录制），暂时冻结**。历史 R4 产物见验收报告，操作见[声道说明](docs/manual/vocal-tract.md)。新增本地关键帧/构形库、0.05 秒短帧与静音帧、150 Hz 默认、重播缓存、当前或六视图同步视频，并修订闪烁和构形边界。本任务只更新 M10，其他模块进展见各自计划。网页服务器/macOS/Linux 原生仍 planned，不代表完整 v3 发行。

## 工作区约定
frontend / backend / desktop / packages / contracts / resources / tests 是 v3 的目标边界。现有 phonetic_toolbox 是受控的迁移来源，两者不能无限期混用。详细路径、迁移顺序和每阶段退出条件见总计划。

用户数据、原始测试语料和本机绝对路径记录在忽略的本地证据文件中；不随公开源码或安装包分发。官方论文与手册使用来源链接；本地继承的历史 papers 目录不自动进入 v3 包。

所有对外部署、代码推送、发行物发布，待明确授权再执行。

M01-E 已完成[目录、共享研究页与Praat显示定向验收](docs/testing/m01-workspace-report.md)：默认单声道/可双声道、较高波形、紧凑文件行与全选、可开关语谱图、长音频峰值显示。按[开发入口](docs/development.md)启动；参数批计算、取消与持久发布已由[M01-F2](docs/testing/m01-persistent-report.md)接通。

2026-09-10 截图反馈后的 [M01 布局与时间交互修订](docs/testing/m01-layout-report.md)：独立列滚动、自适应目录条、可见时间轴、Ctrl+滚轮缩放和播放进度定位。[说明书覆盖门](docs/modules/v2-manual-coverage.md)已加入迁移流程；F2已接通真实TextGrid切分及可选参数同步保存；G最终逐项审阅现已完成，见[收口报告](docs/testing/m01-final-review.md)。

2026-09-10 [M01-F1切分与批次准备](docs/testing/m01-execution-preparation-report.md)已通过244项Windows定向测试及WAV/参数双格式真实合成产物回读。随后井井对[005具体审阅](docs/testing/m01-migration-review.md)授权继续，两库已应用。[F2报告](docs/testing/m01-persistent-report.md)记录实际页面保存/批处理、异常恢复及边界；开发启动使用[scripts/Start-M01-Workbench.ps1](scripts/Start-M01-Workbench.ps1)。

2026-09-11：井井追加起声渐入、静音后渐入、关键帧拖动排序及一键清空，要求修改后直接打包、不做检验。[R5 实现记录](docs/plans/2026-09-11-m10-onset.md)对应 [Windows 录制版 5](dist/m10-recording/PhoneticToolbox-v3-M10-R5.exe)。R4 的 verified 仅属于历史验收，不代表此次 R5 修改已验收。

2026-09-12：[M03-E1 操作与导出对齐](docs/testing/m03-report.md)已限定验证，恢复独立批次设置和源文件/时间保存名，修正单文件 CSV 双 F0 不受显示开关影响，补四图手势与保存反馈。完整 E/M03 仍在联合收口，旧 EXE 未更新。

2026-09-12：[M03-E2 网页与自然录音验收](docs/testing/m03-e2-report.md)已限定验证，修复切换文件时旧结果迟到覆盖的问题。网页双账号、服务器三路径下载、受控配额/到期清理与两份自然录音页面通过。完整 E/M03 尚待范围差异与来源收口，EXE 未更新。

2026-09-12：[M03-E3 微观范围与方法审阅](docs/testing/m03-e3-report.md)恢复5–5000 ms和原版显示抽点，补官方书目与方法差异。完整E3/M03仍在进行，长录音全段计算、字体预检和EXE尚待收口。
