# PhoneticToolbox v3 — Agent 工作规则

## 0. 当前阶段与授权
- 2026-09-12 井井追加EGG总览紧凑布局：波形置顶，双声道/缩放/适合窗口在图下同排。限定Chrome/Qt布局已通过，见docs/testing/m03-overview-report.md。仅EGG启用公共组件compactOverview，其他模块默认布局不变。120秒长文件成果已在9cbbf3c，完整M03仍in_progress，下一项字体预检/剩余收口；旧EXE未打包。
- 2026-09-12 井井在长文件下一步说明后授权继续。M03-E3-B 已限定 Windows 开发态验证120秒/576万帧完整处理、首尾查看与三路径导出，原V2双轮20数组23242942值精确一致，见 docs/testing/m03-long-report.md。按实测设3GB/240秒进程预算，60秒仅总览视窗；核心/V2/环境/旧EXE不变，未DDL/push。完整E3/M03仍in_progress，下一项字体预检/剩余交互及来源收口，再审阅冻结EXE范围。
- 2026-09-12 井井审阅滚动方案后授权继续。P04-SCROLL 现为 verified（限定 Windows 开发态公共内容/二维滚轮/Qt 尺寸），见 docs/testing/p04-scroll-report.md。普通滚轮滚动，EGG/M02 Ctrl＋滚轮缩放，弹窗正文独立滚动；M10 仅外层最小高度/滚动，模型手势与录制未改。旧 EXE 未重打包，完整 M03/E3 仍 in_progress，后续回到 E3-B。未 DDL、push 或改 v2。
- 2026-09-12 井井在E2后授权继续E3。本轮恢复微观5–5000ms、原版显示抽点与边界不重复提交，限定Windows开发态验证见docs/testing/m03-e3-report.md。新增官方书目/原手册截图/Henrich原文方法差异，代码许可未闭合。完整E3/E/M03仍in_progress，下一项E3-B长文件全段语义和预算、字体预检/剩余交互；66.4秒仅导航及拒绝已验，不代表全段分析完成。仅重装项目m03-compatible核心wheel，第三方库/v2/语料/旧EXE不变，未DDL或push。
- 2026-09-12 井井在E1后回复“继续”。M03-E2现为verified（限定Windows开发态Qt/本机托管PG与Chrome），见docs/testing/m03-e2-report.md。双账号读隔离/切换、服务器三路径10文件回读、受控配额与到期清理、两自然录音开头及较响ROI通过；修复历史预览回读迟到覆盖新文件选择。完整E/M03仍in_progress，下一项E3长文件/微观范围差异与来源/剩余边界；F冻结EXE未做。到期为测试调整时间并显式清理，不冒充七天自然经过。未DDL、未改v2/语料/环境/旧EXE、未push。
- 2026-09-12 井井在D后回复“继续”。M03-E1现为verified（限定Windows开发态默认/导出/手势对齐），见docs/testing/m03-report.md：独立单文件/批次默认，单文件CSV双F0不受显示开关影响，源文件名/时间保存及同名保护，四图滚轮/拖动/键盘/中心线与弹窗保存反馈。完整E/M03仍in_progress；E2下一项网页/自然录音页面及切换竞争。原V2微观滚轮5–5000ms与当前10–200ms差异已明确，长文件预算/来源/EXE仍待。未DDL、未改v2/旧EXE/环境、未push。
- 2026-09-12 井井继续授权M03-D，并反馈下方按钮散乱。D现为verified（限定Windows开发态Qt/独立Chrome四图、参数/试听、任务与逐文件批次保存），见docs/testing/m03-ui-report.md。首版遵守v3主题、公共字体与组件，四图贴合v2；控件改为两行分组，高低通集中EGG图上方。122项科学/导出/字体、14项M03契约、47项前端及真实UI通过。广域回归另保留M01 Scratch取消清理的一次WinError32，单项复跑通过但根因未确认；不称全套稳定全绿。完整M03仍in_progress，下一项E的30项/来源/自然语料页面/网页联合收口，F冻结EXE尚未实施。未DDL、未改v2/m09科学环境/旧EXE、未push。
- 2026-09-12 井井在M03-B后回复“好，继续”。M03-C现为verified（限定Windows开发态任务/文件/数值导出），见docs/testing/m03-jobs-report.md。m03/1接既有任务/资产协议，独立MKL子进程，单文件CSV+三PNG、逐文件批次导出及IF双WAV实际回读；取消/失效worker/故障回收/重开/两份授权自然录音通过。未执行DDL或修改m09/m10/v2环境。后续P04-FONT已接入字体快照并完成限定Windows图片专项，见docs/testing/p04-fonts-report.md；完整M03仍in_progress，下一项M03-D。EGG页面/整批目录交互/冻结EXE尚未实施，不能把本轮PNG导出图当V3交互界面。
- 2026-09-12 井井在M03-A和UI统一约束后回复“噢噢 那你继续吧”。M03-B现为verified（限定Windows纯核心及安装wheel），见docs/testing/m03-core-report.md：80项检查、11样例31761项精确比较、输入与原v2源码未变。新核心位于packages/phonetic_core/src/phonetic_core/egg，实际用户默认slope/scale，保留两种ROI旧规则与独立mask，真实Praat帧时间与N/fs元数据明确区分。SciPy同版不同构建产生差异，兼容环境为项目内.venv/m03-compatible（Conda/MKL锁）；.venv/m03-ui为未通过逐位基准的PyPI候选，不能误用。完整M03仍in_progress，下一项C任务/文件/导出。页面/Qt整合/EXE尚未实施，不重复DDL、不改m09/m10/v2环境，UI从第一版遵守既定v3风格。
- 2026-09-12 井井补充：EGG贴合v2仅指布局与功能，v3统一设计规范和公共组件必须从页面第一版落实。页面内控件/图表/状态也须统一，不能只换外壳或留到收尾调整。真实组件与适配边界见docs/design/m03-v2-layout.md；旧Qt截图仅是基准，展示时先标明，不能让用户误认为v3页面设计。
- 2026-09-12 井井在M03计划后回复“好，请继续”，明确EGG布局尽量贴合v2、功能囊括v2，其余可优化。M03-A已完成限定Windows原v2独立基准：11样例双轮一致，公开合成数值与私有语料分开，详见docs/testing/m03-baseline-report.md。完整M03仍in_progress，下一项M03-B纯数组核心；页面/EXE尚未实现。布局以docs/design/m03-v2-layout.md为准：左CQ/SQ与语谱、右音频与EGG微观、下方两行参数及总览，覆盖早期右侧集中设置方案。实际单文件与批处理均GCI slope/GOI scale，底层EGGConfig被GUI覆盖的事实已纠正。不得将基准verified扩大为模块verified，不重复DDL、不推进M04。

**2026-09-12 较早停止点（仅历史，当前状态以上方 M03-E3 为准）：** 井井在进度审阅后授权继续整理成果、补齐M02整幅PNG并细化M03计划。现有成果已形成本地检查点c800ce8。M02-F05整幅PNG已通过限定开发态Windows Qt/Chrome验收，见docs/testing/m02-png-closeout-report.md；旧Research-Fix1/M10-R5 EXE未重新打包。M03源码/说明书审阅及30项验收设计已完成，仍planned，下一步审阅docs/plans/2026-09-12-m03-implementation.md后进入M03-A。P08仍in_progress，P09整体planned，M10/R4仅历史限定范围verified，R5仍未检验。旧R4 EXE当前不在发行目录，保留原报告元数据。本轮未push、发布、执行DDL或修改v2。

### 历史授权与阶段证据（当时的下一步不作为当前执行指令）
- 2026-09-12，井井授权修复已迁移 M01/M02/M09 的实际桌面入口，并明确允许新建 v3 专用本地任务库。Research-Fix1 通过限定 Windows 合成数据与真实单文件双轮验收，见 docs/testing/desktop-repair-report.md。首次新库位于 LocalAppData/PhoneticToolbox/v3/research-v1，只初始化不存在的新目录并复用既有002/005；已有目录只校验，不运行DDL。修复产物为 dist/research-repair/PhoneticToolbox-v3-Research-Fix1.exe，原M10-R5及v2保留。已修冻结worker调度、截图隐藏、TextGrid时间轨并纳入主题修复，其他模块继续停止，网页部署/安装版/跨平台不在此轮。
- 2026-09-11 最新明确授权与停止点：井井要求继续完成M01收口，再迁移M02参数显示（默认同绘图区叠加曲线、可多图窗批量分配，按说明书2.2纠正并复验）和M09语谱图转音频，做完停止。本轮已完成M01最终39项审阅，以及M02/M09限定Windows本地/托管Chrome验收，见docs/testing/m01-final-review.md、docs/testing/m02-m09-report.md与docs/plans/2026-09-11-m01-m02-m09.md。当前三项标为verified的范围以报告为准，M09多屏/DPI截图设备未纳入。新项目内m09-ui与开发入口scripts/Start-Research-Workbench.ps1不修改M10运行环境/录制EXE；M10/R4保持冻结。下列“下一项M01-G/39项暂停/M02不启动”是历史检查点，当前停止点覆盖其执行顺序。不得自行推进其他模块、重复DDL或发布。
- 用户：井井；助手自称秋叶；默认中文。技术判断说明证据、限制和待验证项。
- 当前 **P04 统一工作台 verified（限定公共界面与 Windows 定向验证）**；2026-09-09 井井在 P03 后确认继续，授权范围见 docs/plans/2026-09-09-p04-workbench.md 与 docs/testing/p04-workbench-report.md。井井已反馈当前界面无明显问题；学术优先的分组致谢修订已落实。P04 提供共同界面、真实 WAV 预览和 Qt 演示，不代表 15 模块算法已迁移。井井已授权继续 P05，并对专属空库方案回复“允许”。P05 现为 verified（Windows 账号/会话/项目范围）：实际 PostgreSQL 建表、隔离、事务、并发与受控重启恢复已通过；井井随后回复“好，继续”，已授权 P06 实施；P06 现为 verified（限定 Windows 持久任务流程）：具体建表审阅后的继续指令已核实，实际 PG/SQLite、并发、取消/中断/重试、本机服务重启与网页账号隔离均通过；退出后的收尾复验与记录已完成（docs/plans/2026-09-09-p06-jobs.md、docs/testing/p06-jobs-report.md）；井井在 P06 验收后回复“好，请继续～”，已授权推进 P07；P07 现为 verified（限定 Windows 受控存储、工程生成与有界 ZIP 联合范围）；003 审阅后“好，继续”、004 审阅后“好，允许”的具体授权均已执行，旧行保留；真实 PG/磁盘、配额/TCP/到期、任务输入/结果/旧 worker fencing、进程中断重试与独立浏览器两标签页联合验收通过。证据见 docs/testing/p07-storage-report.md、docs/testing/p07-job-files-report.md，设计与边界见 docs/plans/2026-09-09-p07-job-files.md；未涵盖科学算法迁移、硬断电、生产负载、原生工具任意目录输出或跨平台发行。P08 已完成M01-A基准与M01-B限定Windows科学核心验收，M01-C已完成限定Windows受控原生/格式适配，M01-D已完成限定共享协议与结果语义验收，E目录、草稿、试听与Praat显示已限定验收；F2持久批次及双端真实操作现已接入，范围见docs/testing/m01-persistent-report.md；完整M01/P08仍in_progress；G当前联合验收见docs/testing/m01-report.md，旧PKL图形转换和历史参数导入仍待补齐；P09仍planned，按具体模块计划推进；见 docs/plans/2026-09-09-p05-accounts.md 与 docs/testing/p05-accounts-report.md。
- 2026-09-09 井井在下一步说明后回复“好，继续”，本轮已完成 P08/M01 迁移前源码审阅、文件级计划与细项验收设计，见 docs/plans/2026-09-09-m01-implementation.md 和 docs/testing/m01-planning-report.md。该审阅交付时M01/P08仍planned，随后M01-A的实施与当前进度见下一条。原生输出、持久批次和实际DDL的具体设计门见计划，不把本轮文档完成扩大为整模块verified或新的数据库迁移授权。
- 井井随后回复“好，请继续”，已授权并完成M01-A独立基准补齐及科学环境审计：28例双轮捕获、23项测试，范围与新发现见 docs/testing/m01-baseline-report.md。M01-A结束时M01/P08为in_progress，随后进入M01-B。该轮只对合成数据新建导出SQLite文件，不操作现存/服务数据库；M01-A时科学包仅审计，B轮独立安装/锁定见下一条。
- 井井在 M01-A 后回复“继续”，授权并完成 M01-B：独立科学锁与可安装核心 wheel、149 项 Windows 定向测试，数值/时间/mask 对照及转换字节通过；真实采样率与后端结果元数据单列修正。见 docs/testing/m01-core-report.md。后续C/D/E/F2已完成限定验收，下一项M01-G；B 的小型合成 native 测试适配不得接网页/用户任务，完整 M01/P08 仍 in_progress。
- 前阶段已完成 **P03 科研行为基线的 Windows 定向验收**。2026-09-09 井井在 P02 后回复“好，请继续”，授权执行 P03，并确认 EGG 样例的双声道方向；边界见 docs/plans/2026-09-09-p03-baseline.md 和 docs/testing/p03-baseline-report.md。该结果不代表 v3 算法、完整 EXE GUI 或跨平台验收。公开发布、服务器部署、全局环境变更和全面业务迁移仍不属于本轮范围。
- 用户明确不要额外备份；保留相邻 v2 和现有使用数据。不得擅自删除、移动或修改 v2。这里的 Git 源码基线不是额外整目录备份，也不是已验证的新版本。
- 旧网页目录的删除授权有条件：仅在证实没有必要保留的依赖、独有工作或来源资料后才可删除。当前检查发现旧前端存在未提交修改；本阶段保留两个旧目录。
- 本文件适用于 v3 全目录。进入 frontend、backend、desktop、packages/phonetic_core、contracts、resources、tests、docs、third_party 时，继续读取相应 AGENTS.md 与 ARCHITECTURE.md。

- 井井在M01-B后回复“继续”，已完成M01-C：有界原生Job/命名管道、WAV/TextGrid/安全唇形格式与XLSX/SQLite双产物准备，222项Windows wheel测试及两组160×83实际导出回读通过，见 docs/testing/m01-io-report.md。未接入PG配额/持久发布或正式UI；M01-D已继续完成，完整M01/P08仍in_progress。

- 井井在M01-C后回复“好，请继续”，已完成M01-D：API 1.1/m01/1共享协议、可信输入/TTL边界、实际结果无损JSON与单文件/批次清单，318项Windows wheel测试及三组冻结对照通过，见 docs/testing/m01-contract-report.md。D阶段未开放科学任务HTTP；F2现已接通，见docs/testing/m01-persistent-report.md，下一项M01-G。

## 1. 必读与事实来源
1. README.md：阶段、入口、不能误用的历史目录。
2. ARCHITECTURE.md：组件边界、依赖方向、任务/文件/平台协议。
3. docs/requirements.md、docs/plans/2026-09-09-v3-master-plan.md。
4. 对应模块计划、docs/modules/module-migration.md、docs/decisions/ADR.md。
5. third_party/source-registry.json、third_party/README.md 和 docs/references/source-audit.md。
冲突时以井井最新明确要求为先；其余文档冲突必须记录并修正，不能挑方便的版本执行。D0.1/D0.2 仅提供视觉/功能历史，已更新部分以 D0.3 为准。

## 2. 每次编码的固定流程
- 先给出当前任务 ID（Pxx 或 Mxx）、本轮要改的文件、预期行为及验收命令。
- 工作前查看当前分支、git status、已有差异；不得混入无关用户改动。
- 先迁移已验证的纯算法与数据规则；原样迁移和行为修正分开提交。默认值、单位、帧网格、NaN、有声判定、输出范围不得借重构静默改变。
- 对真实行为风险先建立回归用例，再作最小实现；不写只照抄实现的测试，不为了文档微调启动全套构建。
- 修改后检查 diff、边界、中文/IPA、来源、实际结果；运行计划规定的定向测试。失败必须定位，不注释问题、不扩大容差、不跳过检查来“通过”。
- 一次完成一个可审阅任务；记录命令、平台、环境、产物、未测项目。只有满足退出条件才标完成。
- 需要改变架构/交互语义/算法/依赖/发布目标时，先补 ADR 和受影响计划，再按用户已授权范围实施；重大新范围需井井确认。日常明确任务不反复询问。
- 实施授权后允许按计划做本地小提交；push、发布、生产部署、改 CI/CD/密钥/全局环境和数据库实际迁移仍遵守会话授权边界。

## 3. 依赖与文件边界
- frontend 只调用 contracts 与平台能力接口，不读取服务器路径、不导入 Python 算法、不拼 shell。
- backend 是接口、身份、配额、任务和存储编排；不能实现另一套声学算法。
- desktop 是窗口/启动/本地文件/设备/发行适配；不能复制业务算法或请求公网来完成离线基础功能。
- phonetic_core 不依赖 Qt、FastAPI、数据库、HTTP、用户账号或固定开发机目录。
- 本地服务与服务器服务使用同版核心包与契约；不同平台差异集中在 adapter。
- 不从 ../PhoneticToolbox_v2、旧 Vue 站点或旧 API 工程动态导入。继承的 phonetic_toolbox 是过渡来源；迁移完成后新入口不能依赖它。
- 禁止为某页面添加新的独立 http.server；全部模块挂统一宿主、统一页面注册表、统一任务生命周期。
- 资源有清单和版本；不要把虚拟环境、研究语料、临时输出或第三方整仓库不加筛选地塞进发行包。

## 4. UI 与科研正确性
- 2026-09-12井井授权全局字体实施及已完成模块同步调整，全部IPA固定Doulos SIL，不提供替换选项。P04-FONT已完成限定Windows开发态M01/M02/M09/M10及M03后台字体专项，报告见docs/testing/p04-fonts-report.md，全平台专项仍in_progress。中文、英文与数字、代码/等宽文字可调，图表与导出默认跟随且可独立配置，同级图中文字统一字号，总标题最多1.2倍。详见docs/design/UI_SPEC.md第2.1节及docs/plans/2026-09-12-global-fonts-design.md。所有新页面从第一版接公共字体，已有模块按专项证据标记；M10本轮仅字体为追加授权，旧EXE及原算法/录制流程不扩大范围。字体变更不得改变科研值，后台导出绑定字体快照并验证实际产物。
- 采用已选 U2 紧凑工作台、完整浅/深主题、侧栏导航与标签页；暂定 K2 波形团子图标。
- 以 docs/design/UI_SPEC.md 为视觉标准；禁止各模块自己造一套颜色、弹窗、播放条、加载状态。
- 不允许删功能来迁就设计。15 模块、83 功能组、80 参数、14 设置逐项核对；新功能追加验收项。
- 2026-09-10 井井明确：布局和设计风格允许与 v2 不同，但 v2 功能不能遗漏。每个模块实施前同时读对应 v2 说明书章节与实际源码，建立“说明书操作 → 源码行为 → v3 入口 → 正常/异常验收证据”映射。说明书与源码不一致时单列差异，不能忽略说明书承诺，也不能照抄过时参数。章节入口与 M01 已发现差异见 docs/modules/v2-manual-coverage.md；按钮存在或预览可用不代表切分/导出/计算完成。
- 真实音频时间、选区、单位、标签和曲线必须一致；无数据画空态，不生成假结果。
- 主题与截图仅证明视觉，启动仅证明启动；不能据此宣称算法、全平台、摄像头或实验时序通过。
- 桌面不要求登录；网页账号/项目/配额界面不侵入本地研究流程。

## 5. 网页运行约束
- 目标 10 名研究者同时使用，有登录和用户隔离；计算并行度单独配置并实测。
- 每账号文件空间 5,000,000,000 字节（5 GB），上传、结果、缓存和临时占用均计入；共享安装模型不算用户额度。
- 用户数据最多保留 7 天；下载不要求自动删除、不延长有效期；用户可以直接删除，无需先下载。
- 原子预留额度、受控流式写入、失败回收、到期不可访问和实际物理删除都要实现。禁止仅前端判断额度或刷新页面重置配额。
- 每个任务有独立参数快照、owner、source/version；没有跨用户全局 Settings 单例。
- 长任务、设备和原生库进程由本应用明确持有；禁止按端口或泛化进程名杀掉其他服务。

## 6. 第三方与学术署名
- 凡引用外部代码、移植算法、参考论文、使用模型/字典/字体/数据，必须更新来源登记。
- 登记作者、题名/项目、URL/DOI、实际版本或 commit、使用位置、关系类别、修改说明、许可证、查验日期和未决项。
- 关系必须区分依赖、代码移植、论文方法、仅参考、数据素材与项目集成；登记表的 kind 使用对应细分类别，不能把方法参考写成原创实现或实际运行依赖。
- 上游今日 HEAD 不等于本项目最初使用版本；缺失证据标 unknown，不捏造 commit、作者或许可。
- 软件“关于 → 开源与学术致谢”、模块“方法与引用”、说明书与随包许可从同一登记生成；不得只在源码角落署名。
- PDF 优先链接作者/出版社/官方站点，标清论文/手册版本；不擅自镜像或打包原 PDF、研究数据。
- 声道说明必须包含 VTL 2.4 引擎、VTL 2.3 参考手册、几何资源各自来源及本项目适配边界。
- 未明确的再分发许可是对应发行物的验收阻断项；不把整套软件标为“完全原创”或不加区分地宣称全为 MIT。

## 7. 环境、测试与发行
- 继承 v2 基线的 Windows 打包仍仅使用 conda phonetic_311，并采用 python -m PyInstaller；P01 仅在隔离试验环境构建本地单文件探针，不打包或发布完整 v3。
- v3 开发/发行环境在 P02/P12 中单独创建并锁定，不能升级污染 v2 的 phonetic_311；具体环境名/Qt 宿主在原型验收后冻结。
- Windows 单文件直用版与安装版都保留；不能擅自以文件夹版替代已约定单文件目标。若单文件验证不通过，报告并修订 ADR。
- macOS、Linux 原生组件分别构建测试；WSL 的 Linux 服务测试不能冒充 Mac 或 Linux 桌面设备验收。
- 不改 .env、凭据、系统 CUDA/运行时、WSL 全局配置、CI/CD、生产数据库或对外发消息，除非该动作已在会话中明确授权。
- 操作 Windows 路径用绝对路径和 LiteralPath；递归删除/移动前验证最终路径位于明确目标内。文件编码 UTF-8 无 BOM。

## 8. Codex 测试稳定性暂行约定
- 2026-09-09 两次 Codex 退出都紧随最后一张内置浏览器测试页关闭。根因尚未确认，证据见 docs/testing/p06-recovery-and-codex-exit.md；暂不在本项目调用内置浏览器关闭/清理接口，也不为复现而重复该操作。
- 优先使用独立测试进程与已保存截图；需要新 UI 验证时用项目拥有的独立浏览器及独立用户配置目录，不操作用户正在使用的浏览器。服务清理由其父进程/EOF/停止信号负责，不依赖页面关闭成功。
- Git 操作明确限定 v3 根目录并核对索引；本地提交采用已审阅文件清单，不将用户目录、环境或测试数据库纳入。该措施是绕开触发路径，不宣称已修复 Codex。

## 9. 交付记录
每次交付写清：完成任务 ID、修改原因、真实验证命令和结果、来源更新、剩余限制、下一项依赖。实现状态用 planned / in_progress / verified / blocked；P00 可用 documented-baseline 表示仅规划/源码基线完成，不能冒充业务 verified。

- 井井在M01-D后回复“继续”，并追加单/双声道、波形高度、紧凑列表/全选、Praat语谱图和长音频显示要求。M01-E现为verified（限定Windows目录、草稿、预览与显示），见docs/testing/m01-workspace-report.md。新增m01-ui项目内环境、真实Praat受限预览；随后F2已接通持久页面，完整模块仍in_progress，下一项M01-G。
- 井井在M01-E布局修订后回复“继续”，已完成M01-F1执行准备：244项Windows定向测试、WAV与可选参数切片实际回读、批次策略和005增量SQL，见docs/testing/m01-execution-preparation-report.md。F1当时未执行005。井井随后对docs/testing/m01-migration-review.md回复“好，继续”，已授权并执行两库005和限定合成测试；F2持久提交/取消/恢复/发布及页面实际保存现已接通，见docs/testing/m01-persistent-report.md。下一项G，不重复索取005授权，不重复DDL。

- 井井在F2交付后回复“好，继续”，已执行M01-G本轮Windows联合审阅：真实上传/三组双格式下载回读、四份已授权自然录音对照、明确错误与小窗口布局及说明书更新；见docs/testing/m01-report.md。完整G/M01继续in_progress，优先补旧PKL图形转换和历史参数导入，不能把新文件供v2读取通过说成旧文件导入已实现；005不重复执行，测试继续用独立Chrome/Qt。

- 上条后的“继续”已执行M01-G旧格式批次：本机有界符号PKL转换/图形保存、历史XLSX/SQLite显式关联及原帧同步切分均已Windows定向验证，见docs/testing/m01-legacy-report.md。430项Python、17项前端、15步Qt及真实Chrome下载回读通过；原v2只读。完整M01仍in_progress，下一步39项最终逐条审阅，再迁M02；不重复DDL，不运行Codex内置浏览器关闭。

- 2026-09-11 井井将 M10 声道提前迁移用于 Windows EXE 录视频，之后明确追加 R4 录制增强与短帧/静音帧。M10/R4 现为 **verified / 已迁移（限定 Windows 本机声道录制）并暂时冻结**。历史证据见 docs/testing/m10-report.md，最新证据见 docs/testing/m10-recording-features-report.md，说明见 docs/manual/vocal-tract.md，入口为 dist/m10-recording/PhoneticToolbox-v3-M10-R4.exe。169 项 Python、22 项前端、12 组几何、两组真实单文件 Qt 与当前/六视图独立视频解码通过。最短帧 0.05 秒、新增默认 0.2 秒，静音帧不允许绘制 F0；包含本地文件/构形库、150 Hz 默认、缓存及同步视频。原生桥接 m10/2 保持不变，包含此前打包 ICU/UTF-8 修复。本任务只更新 M10，其他模块按各自计划和授权推进；其他平台声道仍 planned，不把本次标为全平台或正式发行。

- 2026-09-11 井井追加 M10/R5 起声与静音后渐入、拖动排序、一键清空，明确要求修改后直接生成 EXE、不检验。本次仅实施与打包，不运行测试或自检，也不将 R4 verified 扩展到 R5。最新入口 dist/m10-recording/PhoneticToolbox-v3-M10-R5.exe，说明见 docs/plans/2026-09-11-m10-onset.md。
