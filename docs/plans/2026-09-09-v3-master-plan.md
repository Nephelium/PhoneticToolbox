# PhoneticToolbox v3 全平台重构实施计划

> **执行约定：** 秋叶及后续开发者必须遵守根 AGENTS.md，按本计划逐项实施、验证和记录；当前 P01 原型与试用通过、P02 工程骨架已通过 Windows 定向验收；全面业务迁移尚未开始。用户已确认的要求见 requirements.md，具体技术候选不得伪装成已验证选择。

**Goal:** 在保留全部科研功能和正确性的基础上，交付统一 U2/K2 视觉的独立网页版、Windows 单文件与安装版、macOS，以及单独验收的 Linux 桌面版。
**Architecture:** 一个自有仓库，共用 Web 前端与版本化科学核心；桌面本地宿主和服务器 API 通过相同契约接入。用户、任务、配额、文件由外层负责，科学计算不复制为两套。
**Tech Stack:** Vue 3/TypeScript/Vite、Python/FastAPI、Qt WebEngine 候选、PostgreSQL/独立 worker；最终版本与宿主在原型阶段冻结。
**Status:** D0.3 规划基础，2026-09-09 已推进至 P02 骨架 verified。仅已标 verified 并链接验收报告的项目代表实际验证；其他命令/目标文件仍是计划。

## 1. 阅读与执行顺序
先读 [要求](../requirements.md) → [总架构](../../ARCHITECTURE.md) → [UI](../design/UI_SPEC.md) → [模块](../modules/module-migration.md) → [账号/配额](../specs/accounts-storage-jobs.md) → [来源](../references/source-audit.md)。
技术风险先验证，再铺开；一次一个可审阅任务。每个任务的原样迁移与行为变更分开。不能做到一半临时另起一套前端/后端或为个别模块绕过公共架构。

## 2. 阶段与交付
| 阶段 | 任务 | 核心交付 | 当前状态 |
| --- | --- | --- | --- |
| 基线与风险 | P00–P03 | 来源基线、宿主验证、依赖/科研基准 | P00 文档与继承完成；P01 Windows 原型/试用及风险评审通过；P02 骨架与 P03 独立行为基线 verified（限定 Windows 定向范围） |
| 公共基础 | P04–P07 | UI 壳、登录、任务、5 GB/7 天 | P04 公共界面 verified；P05 账号/项目及真实 PG 定向 verified；P06 Windows 持久任务定向 verified（含真实 PG/SQLite 与恢复复验）；P07 Windows 受控存储/工程生成/有界 ZIP 联合 verified（不含科学迁移、原生目录输出和生产跨平台验收） |
| 功能迁移 | P08–P09 / M01–M15 | 全部功能双端与设备适配 | P08/M01 in_progress，A–E已限定验收；持久批次与其他模块待实施 |
| 研究可追溯 | P10 | 软件/说明书/随包来源一致 | 已做首轮调查，未接软件界面 |
| 验证与发行 | P11–P15 | WSL 10 用户、各平台包和部署方案 | 未实施，Mac 暂无实机 |
| 退役与清理 | P16 | 去除重复/旧依赖，条件性旧目录清理 | 不执行，条件尚未满足 |

不承诺无证据的工期。P01 原型和 P03 数值基线决定后续估算；Mac 设备、原生构建和许可是独立外部约束。可先验证 Windows/Web，但不得擅自删除其他平台目标。

## 3. 工程粒度与记录
每个下面的步骤应拆成约 2–5 分钟的可核对动作，遇到较大实现先拆子任务，不将一个阶段变成一个不可审阅的大提交。
固定小循环：确认输入/任务 ID → 定义行为与失败用例 → 运行观察真实失败 → 最小迁移/实现 → 定向通过 → diff/来源/视觉审查 → 记录并本地提交。文档低风险改动只做合适的文档验证。
实现记录至少含 source_ref、changed_files、commands、results、screenshots（适用）、remaining_risks、next_dependency。未实现文件不要填“已创建”，跳过的测试不能写“通过”。

## 4. 基础与平台任务详情

### P00 · 建立可追溯源码与规划基线

**依赖：** 无。**状态：** 本轮已完成源码继承/文档准备；业务验证未完成。

**目标文件：** docs/baseline/source-manifest.json；docs/requirements.md；AGENTS.md；ARCHITECTURE.md。

1. 核对当前 v2 工作状态、未跟踪运行资源和用户 EXE 校验和。
2. 以独立索引建立本地源码基线，创建 codex/v3-rebuild worktree；不更改 v2 index/HEAD。
3. 继承可复用源码，记录 427 文件 hash；不拷贝环境和私有研究语料。
4. 归档已选设计和功能矩阵，标明旧架构已被替代。
5. 校验原目录、文档覆盖与新目录状态。

**退出条件：** 原 v2 文件/HEAD/index 不变；所有新文档能定位，状态不虚报。

**验证与文档：** 本轮执行源码 SHA-256、Git HEAD/index/status 对照和文档结构/链接/矩阵检查；报告入 docs/baseline 和 output/validation。

### P01 · 先验证宿主、原生依赖和许可风险

**依赖：** P00。**状态：** verified（Windows 原型与风险评审范围，用户试用通过，开发主线冻结见 ADR-013）；对应发行与跨平台门槛继续保留。证据见 [P01 报告](../testing/p01-host-probe-report.md) 和 [细化计划](2026-09-09-p01-host-probe.md)。

**目标文件：** desktop/experiments/host_probe.py；frontend/experiments/audio-viewport/；docs/decisions/ADR.md。

1. 在一次性隔离探针环境搭建最小本地 Qt WebEngine＋已构建页面原型，不修改 phonetic_311，不复制业务页面。P02 再建立正式 v3 开发/构建环境。
2. 测试播放、中文/IPA、多个轨道、精确采样选区和窗口缩放。
3. 验证 WebEngine 随单文件包启动/退出和非项目目录运行。
4. 确认 VTL/REAPER 跨平台构建来源、IRAPT/WM-PC/载瓦语材料许可缺口。
5. 比较 Qt 宿主与 Tauri/Electron/PySide6 的实际代价，冻结一个方案。

**退出条件：** 给出宿主原型报告；未通过单文件、设备和来源审查时不推进全面 UI 移植。

**验证与文档：** 按 docs/deployment/platform-release.md 与 docs/testing/verification-plan.md，保存实际构建/启动/设备平台证据，更新对应 ADR 和平台报告。

### P02 · 包边界、环境和契约脚手架

**依赖：** P01。**状态：** verified（Windows 包、环境与契约范围）。详见 [P02 计划](2026-09-09-p02-scaffold.md) 和 [验收报告](../testing/p02-scaffold-report.md)。

**目标文件：** packages/phonetic_core/pyproject.toml；frontend/package.json；backend/pyproject.toml；desktop/pyproject.toml；contracts/versions.md；scripts/validate_docs.py。

1. 清点实际已安装依赖与 v2 声明冲突；不改 v2 环境。
2. 创建 v3 独立开发/构建环境，锁定依赖和可用原生平台。
3. 建立最小可安装核心包、独立前端入口、本地/服务器服务入口。
4. 生成 OpenAPI/客户端类型流程及单位/轨迹 schema。
5. 建立禁止跨层导入、旧路径、未登记资源的架构检查。
6. 确认版本元数据来源，旧 v2 入口与新入口区分。

**退出条件：** 空环境可安装新包；同一 schema 不被两套代码分别手写；禁用依赖检查能抓住故意的越界例子。

**验证与文档：** 干净安装、架构导入检查、协议往返与锁定依赖核验；不在原 v2 环境升级包。

### P03 · 科研行为与数据基线

**依赖：** P02。**状态：** verified（独立原 v2 捕获、同平台重复性与 EXE 提取代码对照；完整 GUI 与科学真值未测，见 [P03 报告](../testing/p03-baseline-report.md)）。

**目标文件：** tests/fixtures/manifest.json；tests/parity/test_baseline.py；scripts/capture_v2_baseline.py；docs/baseline/capture-protocol.md。

1. 从本机只读语料中选取持续元音、嘎裂声、EGG 和带 TextGrid 样例。
2. 记录源文件 hash、音频头、完整参数、真实后端与环境。
3. 在独立输出目录捕获当前 v2 数值结果，并对用户使用的 EXE 做关键行为对照。
4. 为缺失类别生成可公开的合成 fixture，不提交个人语料。
5. 定义每参数容差、缺失 mask、时间轴、文件数及精确边界。
6. 将已知问题与迁移回归分开登记。

**退出条件：** 基准结果不能由 v3 自己生成；参数/时间/后端可复现，尚缺样例清楚标记。

**验证与文档：** python -m pytest tests/parity -q；样例 manifest 与原始数值由独立基准生成，报告 mask/时间/精度差异。

### P04 · 统一前端壳、主题、首页与公共组件

**依赖：** P02。**状态：** verified（公共界面经井井审阅，致谢分组已修订；限定工程范围见 [P04 报告](../testing/p04-workbench-report.md)）。

**目标文件：** frontend/src/app/AppShell.vue；frontend/src/design/tokens.css；frontend/src/components/AudioTransport.vue；frontend/src/components/MethodReferences.vue。

1. 实现 U2 主题令牌、字体和 K2 品牌资产，不夹带各页算法。
2. 建立全部 15 模块注册、侧栏搜索/折叠、标签与首次首页。
3. 实现播放器、选区状态、波形视图、参数抽屉、任务面板、统一错误空态。
4. 建立 browser/desktop 能力适配接口，让同组件在两个宿主演示。
5. 验证主题/标签切换保持未保存编辑，键盘焦点和多窗口播放协调。
6. 提交浅深色/缩放对照供审阅后，再用公共组件迁移页面。

**退出条件：** 首页和公共组件视觉通过；图表为真实 fixture；全部菜单存在但未实现模块不能伪装可用。

**验证与文档：** npm --prefix frontend run typecheck / test / build 和 UI_SPEC 中的状态/主题矩阵；评审实际截图而不是设计图。

### P05 · 服务器账号、项目和会话隔离

**依赖：** P02。**状态：** verified（限定 Windows 账号/会话/项目，实际 PG 建表、并发及受控重启通过；第 6 项任务/事件/取消/重试隔离已在 P06 通过，文件资源与下载仍待 P07；见 [P05 报告](../testing/p05-accounts-report.md)）。

**目标文件：** backend/src/ptb_api/auth.py；backend/src/ptb_api/projects.py；backend/tests/test_auth.py；tests/security/test_resource_ownership.py。

1. 设计并审阅 users/sessions/projects/owner 模型和数据库迁移。
2. 选择成熟密码哈希/会话实现，落实 Cookie、CSRF、退出撤销和登录限流。
3. 实现后端从会话确定 owner，任何请求不能选择其他 owner。
4. 实现登录/退出/当前用户和临时项目接口。
5. 前端登录页沿用 U2/K2；桌面默认无登录要求。
6. 以两个账号验证资源、任务、日志、取消与下载的全面隔离。

**退出条件：** 跨用户访问全部拒绝；登录可恢复，不依赖旧站；实际迁移执行前满足授权。

**验证与文档：** 运行 python -m pytest backend/tests tests/security tests/contracts -q；quota/job 对应场景不得以 mock DB 代替真实 PostgreSQL 竞态检查；更新接口和操作说明。

### P06 · 持久任务、worker 和本地统一服务

**依赖：** P02,P03,P05。**状态：** verified（限定 Windows 流程探针；已建表并通过真实 PG/SQLite、10 账号排队、取消/中断/重试、本机服务重启和网页隔离，见 [P06 报告](../testing/p06-jobs-report.md)）。

**目标文件：** backend/src/ptb_worker/claims.py；backend/src/ptb_worker/leases.py；backend/src/ptb_worker/executor.py；desktop/src/ptb_desktop/local_service.py。

1. 实现任务状态机、输入/配置快照与幂等请求键。
2. 实现 PG 原子认领、心跳、lease_generation 和崩溃后 interrupted。
3. 工作进程加载核心包，按任务隔离原生状态；API 不长时间占计算。
4. 实现可恢复进度、取消与结果 manifest 的事务提交。
5. 桌面使用相同 job contract，本地文件/设备由专属适配器管理。
6. 故意中断 worker、重复提交、旧租约提交和服务重启测试。

**退出条件：** 没有伪成功/重复最终产物/遗留本任务进程；10 用户可排队，不把进程内线程当持久队列。

**验证与文档：** 运行 python -m pytest backend/tests tests/security tests/contracts -q；quota/job 对应场景不得以 mock DB 代替真实 PostgreSQL 竞态检查；更新接口和操作说明。

### P07 · 5 GB 配额、7 天清理和结果管理

**依赖：** P05,P06。**状态：** verified（限定 Windows 受控存储、工程生成和有界 ZIP；003/004 均获具体授权执行，真实 PG/磁盘/任务引用/旧 worker/独立浏览器联合验收通过；见 [P07 计划](2026-09-09-p07-storage.md)及 [单文件报告](../testing/p07-storage-report.md)、[联合报告](../testing/p07-job-files-report.md)）。

**目标文件：** backend/src/ptb_api/quota.py；backend/src/ptb_api/storage.py；backend/src/ptb_worker/cleanup.py；frontend/src/modules/storage/StoragePage.vue。

1. 实现账户字节计量、原子预留、受控 writer 和恢复核对。
2. 实现上传/解压/生成/ZIP 全链路计量，不能靠前端判断。
3. 实现输入与输出独立 expires_at、到期禁读、删除与重启清理。
4. 实现占用列表、按大小/到期排序、直接删除和下载后手动清理。
5. 覆盖 Q01–Q20：竞态、断电、满额、物理删除失败、下载到期等。
6. 压测全局磁盘水位，建立错误/清理延迟可观测记录。

**退出条件：** used+reserved 不越界；满额仍可下载清理；过期数据不可读，正常运行按时物理删除。

**验证与文档：** 运行 python -m pytest backend/tests tests/security tests/contracts -q；quota/job 对应场景不得以 mock DB 代替真实 PostgreSQL 竞态检查；更新接口和操作说明。

### P08 · 普通研究模块逐页迁移

**依赖：** P03,P04,P06,P07。**状态：** in_progress（M01-A至E限定验收完成，其余模块与M01持久批任务仍待迁移）。

2026-09-09：M01迁移前审阅和[文件级实施计划](2026-09-09-m01-implementation.md)已完成；30源文件/80参数/14设置已核对，见[本轮报告](../testing/m01-planning-report.md)。[M01-A独立基准](../testing/m01-baseline-report.md)已完成：28例双轮捕获与23项测试通过；[M01-B核心](../testing/m01-core-report.md)已完成149项Windows独立wheel测试，M01-C适配已通过222项Windows wheel测试与真实双产物回读，[M01-D契约](../testing/m01-contract-report.md)已通过318项wheel测试与三组真实结果往返，下一项为M01-F持久批次与双端任务；不把核心验收计作整个模块verified。

**目标文件：** docs/plans/modules/M01-parameter-estimation.md 等模块计划；packages/phonetic_core/src/phonetic_core/；frontend/src/modules/。

1. 按 M01/M02/M03/M04/M06/M07/M08/M09/M13/M14 顺序逐个执行对应文件级计划。
2. 先原样搬可用算法，再拆外层 I/O；不把算法重写和 UI 换肤混成一个提交。
3. 每模块先对照 v2/契约，再接统一页面与本地/服务器两个 adapter。
4. 保留原入口和全部输出/参数，新增来源说明从注册表读取。
5. 每模块完成浅深色、取消/错误和结果导出审查后才迁下一个。
6. 跑该模块科研回归，不在失败时跳向更多模块堆积债务。

**退出条件：** 10 个模块各自满足原功能行和新增双端验收；无第二套服务端算法。

**验证与文档：** 逐模块运行其计划中的数值回归、契约、UI E2E 与实际设备检查；每个 Mxx 独立收口后再汇总，不能只跑一次主窗口启动。

### P09 · 设备、时序和原生高风险模块迁移

**依赖：** P01,P03,P04,P06,P07。**状态：** planned。

**目标文件：** docs/plans/modules/M05-lip-extraction.md；M10-vocal-tract.md；M11-mfa.md；M12-annotation.md；M15-perception.md。

1. 唇形：验证摄像头/麦克风权限、帧率、时间轴与平台差异。
2. 声道：VTL 原生构建、实例隔离、真实音频输出、几何与生理说明。
3. MFA：平台模型与可执行环境、持久任务、真实取消和日志。
4. 标注：前端编辑、TextGrid、独立唇偏保存和未保存保护。
5. 感知：刺激预加载、客户端时间戳、随机化/结果导出、断网策略。
6. 将未达到目标的设备/平台限制写入能力矩阵，不做静默降级。

**退出条件：** 5 模块分别有可复核数据/设备证据；Web 不假装等同原生硬件时序。

**验证与文档：** 逐模块运行其计划中的数值回归、契约、UI E2E 与实际设备检查；每个 Mxx 独立收口后再汇总，不能只跑一次主窗口启动。

### P10 · 全项目来源、说明书与软件致谢

**依赖：** P04,P08,P09。**状态：** planned。

**目标文件：** third_party/source-registry.json；third_party/references.bib；docs/manual/methods-and-credits.md；frontend/src/modules/about/ReferencesPage.vue。

1. 逐函数复查优先算法，区分移植、依赖、论文方法和项目集成。
2. 补齐实际使用版本、原版权/许可证、修改说明和数据使用范围。
3. 软件关于与各模块方法页接同一注册表；提供仓库、DOI、官方 PDF 和引用复制。
4. 重新编排说明书全部章节，操作步骤与实际页面逐项对应。
5. 生成随包许可/参考文献/依赖清单，检查所有 UI 来源入口离线可读。
6. 解决未明确授权的代码/数据；必要时联系作者前另获发信授权，或采用合规替代方案并审阅。

**退出条件：** 没有未登记外部组件；不存在“全部原创”错误归属；对应发行许可缺口未解决则该发行物不发布。

**验证与文档：** python scripts/verify_sources.py、注册表/软件/手册三方一致性与离线许可检查；未知来源不自动赋 MIT。

### P11 · WSL 多人集成、科学对照和负载测试

**依赖：** P08,P09,P10。**状态：** planned。

**目标文件：** tests/security/；tests/performance/；tests/e2e/；docs/testing/load-report.md。

1. 在明确归本项目的 WSL 环境部署 API/worker/PostgreSQL/存储。
2. 10 个测试账号执行不同模块、排队、下载、到期和清理。
3. 验证 2 重任务+8 轻交互与 10 人同时提交，记录资源和等待时间。
4. 注入数据库/worker/磁盘/网络故障，核对恢复与跨用户边界。
5. 运行所有模块数值对照与 UI 场景，按真实环境标记缺口。
6. 用数据形成未来服务器规格与容量建议，不在本轮购买。

**退出条件：** 10 人场景指标有实测；安全/配额/TTL 必测全通过，设备未测不被掩盖。

**验证与文档：** 运行 python -m pytest backend/tests tests/security tests/contracts -q；quota/job 对应场景不得以 mock DB 代替真实 PostgreSQL 竞态检查；更新接口和操作说明。

### P12 · Windows 单文件与安装版发行链

**依赖：** P01,P08,P09,P10。**状态：** planned。

**目标文件：** desktop/packaging/windows/phonetictoolbox.spec；desktop/packaging/windows/installer.iss；scripts/build_windows.ps1。

1. 构建前端与核心 wheel，生成带版本和 hash 的发行清单。
2. 收集 WebEngine、Python、REAPER/VTL/其他原生依赖和字体字典。
3. 生成单文件 EXE，验证离线/首次启动/退出/目录权限/Unicode。
4. 制作安装包、真实快捷方式/卸载入口和升级策略。
5. 干净 Windows 账号、无 Python/Node 环境安装/升级/卸载测试。
6. 验证版本一致、来源清单在包内、默认保留用户研究数据。

**退出条件：** 单文件和安装版分别有验收记录；文件夹诊断版不冒充单文件交付。

**验证与文档：** 按 docs/deployment/platform-release.md 与 docs/testing/verification-plan.md，保存实际构建/启动/设备平台证据，更新对应 ADR 和平台报告。

### P13 · macOS 构建与实机验收

**依赖：** P01,P08,P09,P10。**状态：** planned。

**目标文件：** desktop/packaging/macos/；scripts/build_macos.sh；docs/testing/macos-report.md。

1. 准备经授权的 macOS 构建环境，确定 arm64/x86_64 支持矩阵。
2. 构建 Qt helper、Python wheel、REAPER/VTL dylib 并检查架构和加载路径。
3. 制作 .app/分发包，验证音频/摄像头权限文案与原生窗口。
4. 依据实际发行方式处理签名/公证，保留可追溯构建信息。
5. 取得 Mac 实机或测试者证据，执行离线/设备/退出/主题验收。
6. 没有实机的项目明确 pending，不以 CI 构建成功代替实际可用。

**退出条件：** 发布目标架构均通过相应构建和设备验收；当前暂无 Mac 不阻断其他平台规划。

**验证与文档：** 按 docs/deployment/platform-release.md 与 docs/testing/verification-plan.md，保存实际构建/启动/设备平台证据，更新对应 ADR 和平台报告。

### P14 · Linux 桌面发行与兼容性

**依赖：** P01,P08,P09,P10。**状态：** planned。

**目标文件：** desktop/packaging/linux/；scripts/build_linux.sh；docs/testing/linux-desktop-report.md。

1. 选择最小支持发行版和架构，记录 glibc/图形/音频依赖。
2. 构建原生工具与 VTL、核心 wheel 和共用前端。
3. 制作选定 AppImage/安装包及 .desktop 图标信息。
4. 测试 X11/Wayland、字体、音频设备、相机与沙箱配置。
5. 执行非项目目录/离线/退出/升级验证。
6. 列出真实支持矩阵，不把 WSL 后端测试计入桌面通过。

**退出条件：** 目标 Linux 桌面环境可运行并有设备证据；不能承诺所有发行版。

**验证与文档：** 按 docs/deployment/platform-release.md 与 docs/testing/verification-plan.md，保存实际构建/启动/设备平台证据，更新对应 ADR 和平台报告。

### P15 · 正式部署与分平台发布准备

**依赖：** 网页依 P11；Windows 依 P12；macOS 依 P13；Linux 桌面依 P14，分别验收发布。**状态：** planned。

**目标文件：** docs/deployment/production-runbook.md；docs/deployment/rollback.md；docs/releases/3.0.0.md。

1. 审查各平台验收状态，区分完成/未测/阻断，不夸大覆盖。
2. 根据负载报告查询阿里云实时规格/价格，提出具体部署资源。
3. 形成同源 HTTPS、数据库、worker、存储/配额/清理与最少日志配置清单。
4. 演练部署迁移、服务恢复和版本回退；核对备份不会超期保存用户音频。
5. 准备发布说明、说明书、下载文件名/校验和、签名和第三方材料。
6. 把生产部署/push/对外发布作为明确可审阅的最后动作，获授权后执行。

**退出条件：** 有可复核发布包和运维步骤；未通过的平台可单列待发布，不能在发布说明中宣称可用。

**验证与文档：** 保存可复核的文件、状态和验证清单，实际变更前重新核对授权与路径；公开动作最后执行。

### P16 · 退役过渡代码与条件性旧目录清理

**依赖：** P08,P09,P10,P11。**状态：** planned。

**目标文件：** docs/baseline/retirement-report.md；新工程 import/build 扫描；旧目录处理记录。

1. 确认新入口只依赖新包，所有功能验收映射已闭合。
2. 检查 v3 继承旧目录的唯一资源/说明/来源是否已迁入正式位置。
3. 扫描构建产物与运行时，证明没有旧路径和无意双实现。
4. 旧前端/API 重新查依赖、未提交修改和独有数据，不能因 v3 不依赖就判整个项目无用。
5. 只对明确确认无必要保留的精确目标应用用户的条件性删除授权。
6. 先列出清理清单与证据，再执行边界校验；v2 一直保留。

**退出条件：** v3 结构简洁且完整；没有删除仍有用途/独有改动的旧项目，更不会删除 v2。

**验证与文档：** 保存可复核的文件、状态和验证清单，实际变更前重新核对授权与路径；公开动作最后执行。

## 5. 每个模块的固定交付包
- source-map：旧路径、继承 hash、来源 ID 与迁移去向。
- config/schema：原参数键、默认值、单位、范围、输出格式。
- core 与 service：先复用，再去除 Qt/机器路径耦合；算法修改单列。
- 页面：U2 公共组件、浅/深、原版所有操作、错误/取消与未保存保护。
- adapter：桌面目录/设备与 Web 上传/任务/下载各一套边界适配。
- tests：独立数值对照、格式和原生设备（适用）；缺口明确。
- documentation：说明书、方法与引用、来源登记、平台能力和变更记录。

完整的 M01–M15 文件级计划见 [模块索引](../modules/module-migration.md)，83 功能组逐行包含其中。不能用此模板替代逐模块具体内容。

## 6. 审阅门与变更范围
G1：P01 宿主/单文件/原生/许可风险可评估后冻结技术主线。
G2：P03 科研基线可重复后开始大规模迁移。
G3：P04 首页/公共组件视觉获审阅后统一应用到所有模块。
G4：P07 账号/额度/删除行为正确后接入真实多用户处理。
G5：M01–M15 完整功能矩阵闭合，软件/说明书来源一致。
G6：分平台发行准入和生产部署清单已验证后才公开发布。

新想法先写变更条目和影响分析；不因“顺手优化”改变输入输出、额外存储、默认算法或原生依赖。
源项目查验发现的科学错误、缺失授权和依赖不一致各自有 issue；既不能无视，也不能把整个任务无限扩成新研究。

## 7. 当前明确未决项
- 载瓦语补充仓库未发现明确许可证；确认代码/数据的分发条件，不把论文 CC 条款直接套给补充代码。
- VoiceSauce 直接移植片段与 OpenSauce 参照范围仍需逐函数对应；旧文档 Iseli 1999 年份需改为实际查验来源。
- v2 依赖约束与安装版本不一致；不能原样安装声明便认定数值可复现。
- VTL 跨平台原生构建、实时音频和单文件 WebEngine 启动待原型验证。
- 无 Mac 实机；云构建只覆盖部分环节。
- 完整科研预期结果尚未捕获；本轮只做输入头部、源码与文献证据检查。
- 旧前端有未提交修改，旧 API 当前没有有效 Git 根；删除条件不成立，继续保留。

## 8. 启动实施前应完成的审阅
井井主要审阅外观、功能范围、账号/5 GB/7 天规则和分平台目标；依赖、路径、代码来源、测试组织由秋叶按本计划处理。
确认后从 P01 开始，完成可审阅原型与风险记录再推进；不要直接重写全部页面。

2026-09-10补充：[M01-E](../testing/m01-workspace-report.md)目录/草稿/试听/显示已限定验收，包括井井追加的Praat语谱图与长音频显示要求；M01/P08整体仍in_progress。
