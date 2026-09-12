# PhoneticToolbox 3.0 · 开发工作台

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
| 6. 登录、5 GB、7 天 | [账号与存储规格](docs/specs/accounts-storage-jobs.md) |
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
