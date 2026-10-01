# M04 LPC 谱图实施计划

2026-09-13，井井在EGG收口后明确继续M04。完整M04为in_progress，按下面独立检查点推进；LPC学术引用补齐、代码来源做有界查找（井井最新追加授权），EXE/生产部署暂停，不推进M05，不执行DDL或修改V2。

## 架构与页面

沿用既定V3工作台：左侧紧凑文件列表，主区切换波形/LPC，参数区设置阶数与显示范围，公共试听/任务/保存控件。小窗正常滚动，浅深色和全局字体从第一版接入，IPA固定Doulos SIL。复用已有TextGrid与波形能力；两套时间/频率视图分别保留状态。

优先复用已迁移的文件/任务/导出适配，增加真实LPC核心与契约。完整复制旧Qt页面会破坏共用界面，浏览器另写LPC会引入第二套数值实现，均不采用。纯显示切换无需新建任务；开始处理产生带参数/输入快照的持久结果与PNG。桌面保存到授权目录，网页通过已有结果下载。

科学规则先按 [源码映射](../modules/evidence/M04-source-map.md)原样迁移。清选区、谱图重算时间轴和同名保护作为单列行为修复。原直接自相关存在长ROI耗时风险，B阶段测定受控进程预算和输入上限，再开放页面，不以异步按钮代替计算资源限制。

## 分阶段文件与验证

| 阶段 | 文件/工作 | 退出条件 |
| --- | --- | --- |
| A 基准 | `scripts/capture_m04_baseline.py`、`tests/fixtures/m04/`、说明书/源码映射 | 原V2两轮谱线、错误、格式、标签、PNG复核，来源文件不变 |
| B 纯核心 | `packages/phonetic_core/src/phonetic_core/lpc/`、纯核心/冻结对照测试、项目环境锁 | 与A逐值对照，输入无修改，预算与错误明确，安装wheel复验 |
| C 任务/导出 | contracts、`ptb_worker`模块适配、300DPI/字体快照与结果清单 | 使用既有任务/配额/资产协议，取消/重试/保存与双账号隔离适用路径通过；不预设需要新DDL |
| D 页面 | `frontend/src/modules/lpc-spectrum/`、公共波形可选适配、平台能力及页面注册 | 四功能组完整，原有模块手势不受影响，Chrome真实任务及小窗/浅深色/草稿检查 |
| E 收口 | 使用说明、验收报告、功能矩阵 | A01–A20有明确证据和未测边界，开发态阶段收口 |

A已完成限定基准，见 [报告](../testing/m04-baseline-report.md)。B纯核心现为限定Windows verified，见[核心报告](../testing/m04-core-report.md)；C任务/导出已限定Windows verified，见[任务报告](../testing/m04-jobs-report.md)；D页面已限定Windows独立Chrome verified，见[页面报告](../testing/m04-ui-report.md)。E 已完成自然录音 Qt 宿主与新增语谱图交互，但托管账号 Chrome 被专用旧政策测试库的写入门禁挡住，状态仍为 in_progress；20 项逐项结论见[专项报告](../testing/m04-e-report.md)。完整 M04 不标 verified。

## 20项验收

| 编号 | 正常路径 | 边界/异常 |
| --- | --- | --- |
| A01 | 目录/上传及刷新 | 无WAV/目录撤回 |
| A02 | 文件切换 | 旧读取/结果迟到不覆盖新选择 |
| A03 | 位深转换/单多声道均值 | 空文件、损坏、非有限值 |
| A04 | 同名或显式关联TextGrid | 缺失/无效/时间范围不匹配 |
| A05 | 首层/循环层级/标签 | 空tier、重名、跨区末端与重复标签 |
| A06 | 全部默认值与草稿 | 阶数/空值/频率/dB无效输入 |
| A07 | 动态纵轴 | 显示上限超过Nyquist不补假曲线 |
| A08 | 缩放/平移/普通页面滚动 | 边界、窄窗、键盘可达 |
| A09 | Shift框选/清除选区 | 反向、零长度、移出画布 |
| A10 | 无框选采用可见时间范围 | 谱图状态重新处理不把Hz当秒 |
| A11 | 框选优先/样本半开范围 | 非整样本时间、尾部、越界 |
| A12 | 固定1024点原LPC | 不同采样率/阶数/谱值冻结对照 |
| A13 | 最短有效ROI | 阶数过高、静音、奇异输入 |
| A14 | 长ROI受控计算 | 超预算、取消、超时回收 |
| A15 | 波形/谱图切换保留结果 | 编辑参数后旧结果明确状态 |
| A16 | 可见范围/选区试听与停止 | 切文件/切标签/关页时停止 |
| A17 | 300DPI白底黑线PNG | 字体缺失、失败保留图、像素/DPI回读 |
| A18 | 含IPA标签命名/保存下载 | 非法字符、同名保护、输出目录取消 |
| A19 | 任务提交/取消/重试/重开 | 迟到返回、服务失败与草稿失败 |
| A20 | 本机与托管网页流程 | owner隔离/配额/到期适用检查，设备/生产单列 |

所有阶段记录实际命令。A执行 `& .venv/m09-ui/Scripts/python.exe -X utf8 scripts/capture_m04_baseline.py --freeze-public`（只首次冻结，已有基准时省略该开关），静态检查使用 `scripts/validate_docs.py` 和 `scripts/check_architecture.py`。B–E的具体回归命令随实现登记，不将旧计划尚不存在的npm命令冒充可运行入口。

B边界依据ADR-047：单次48,000样本上限、参数/ROI与有限值验证、前后协作取消。运行 `scripts/probe_m04_budget.py`、`tests/parity/test_lpc_spectrum.py` 与 `tests/parity/test_lpc_boundaries.py`，安装独立wheel目标目录后复验。C再实施进程硬预算。

## C 任务与导出实施（2026-09-13）

井井在B交付后确认继续。新增 `backend/src/ptb_api/lpc_models.py`、`ptb_worker/lpc_{jobs,child,runtime,exports}.py` 与固定bootstrap，公共executor/manifest/retry/路由增加lpc_analysis，desktop TaskBridge复用既有本地输入与非覆盖保存。契约由后端生成。无新表或DDL。

m04/1单文件请求包含WAV引用、可选TextGrid引用、显式ROI、tier和完整字体快照。产物固定lpc.ptb.json（谱值/样本选区/参数/来源）、lpc_SPECTRUM.png（2400×1350、300DPI白底黑线）、lpc_AUDIO.wav（单声道原比例FLOAT64选区）。标签依原V2规则，图中IPA使用Doulos SIL。显示名字含源名/标签/时间，实际文件资产使用固定安全名。

复用M03兼容运行环境，LPC独立固定入口；核心wheel更新仅限本项目m03-compatible，第三方依赖不变。整份输入64MB、最多800万帧和8声道、8–96kHz，读取后只复制/转换ROI（最多48000帧），不对整份文件做LPC。TextGrid限2MB并复用已有解析器。单子进程30秒/2GB、结果8MB，任务截止300秒， scratch与结果占用沿用现有计量。首次缺字体以明确错误结束，不发布半套结果。

定向命令：`pytest backend/tests/test_m04_contract.py backend/tests/test_m04_exports.py`；`scripts/verify_m04_jobs.py`（实际本地HTTP/存储/子进程/保存/取消/重试/故障）；`scripts/verify_m04_server.py`（项目专用测试PG、双账号与结果隔离/配额/到期）。协议生成检查、前端typecheck与构建。接口和导出验收不代表D页面或EXE完成。

## D 页面实施（2026-09-13）

井井在C交付后继续授权。文件为 `frontend/src/modules/lpc-spectrum/`、`platform/research.ts`、`platform/desktop.ts`、`AppShell.vue` 及公共波形可选Shift行为。详见ADR-049。复用真实任务/字体/保存接口，原始波形、频谱、参数、任务历史置于紧凑可滚动主区，TextGrid及文件放左栏。验证 `npm test`、`npm run typecheck`、`npm run build`，独立Chrome用实际TaskBridge/兼容子进程与合成输入检查计算、保存、切换、取消、草稿和小窗。托管认证联合与完整A01–A20收口留E，不把本机浏览器桥接称为托管部署验收。

## E 追加交互与退出门（2026-09-26）

井井追加手动选区与可切换语谱图，见 ADR-056。既有波形 Shift 拖动已在 D 和 P04 浏览器回归中通过，本轮仍用实际浏览器核对。M04 在波形下方使用现有 Praat 语谱预览，拖动语谱图与波形共用同一时间选区；异常预览不能阻止原有 LPC 任务。修改限 M04 页面与通用显示组件的可选接口，后端科学核心、任务协议和默认值不变。回归包含双图正反向选区、清除与可见窗、缩放后的时间映射、预览失败、频谱视图时间隔离，以及其他模块默认手势。

E 的 20 项证据逐项收录在专项报告。Windows 本机 Chrome/Qt、授权自然录音、Linux 受限合成后端、真实托管账号网页、资源与发布分别标状态。专用 PostgreSQL 旧政策库遇到 `storage_policy_migration_required` 时停止写入验收；本任务不执行 DDL 或政策迁移。只有托管 Chrome 的上传、任务、历史和三文件下载实际通过后，才能将 A01/A20 的该范围标为 verified。

## M04-R1 试用修复（2026-09-29）

井井授权修复自然文件的全部任务失败、统一左键拖选、操作收进左栏及关联 bug。当前限定 Windows 修复和验证见[专项计划](2026-09-29-m04-repair.md)与[报告](../testing/m04-r1-report.md)。本节覆盖 A04 的空白尾段处理、A09 的 Shift 必须按住旧行为和 D 的参数位置；原核心算法与数值基线保持。原 E 托管账号网页、Linux/设备与发行未验边界仍独立保留。
