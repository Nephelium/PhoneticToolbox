# M04-E LPC 谱图开发态收口与 Linux 证据复核

2026-09-27。**M04-E 为 in_progress；Windows 本机开发态的新增交互及正式 Qt 宿主自然录音范围 verified，托管账号 Chrome 联合链路未通过。** A/B/C/D 的原有限定 verified 不变，完整跨平台、生产模块不标完成。本报告汇总历史证据，不为写报告复跑科学全套。对应[实施计划](../plans/2026-09-13-m04-implementation.md)、[操作说明](../manual/lpc-spectrum.md)和[V2 说明书/源码映射](../modules/evidence/M04-source-map.md)。

## 本轮结论与改动

- 原 M04 波形已支持 **Shift＋拖动**手动选区，反向框选及样本半开区间在真实 Chrome 再验。新增“显示语谱图（Praat）”开关：波形下方紧接实际语谱图，直接拖动语谱图可选同一时间范围；波形、时间输入、下一次 LPC 任务使用同一选区。实测拖动 80%→20%，任务请求与显示时间一致。预览故障独立提示，不妨碍 LPC。该预览不参与 LPC 求解。
- 仅改 M04 页面和公共波形/语谱组件的**可选接口**；其他模块默认手势不变。科学核心、任务协议、默认 50 阶/8000 Hz/−5～35 dB、导出算法未改。页面沿用 P04 统一工具栏、ModuleStatus、公共字体、科学图表、任务和播放组件，无重复模块大标题/关闭按钮；图像仍为单条 LPC 谱包络、Hz/dB 轴及固定 1024 点语义。
- Qt 正式工作台从 P03 已授权的 LOCAL-01、LOCAL-12 原路径**仅在本机读取**，复制到被忽略的独立测试目录，不上传远程。两例 44.1 kHz，各分析 0.2 秒/8820 帧，原 WAV/TextGrid 哈希前后一致，同名 TextGrid 已关联，结果 JSON 各含 1024 点；保存的 PNG 为 2400×1350、WAV 为 FLOAT64 8820 帧，JSON 源哈希匹配。Qt 中的可选语谱图实际呈现在波形下方，截图已人工检查。
- 托管网页使用既有 P05/P07 **专用测试 PostgreSQL** 与本轮自启的本机服务，不接触用户手动服务。共享 P07 政策运行时代码要求 policy version 2，专用旧库仍为 version 1；只读预检返回 `storage_policy_migration_required`。此前三次浏览器尝试在上传第一步被门禁拒绝，无法完成任务、历史和三文件下载。三次尝试可能留下各两组测试账号/项目，无成功上传资产；不删已有行。本轮没有执行 006 DDL/迁移，也不把旧 5 GB/7 天的 C 阶段适用检查冒充新 1 GB/3 天验收。

## 原 20 项验收与功能映射

表中 B/C/D 分别指[纯核心](m04-core-report.md)、[任务/导出](m04-jobs-report.md)、[页面](m04-ui-report.md)的既有范围；新证据在下节给出。`限定通过`仅指列出的实际平台和路径，不表示整项所有边界已在托管网页重验。

| 项 | 说明书/源码功能与已有证据 | 本轮补证及现状 |
| --- | --- | --- |
| A01 | 目录、上传、刷新；D 真实本机 WAV 列表/刷新、失效源清预览 | Qt 两自然录音目录通过；托管上传被旧政策门禁阻断，**网页未完成** |
| A02 | 切文件；D 旧读取、结果、目录选择迟到归属及刷新失效 | 本轮 Chrome 27 组含旧读/历史迟到，限定通过 |
| A03 | 位深及多声道均值；A 冻结 PCM16/32、uint8、float32，B 5 组逐字节，C 1/8 声道 | Qt 两自然 WAV 实际计算，原文件哈希不变；损坏/非有限值沿用 B/C 错误证据 |
| A04 | TextGrid 同名/显式关联；C 越界/无效拒绝、D 关联 | Qt 两份同名 TextGrid 关联并保留哈希；托管显式资产关联未重验 |
| A05 | 首层/循环层/标签；A/B 冻结 IPA、末端排除与去重，D 页面切层 | Qt 显示首层标注；完整异常层沿用 A/B/C |
| A06 | 默认值与草稿；B 默认/范围，D 空值、重开草稿、关闭保护 | Chrome 旧快照、草稿保存及无效输入拒绝复验，限定通过 |
| A07 | 动态纵轴；A/B 冻结，D 动态控件、Nyquist 空白 | Chrome 24px 刻度与手动 dB 禁用复验；无假曲线，限定通过 |
| A08 | 缩放/平移/页面滚动；D 频率键盘及深色窄窗、普通滚轮 | Chrome 深色 1440/1000/390、滚动与频率轴隔离复验，限定通过 |
| A09 | V2 波形 Shift 框选/清除；D 反向拖动 | Chrome 再验 Shift 手动框选；新增 Praat 语谱图直接反向拖选并联动波形/任务；预览故障恢复，限定通过 |
| A10 | 无框选采用可见时间窗；D 清除、频率视图不混时间 | Chrome 重新处理范围复验，限定通过 |
| A11 | 框选优先、半开样本；B/C 切片与尾部，D 4800–9600 例 | Chrome 新语谱图 ROI 与下一任务请求一致；Qt 0–8820 和 176400–185220，限定通过 |
| A12 | 固定 1024 点 LPC；A 12 场景/38 数组/44,497 值两轮逐字节，B wheel 冻结 | Qt 两自然录音 JSON 各 1024 点；不把不同平台浮点称逐位相同 |
| A13 | 最短 ROI/静音/奇异；A/B 51/52 样本、NaN/Inf，C 错误发布边界 | Chrome 静音失败、修改后重试及草稿保留，限定通过 |
| A14 | 48,000 样本上限与受控计算；B 预算，C Windows 取消/超时/回收 | Linux P11 真实受限进程的取消、SIGKILL、30 秒超时沿用；自然长录音/极限压力未做 |
| A15 | 波形/频谱切换保留结果；D 旧结果提示、无额外任务 | Chrome 再验切换及旧快照，限定通过 |
| A16 | 可见/选区试听停止；D 实际浏览器播放与切换清理 | Chrome 播放及选区再验；真实声卡输出、多屏设备未测 |
| A17 | 300 DPI 白底黑线 PNG；A/C 像素、pHYs 约 299.9994、缺字体失败 | Qt 自然录音保存 PNG 回读 2400×1350，Chrome 字体/导出历史证据沿用，限定通过 |
| A18 | IPA 命名/保存/下载；C 同名保护，D Chrome 三文件实际下载 | Qt 自然录音本地三文件回读；托管账号下载**未完成**，新政策额度/期限也未联合验收 |
| A19 | 任务提交/取消/失败恢复/历史/草稿/关闭；C 真任务，D 迟到及关闭 | Chrome 27 组含取消、故障重试、历史快照；无效草稿关闭失败、取消保留编辑、保存后关闭已实测；Qt 成功路径，限定通过 |
| A20 | 本机与托管网页；C 实际 PG 双账号 ASGI owner/旧配额/受控到期，D 本机 Chrome | Qt 正式宿主自然录音、P11 Linux 合成后端已验；**托管 Chrome 登录→上传→任务→历史→下载未通过**，生产/自然 Linux 未测 |

V2 说明书的四组 F01–F04 均有入口与证据。原手册与源码的差异仍按[源码映射](../modules/evidence/M04-source-map.md)处理：频率上限仅裁剪显示，标签末区间规则原样，原图只有 LPC 曲线；“清除选区”、同名保护及可选语谱图为明确的 V3 交互补充。

## 平台、网页及资源分列

| 范围 | 状态与证据 |
| --- | --- |
| Windows 科学/任务 | A/B/C 原范围 verified；无本轮算法或默认值修改。C 的真实 PG ASGI 双账号/旧配额/受控到期仅为当时政策的历史证据 |
| Windows Chrome 页面 | 新增/回归 27 组通过；`output/validation/m04-ui/5d377372e603444f9260f3894b6828f2/report.json` 与 `wave-spectrogram-selection.png`。实际本机文件桥接和兼容科学子进程，不称托管账号页面 |
| Windows 正式 Qt / 自然语料 | `output/validation/m04-e/qt-73081b9227cb46a797a26971fdc375ff/report.json`、两份谱图、`LOCAL-01-wave-spectrogram.png`、保存三文件。两份为 P03 授权样例，无远程上传 |
| 托管 Chrome / 专用库 | `output/validation/m04-e/web-7a26c794cf4c4992a5e2d5bc081e02ca/report.json`：只读预检 `storage_policy_migration_required`、`schema_applied=[]`、本轮拥有 PG 已停。初次实际页面失败截图在 `web-1e47d3b9d8a04be4bba59e26f169d319/web-failed.png`。脚本 `scripts/verify_m04_web.py` 与 `tests/e2e/m04-web.cjs` 保留供专用库经独立授权完成迁移后重验；当前**不标通过** |
| Linux 后端 | 沿用[P11 报告](p11-task-flow.md)真实阿里云 Ubuntu 24.04 受限服务：合成 LPC 7 任务含成功/部分写入失败/取消/SIGKILL/超时，回环 HTTP 及 PNG/JSON/WAV hash 下载，Praat 预览、字体三角色。LPC 进程组峰值 105,717,760 bytes，MemoryMax 1 GiB、1 核、无 swap。未测 Linux 自然语料、账号 PG 浏览器与全机多任务压力 |
| 统一 UI | 沿用[P04-UNIFY](p04-unify-report.md)工具栏/状态/标签关闭证据；本轮真实 Qt 与 Chrome 图像检查保留原科研图语义，M03/M01/M02 共用波形手势 5 组回归通过 |

## 实际命令与结果

在项目根目录运行：`node tests/e2e/m04.cjs` 27 组通过；`node tests/e2e/m03-function-review.cjs` 5 组通过（Vite 扫描提示既有 `three` vendor 未解析，实际页面检查完成）；`npm --prefix frontend test` 137 passed；`npm --prefix frontend run typecheck` 与 `npm --prefix frontend run build` 通过，构建保留既有大 chunk 提示。`backend/tests/test_m04_contract.py` 与 `desktop/tests/test_m03_save_names.py` 定向 13 passed。Qt 命令使用既有 `.venv/m09-ui/Scripts/python.exe scripts/verify_m04_qt.py` 和项目内 `PYTHONPATH`，成功如上。网页同环境运行 `scripts/verify_m04_web.py`，按预期在只读政策门禁返回非零；没有跳过失败来标绿。`scripts/check_architecture.py` 的 `errors=[]`，`scripts/validate_docs.py` 最后快照检查 761 文件，仅有两条既有 M10-R5 EXE 缺链；未在 M04 范围修补。所改文件 `git diff --check` 与两个新增 Python 脚本 `py_compile` 通过。

浏览器测试使用真实本机 RPC、文件与科学子进程；旧响应、缺字体、预览失败等为受控注入，不冒充自然生产故障。Qt/Praat 截图证明可见布局，语谱图属于定位预览，不推断其与 LPC 频谱数值等价。原始录音、TextGrid、V2 源码均未写改。

新增语谱图浏览器用例首轮在上一任务结果尚未显示时读取当前谱图，导致测试等待对象错误；改为先确认结果所属源文件与时间，再继续切换，产品逻辑未因此改变。Qt 截图首轮拍到缩放触发的语谱预览加载态，增加等待最终画布绘制后复拍。失败证据保留在各次忽略的验证输出中。

## 未满足退出条件与统筹摘要

**需要另行完成的门：** P07 新政策专用库在其归属任务下完成可审阅迁移并验证版本后，再运行本报告的托管 Chrome 脚本，实际完成双账号登录、上传/刷新、任务与历史、PNG/JSON/WAV 下载、owner/1 GB/3 天适用检查。M04 本轮不执行 DDL。Linux 自然录音、网页真实账号生产环境、全机并发/长录音资源压力、物理声卡/多屏、EXE 与部署各自仍未验证；没有要求新科学判断的差异。

**供统筹使用：** M04 A/B/C/D 各保持原报告限定 verified；M04-E Windows 本机新交互及 Qt 两自然样例为限定 verified，E 整体与 M04 完整模块仍 **in_progress**。P11 Linux 仅可继承受限合成后端服务证据。唯一当前联合阻断为专用库旧政策版本；修复范围属于 P07 政策/数据库门，M04 不改共享执行器或契约。无 DDL、push、部署、EXE、全局依赖/系统配置变更。
