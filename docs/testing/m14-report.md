# M14 音系归纳验收报告

2026-09-27。**模块整体 `in_progress`**。五组原功能已实现并在限定 Windows 正式工作台链路验收；Linux 核心/导出及受限 child 已验，Linux 正式任务与原生浏览器、托管 PG 联合门尚未完成。只使用公开合成字表，不扩大为全平台 verified。

## 交付入口

- 正式开发入口：[Start-M14-Workbench.ps1](../../scripts/Start-M14-Workbench.ps1)，统一工作台侧栏“音系归纳”。依赖项目内 `.venv/m14` 与已构建 `frontend/dist`，不启动模块独立产品宿主。
- [操作说明](../manual/phonology-induction.md)、[完整说明书/源码映射](../modules/evidence/M14-source-map.md)、[实施记录](../plans/modules/M14-implementation.md)。
- 核心：`packages/phonetic_core/src/phonetic_core/transcription/phonology/`。解码、持久任务、固定 child 与资源/存储适配：`backend/src/ptb_worker/m14_*.py`，契约 `backend/src/ptb_api/m14_models.py`。
- 页面与状态：`frontend/src/modules/phonology-induction/`。正式双端 port：`frontend/src/platform/m14.ts`，桌面授权目录交付：`desktop/src/ptb_desktop/m14_bridge.py`。

## 功能结论

| 功能组 | 已实现与通过证据 | 状态边界 |
| --- | --- | --- |
| F01 导入/策略 | XLSX/XLS/CSV/TXT/TSV；跳首行开/关；零声母/空韵两策略；无表头、缺失、重复、损坏、空数据、UTF-8/全角调值 | Windows 正式 API/持久任务/Chrome；三平台核心对照 |
| F02 调类/调值 | 拖动交换、调类名称、同名映射、确认、取消 | Windows Chrome 真实拖动与对话框；Qt 实际修改调类 |
| F03 声韵排序 | 两列表 Ctrl/Shift 多选及整组拖动；默认顺序保留 | 状态测试和 Windows Chrome |
| F04 声韵归并 | 单源→目标、链式归并、空韵源/目标、取消源选择/整个弹窗、确认、映射与原记录审阅 | 归并前后均 16 条，字音备注及组内原顺序不变；无原输入改写 |
| F05 三件交付 | 两种层次 DOCX、同音字矩阵 XLSX，完整发布、目录保存、逐件下载、帮助与关闭保护 | Windows 原生后台、正式桌面 adapter、Chrome、实际 Qt；结构与 Office 视觉检查 |

## 独立 V2 基准与差异

`tests/support/m14_baseline.py` 在 V2 解释器调用 V2 原源码，公开输入来自 `scripts/prepare_m14.py`。两轮捕获共 16 个文件/跳行案例（含错误样例），正常五格式×两跳行×两策略为 20 组。逐条记录、全部分析字段、链式 aliases、两份 Word 的段落/表格文本、矩阵全部单元格及六份源码/手册哈希相等。V3 未生成自己的 expected。

V2 原环境缺 xlrd。XLS 扩展基准临时引用已授权 M14 依赖目录，未修改 V2 环境。证据：`output/validation/m14/v2-original`、`v2-xls-overlay`、`v2-repeat`，固定 expected 为 `tests/fixtures/m14/v2-baseline.json`。机械迁移本地提交 `ad165a6`，后续行为修正独立在呈现/配置/适配层，详见映射文档。

明确保留 V2 的整词备注、NA 识别差异、文本空字段补位、重复条目、无调值记 0、未知 IPA 不自动纠错。明确修正：XLSX 调类顺序、空韵归并、重名/部分保存、长格行高、Office 富文本空白标记。没有修改解析算法或凭模块名称重新设计。

## 已执行验收

| 平台/入口 | 结果 | 原始证据 |
| --- | --- | --- |
| Windows CPython 3.11，core/adapter/原生保存故障回滚 | 38 passed，4.89 s | 下方 pytest 命令；`backend/tests/test_m14.py`、`desktop/tests/test_m14_save.py` |
| WSL NInfer 原生 Python | 37 passed，4.61 s | 同一核心/adapter 测试；无 systemd，不冒充进程组门 |
| Ubuntu 实际服务器，512 MB systemd 组 | 37 passed，10.74 s；组总峰值 107,036,672 bytes，cleaned=true | `output/validation/m14/server-evidence/evidence/linux-1790477816493908773/{tests.txt,process.json}` |
| Ubuntu 固定 M14 child 独立资格探针 | 三件 bundle/input/output SHA 回读通过；峰值 69,476,352 bytes，执行约 1.872 s、等待约 1.020 s，cleaned=true | 同目录 `child-process.json` 与三份文件；这是模块资格探针，未绕过正式 task capability |
| Windows 正式本地 API→durable worker→受限 child→存储 | 五格式预览、幂等、完整三件、损坏失败恢复、持久取消通过 | `output/validation/m14/wiring/a572691e267149c8a75b1a58809f28a7/report.json`；末轮同脚本复验 |
| Windows Chrome 正式 AppShell/desktop port | 7 组流程通过，包括五格式、两策略、Ctrl/Shift/拖动、确认取消、归并、草稿失败、关闭恢复、下载、重名 | `output/validation/m14/wiring/8158466bf4464a3f90543d70a296981b/browser-report.json`，light/dark/compact/selection.png；末轮追加韵母拖动复验 |
| Windows 实际 Qt/QWebChannel + 生产前端 | 导入、调类编辑、持久生成、原生目录写入、重名恢复通过 | `output/validation/m14/wiring/3d940b632d794f5c87f472e8eb739771/qt-report.json` 与 qt.png |
| Windows Chrome→Linux 统一 API | SSH 本地隧道访问真实 Linux 页面、health、capabilities；M14 正确不开放 | `output/validation/m14/windows-browser-linux.json`、同名 png；无 M14 Linux 成功任务声称 |
| Linux 正式 API 资格拒绝 | 有效模型请求返回 503/m14_runtime_unavailable，capabilities 不含 M14 | `server-evidence/evidence/linux-1790477888421976427/service-report.json` |
| Linux 原生 Chromium | 已按授权在项目专用目录放置，因 9 个系统库缺失无法启动 | `server-evidence/evidence/browser-environment.json`；未安装系统依赖 |
| 前端检查 | 140 tests passed；typecheck、build、契约生成通过 | npm 命令；M14 状态测试含 3 项，计在 140 中 |

Qt 使用 offscreen 实际宿主，原生目录选择器返回值由测试指定到自有目录，输入 File 为真实合成 XLSX 字节。没有把浏览器 QWebChannel 传输替身单独称为 Qt 验收。Qt 无 GPU 环境打印 GLES 上下文警告，实际页面和文件功能通过，不据此证明显卡/多屏设备行为。构建的既有大 chunk 提示仍保留。

末轮正式 API/worker 复验为 `output/validation/m14/wiring/2064c8222556438792246661e912e3ce/report.json`，8 项通过。末轮 Chrome 为 `output/validation/m14/wiring/a4c5a809de3b48b68915acf272e58592/browser-report.json`，7 组通过，并覆盖声母与韵母两列表的整组拖动；`selection.png` 已人工视觉回读，选中状态、中文和组合 IPA 可见。远程证据绑定 `output/validation/m14/server-stage2-manifest.json` 快照，之后本机的小边界修订（有效行数限制、超时错误映射及完整结果 manifest 校验）没有重新传输；Linux 正式准入仍需绑定最终源码的 receipt。

收口文档检查扫描 798 个文件，M14 链接无错误；全仓门仍因 README 中既有 M10-R5 EXE 链接缺失而失败，另保留历史快照链接清单，未跨模块修订。完整输出为 `output/validation/m14/docs-validation.txt`。V2 六份源码/手册最终 SHA-256 全部与独立基准一致，见 `output/validation/m14/final-v2-preservation.json`。

## 文件内容与视觉证据

实际 Chrome 导出并保存的归并后文件已回读。声母 `m` / 韵母 `a` 单元格严格为：`[214]马1 [合调]妈麻文巴鼻鼻化妈 [51]怕`。两份 DOCX 相应段落都为同一调类顺序，保留 `妈 麻文 巴 鼻鼻化 妈` 的原记录顺序。重复“妈”未丢失；下载的正序 DOCX 与目录保存字节一致。见对应 wiring 目录 `content-report.json`。

DOCX/XLSX 均检查真实内容和字体属性。实际 Word 16、Excel 16 只读打开，并分别导出 PDF、用 PDFium 渲染 PNG。检查了原表与归并后代表页面，中文、组合 IPA、备注下标、统计、分组和矩阵方向可读，无代表样例的遮挡/缺格。`output/validation/m14/visual`、`visual-edited` 保存两份各 2 页的 Word PDF 和 1 页矩阵 PDF/PNG。Excel 打开真实发现并修复 V2 富文本空格缺 `xml:space` 导致拒绝打开的问题，未用 openpyxl 自回读冒充 Office 兼容。字体为名称快照，未嵌入 Office 文件。

## 规模、执行与错误恢复

- 导入最大 2,000,000 bytes、10,000 个有效记录；最多 64 列、单字段 512 字符；XLSX 展开最多 32 MB/512 entries。原始输入不变。
- 导出矩阵最多 20,000 格、单格最多 32,767 字符、三件总计最多 16 MB。超预算拒绝完整任务，不截断/降采样/丢记录。
- 默认 Windows 每组 512,000,000 bytes、60 s，使用现有 P11 `collect_pipe`/Job Object。公开同音字高密度表 100 条约 1.140 s/76,189,696 bytes；1,000 条约 7.638 s/83,660,800 bytes。10,000 条附备注样例命中单格格式上限，约 1.120 s 拒绝。见 `output/validation/m14/resources/c721e5319d7a4a72938ba855468892f0/report.json`。这是短样例，不保证所有上限组合均能在 60 s 内完成；超时明确终止并可重试/分表。
- 重型读取/导出在受限 child 中。正式接线首轮发现文档库在 worker 心跳前加载造成租约失效，已把常量移至轻量 models，增加禁止编排导入 openpyxl/pandas/docx 的回归，后续正式链路通过。
- 空文件、缺列、无有效行、损坏、公式、错误编码分别拒绝。跳过/重复诊断可审阅。失败/取消保留旧编辑与已有结果；成功导入才替换工作数据。
- 输出发布复用 files.output/write/seal/complete。原生保存先验证整组三件，同名拒绝。注入第二次 rename 失败后本次新文件全部回滚，无关哨兵文件不变，之后重试成功。草稿存储失败保留页面。输入/结果失效依现有平台错误处理，不另设 TTL。

## 依赖、许可与并行接线

井井明确批准项目内 docx/xlrd 及必要依赖、测试 xlwt。Windows `.venv/m14`、WSL `/home/ninfer/ptb-m14-20260927`、服务器 `/home/admin/ptb-m14-20260927/venv` 各自隔离；既有 API/Qt 环境只读复用。`requirements-m14.in`、`requirements-m14-additions.lock` 与 `output/validation/m14/dependency-audit.json` 记录版本、包哈希和许可文件。python-docx 1.2.0 MIT，xlrd 2.0.2 BSD，lxml 6.1.3 BSD-3-Clause，typing-extensions 4.16.0 PSF-2.0；xlwt 1.3.0 BSD 仅生成公开 XLS。没有安装全局依赖。

M08 释放共享宿主文件后才接线。P11-PERF chat `01a0e0a0-8061-7410-80d4-82aab8fbde11` 明确释放 executor.py/process_entry.py/main.py 后，只增加 M14 固定分派/Windows 白名单/能力条目。没有修改其 native 资源/进程实现；Linux entry 留待其后续接入。共享 AppShell、平台 port、task bridge、API/models、manifest 和生成契约的改动与 M08 既有成果共存，不覆盖/提交其他模块成果。

服务器连接经用户追加授权；认证不写项目、报告或脚本。传输仅包含代码、依赖和公开合成测试材料。临时统一服务仅绑定 loopback，180 秒退出，SSH 隧道已关闭。M14 自己的 cgroup 均 cleaned=true；观察到其他任务的活动 unit，未操作它。不宣称整机零任务。

## 尚未完成的明确接口/证据

1. **Linux 正式 M14 任务**：需 P11 登记 `native.linux_runtime.ENTRIES['m14']='ptb_worker.m14_child'`，确定正式收集调用、生成当前源码/环境 hash 的 runtime profile 与验证 receipt，再验实际提交/取消/故障/下载。M14 当前 Windows handler 主动拒绝 Linux；不能仅删除平台判断就宣布开放。
2. **Linux 原生浏览器**：缺 `libatk-1.0.so.0`、`libatk-bridge-2.0.so.0`、`libXcomposite.so.1`、`libXdamage.so.1`、`libXfixes.so.3`、`libXrandr.so.2`、`libgbm.so.1`、`libasound.so.2`、`libatspi.so.0`。本轮已约定不装系统依赖，浏览器没有运行成功。
3. **托管网页存储联合门**：实现复用现有 owner/project、上传/结果 writer、额度与到期规则，没有另造政策。P07 新规则现存 PG 迁移与本轮真实账号/配额/到期联合证据未完成；未执行 DDL，因此不标这部分 verified，也不把 1 GB/3 天视作已在库生效。
4. 未做生产部署、EXE、多屏/人工目录选择器、长时负载与非合成研究材料验证。来源仍 `PENDING-PHONOLOGY`，不把功能迁移当更早许可链闭合。

## 可复现命令

在项目根目录，项目环境存在时：

```powershell
$env:PYTHONPATH="$PWD/packages/phonetic_core/src;$PWD/backend/src;$PWD/desktop/src"
$env:PYTHONIOENCODING='utf-8'
.venv/m14/Scripts/python.exe -m pytest -c tests/pytest.ini tests/parity/test_phonology_induction.py backend/tests/test_m14.py desktop/tests/test_m14_save.py -q --tb=short
.venv/m14/Scripts/python.exe scripts/verify_m14_wiring.py
.venv/m14/Scripts/python.exe scripts/verify_m14_qt.py
.venv/m14/Scripts/python.exe scripts/verify_m14_resources.py
node tests/e2e/m14-host.cjs
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run build
.venv/m14/Scripts/python.exe scripts/generate_contracts.py
npm --prefix frontend run contracts:check
```

Linux 原生资格探针为 `scripts/verify_m14_linux.py`，只在已授权项目测试目录及当前 hash 快照运行。`--serve` 仅验证正式 API 关闭门并提供短时统一前端，不开放业务能力。Office 视觉入口为 `scripts/verify_m14_office.ps1 -EvidenceFolder output/validation/m14/visual-edited`，只读取本任务合成三件文件。没有 push、部署生产、打包 EXE、现存 DDL、改凭据/CI/系统设置、修改 V2/用户语料。
