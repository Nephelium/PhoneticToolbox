# M07 实施与验收报告

2026-09-27。**Windows开发态闭环 verified（范围如下），完整跨平台M07仍 in_progress。** Linux同环境核心对照通过，Windows/Linux逐位基准未通过，Linux正式服务、可信远程节点和小服务器回退保持关闭。未打包EXE、push、公开发行、生产部署、执行现存库DDL或改V2。

## 入口和功能

运行 `scripts/Start-M07-Workbench.ps1`，选择合成与操控 → 发声类型合成。使用现有 `.venv/m09-ui`、统一Qt宿主和构建后的前端。详见 [使用说明](../manual/phonation-synthesis.md)、[功能与来源映射](../modules/evidence/M07-source-map.md)。

已实现源/目标授权输入与波形/试听、Parselmouth与固定原生REAPER、完整高级参数、两种控制点对齐、显式F0应用/完整CSV、三类型双方向、当前/六组生成、独立幅度策略、逐步/拼接WAV、完整参数清单、桌面保护保存和网页ZIP下载。正式历史/重试、取消保留完整组、配额/账号隔离、失效提示、迟到隔离、草稿与关闭保护接公共能力。

每组独立正式任务，批次ID/计划组数/序号关联六组。组内完整发布，组间允许部分完成；取消后未提交的组不伪装成功。方案见 [ADR-M07-001](../decisions/ADR-M07-001.md)。未增加父子任务或DDL。

## 实现文件与边界

- 纯核心：`packages/phonetic_core/src/phonetic_core/manipulation/m07_models.py`、`m07_legacy.py`、`m07_grid.py`、`m07_api.py`。保留原计算，外围有限值/尺寸拒绝独立。
- 合约与执行：`backend/src/ptb_api/m07_models.py`，`ptb_worker/m07_task.py`、`m07_executor.py`、`m07_child.py`、`m07_science.py`、`m07_errors.py`。
- UI与适配：`frontend/src/modules/phonation-synthesis/`、`platform/m07.ts`、`m07Archive.ts`、`desktop/src/ptb_desktop/m07_bridge.py`。
- 公共最小接线：jobs/main/JobView及manifest union、store.retry、executor/process_entry、结果与snapshot文件角色、公共policy新计算结果类型、TaskBridge、desktop/server port、AppShell。按现有生成工具更新OpenAPI/TS与来源数据。
- 验证：`tests/support/m07_*`、`tests/parity/test_phonation_synthesis.py`、`backend/tests/test_m07*.py`、`frontend/tests/m07.test.ts`、`tests/e2e/m07-*.cjs`、`scripts/verify_m07_*.py`。

开始时分支 `codex/v3-rebuild` 已有其他模块大量改动，未回滚或提交混杂成果。共享总进度交接使用 [可合并摘要](m07-coordination-summary.md)，不覆盖其他agent正在编辑的ledger。

## 科学证据

[A基线](m07-baseline-report.md)：原实现独立双轮208数组、2,279,208数值精确一致。三类型×两方向×幅度四组合，默认及非默认分析参数、两对齐、尾帧、中间量、WAV字节均捕获。

[B核心](m07-core-report.md)：Windows源码44项通过，独立安装wheel42项既有科学测试通过。Parselmouth和原生REAPER分别有原实现对照。REAPER补充二输入/六组精确相等，无Python静默回退。继承五源文件hash与相邻V2保持一致。

行为差异明确登记：显式反向当前生成、显式提取/应用、有限值拒绝、固定后端、资源上限、事务发布及保护保存。没有将原peak/mean-absolute幅度策略改成RMS或知觉响度；没有把LPC残差方法换成其他合成器。论文方法、Python改编、v3迁移分别记录。

## 实际验证

统一Python环境：Python3.11.14、NumPy2.2.6、SciPy1.16.3、Parselmouth0.4.7。PowerShell中先设置绝对PYTHONPATH到backend/src、packages/phonetic_core/src、desktop/src。

| 命令/工具 | 结果与证据 |
|---|---|
| `.venv/m09-ui/Scripts/python.exe -X utf8 tests/support/m07_baseline.py` | 原双轮一致；fixtures及baseline/round1、round2 |
| `python -m pytest -o addopts='' tests/parity/test_phonation_synthesis.py -q` | 44通过；`output/validation/m07/windows-core.xml` |
| `python -m pytest -o addopts='' backend/tests/test_m07.py -q` | 8通过；`windows-tasks.xml`；真实进程分析/apply/六组、REAPER缺失/成功、取消、超时、工作进程终止、重试、幂等 |
| `$env:PTB_POLICY_FRESH_PG='1'`后运行`backend/tests/test_m07_pg.py` | 2通过，最终独立复验见 `pg-final.xml`；新独立PG，双账号输入/成果、1GB配额拒绝、3天结果、不续输入/下载、到期、旧generation拒绝 |
| `python scripts/verify_m07_wiring.py` | 正式本地HTTP8检查通过；`host/092007c72cec44589e55359ac621e59e/report.json` |
| `python scripts/verify_m07_native.py` | 原生二输入/六组逐位一致；`native/report.json` |
| `node tests/e2e/m07-host.cjs` | 9组Chrome闭环，真实FileProvider/TaskBridge/LocalService，仅QWebChannel传输由测试页代替；`host/cc9f109ef6a84e1b88b201f053ec76b9/browser-report.json`，包含保护保存和未应用草稿恢复 |
| `python scripts/verify_m07_qt.py` | 实际Qt/QWebChannel、生成前端、分析/编辑/生成及四WAV完整保存通过；`host/33cc747e9d6e4f97a0908a3dc6394c85/qt-report.json` |
| `python scripts/verify_m07_web.py` | 实际Chrome/账号/PG/worker上传到ZIP下载成功，ZIP逐项SHA回读；另一账号任务及成果均404；`web/027a5579bd8f4d6383b257c2771f073d/web-report.json` |
| `npm --prefix frontend run typecheck`、`test`、`build` | 类型通过、全局178项通过、生产构建通过；大chunk警告来自已有其他模块，未隐藏 |
| `.venv/v3-dev/Scripts/python.exe scripts/generate_contracts.py`、`npm --prefix frontend run contracts` | 自动生成，无手改接口快照 |
| `pytest -c tests/pytest.ini tests/architecture/test_boundaries.py tests/contracts/test_m01_contract.py -q` 与 `scripts/check_architecture.py` | 72通过，全项目架构errors为空；`shared-regression.xml` |
| `uv build --offline`、`uv pip install --offline --no-deps --target ...` | 项目内临时target安装与科学测试通过，未改现有环境 |

测试全部采用公开合成输入。故障/到期采用明确注入：0.001秒超时、终止回调提供的本任务PID、测试库调整到期时间。不能冒充自然三天经过或生产事故。PG集群新建、仅监听本机、任务结束停止，未读取用户DSN或修改现存库。Qt使用offscreen，GPU上下文降级日志保留，基础实际链路通过，不代表声卡听感或全部DPI设备验收。

修复中保留的失败：TaskBridge原结果白名单漏M07、源/目标共用读取代际导致相互废弃、组保存目录保护函数错误导入、F0图受公共svg图标高度影响、测试读块超过公共上限。均定位修复，未放宽数值比较；旧失败截图/报告保留，不当成功证据。

## 资源

Windows真实受控进程组，硬限1,000,000,000字节和120秒，任务期限300秒；输入每份10秒/480000帧/8MB，输出包64MB，受管临时+发布预算128MB。没有静默调小参数。

`verify_m07_resources.py`共两轮，最新证据 `output/validation/m07/resources/679f6019370a4c83b50e7f88356359ae/report.json`。数值为单机测量，非稳定性能保证；MB按十进制。

| 场景 | 进程峰值 | 子进程耗时（含导入） | 全任务耗时 | 临时文件峰值采样 |
|---|---:|---:|---:|---:|
| 典型双输入分析 | 198.2MB | 4.375s | 5.015s | 0.028MB |
| 六组各9步，逐组执行 | 83.0–84.1MB/组 | 1.265–1.344s/组 | 2.953–3.438s/组 | 0.584MB |
| 双10秒/480000帧分析 | 259.5MB | 19.906s | 23.204s | 1.922MB |
| 10秒50步单组 | 314.0MB | 35.063s | 51.468s | 14.370MB |

典型调度/输入准备到子进程创建0.203秒，生成0.484–0.813秒；最大snapshot准备9.140秒。该时间不等于纯Python导入时间，未宣称冷启动已单独隔离测量。临时空间每20ms采样，可能漏极短峰值；另有硬预留上限。50步实际成果22,860,870字节。九任务均记录进程组cleaned=true，取消/超时/强制故障也有回收检查；无按名称批量杀进程。

## 分平台状态与停止点

| 范围 | 状态 |
|---|---|
| Windows核心/正式本地任务 | verified，限定上述环境、边界和合成材料 |
| Windows Chrome桌面适配/实际Qt | verified，限定实测功能；硬件听感、多屏DPI未验 |
| Windows认证网页与新PG | verified，限定本机测试服务；非公网生产部署 |
| WSL Linux核心 | 同平台原/新逐位一致；直接Windows基准14通过28失败 |
| Linux正式服务/原生REAPER/进程组资源准入 | 未验证，关闭 |
| remote/1科学桥、实验室节点、实际服务器回退 | 阻断，未开放或部署，见[具体交接门](../specs/m07-runtime-handoff.md) |
| 自然语料广泛覆盖、最大参数组合完整矩阵、知觉效度 | 未验证，不能由本轮合成样例推断 |

旧源码引用/许可未闭合部分保持review-required，未加入论文数据。真实Linux与远程门依赖公共桥和运行时receipt，已交付具体可审阅方案，不绕过保护。M07本轮工作在此停止，不自动推进其他模块。
