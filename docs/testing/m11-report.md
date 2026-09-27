# M11 MFA 可选运行时与正式任务验收

2026-09-27。**完整 M11 为 in_progress**。Windows 本地开发态已完成真实可选环境／离线组件／统一工作台／持久任务链路；Linux MFA、实际服务器及实验室节点尚未通过。未 push、公开发布、生产部署、修改系统或全局依赖、执行现存库 DDL、修改 V2／研究语料／旧发行物，未重打 EXE。

入口：[功能映射](../modules/evidence/M11-source-map.md)、[用户说明](../manual/mfa.md)、[运行适配决策](../decisions/ADR-M11-001.md)、[远程接线要求](../specs/m11-remote-handoff.md)、[依赖许可审计](../references/m11-runtime-audit.md)。

## 正式实现

- 主进程不导入 MFA／Kaldi；页面、协议、组件检查／管理与任务编排留在主程序，重计算仅运行固定外部 child。
- `POST /api/v1/jobs/m11/create` → 原 JobStore／输入资产／租约与 fencing → `m11_executor` → Job Object → 可选 MFA → 实际 TextGrid 回读 → 原有原子 manifest 发布。
- 桌面入口 `scripts/Start-M11-Workbench.ps1`，统一 AppShell 内的 MFA 自动标注页；实际 Qt/QWebChannel 通过，未注入测试 task adapter。
- 模型、词典、参数与输入摘要进入溯源。实际内容指纹覆盖运行时源文件、原生二进制和数据文件，排除 pycache 与宿主自身的组件收据。每次执行前复核，准备阶段支持协作取消。
- 组件导入校验独立可信清单、平台／版本、逐文件 hash 和安全路径。新版本在最终目录执行实际探针后激活；失败不替换可用版本。模型／词典按 hash 独立保存，不自动下载所有语言。
- 网页任务进入原队列并明确等待已验证节点，未开放服务器回退。账号配额／期限继续使用 P07 新政策。节点公共桥未接通，未将 SyntheticFiles 当正式计算。

## 实测与证据

路径均相对仓库。结果中的时长是所测主机墙钟时间，可能有其他验证任务并行，不能外推服务器吞吐。

| 验证 | 结果／证据 |
| --- | --- |
| 旧基准 | `output/validation/m11/v2-baseline-a6c798da794c4290869e1ccafc658f21`：原管线独立运行失败于禁用 JIT 的 MFCC。保留失败，不能声称与旧 TextGrid 数值等价。 |
| 编码原样迁移 | `artifact-audit.json` 中 codec 来源／目标 SHA-256 均 `cb551438a03f9bb8a3504f57722d03830b1c4ab7d413216d432a68c769890d93`；四份继承源码与相邻 V2 对照一致。 |
| 小任务与并发隔离 | `resources-a582645f0b3c45ad99bffa3accfb98c0/repeat-1` 与 `repeat-2`：独立目录／SQLite／缓存并发执行，全部时间／标签结构相等，实际输出解析通过，组回收为 true。 |
| 100 份短文件 | `resources-7519a4fa4408496a951853bf4c7c9e09/100-file-boundary` 成功，100 份 TextGrid 全部解析。 |
| TextGrid 输入 | `resources-047984c39e4f48cabff97739cdc6559e/textgrid-transcript` 成功。初次测试只含保留 words 层失败，根因已定位并增加说明／明确拒绝，未改用户层名。 |
| 119.8 秒整段转写 | 上述 resources 下 `long-unsegmented-119.8s`，默认参数无法得到对齐，显式 `m11_no_alignments`。这是支持边界的失败证据，非成功验收。 |
| OOV／词典错误 | `resources-047984.../oov` 为 m11_oov_words；`resources-6fc5e62d402f4251b2c9dbbb250108b5/dictionary-mismatch` 为 m11_model_mismatch。先前 MFA 将词典错误当 OOV 的行为已识别，显式检测 excluded_phones，未默默忽略发音。 |
| 损坏模型／崩溃／取消／超时 | `resources-b2b63fd992214de398a2b8fb783ca5fb/report.json` 通过。真实外部进程；崩溃为仅终止本任务主进程的受控故障，最终整组回收均 true；模型错误码已修正。 |
| 正式本地宿主 | `wiring-9fd09b905ee94a0898b100c956fc58b6/report.json`、`wiring-e534ae9509814f67b461f88c3f4d1877/report.json`：真实 HTTP／SQLite／worker、幂等、结果字节/hash、取消、重启保留完整 manifest。前次取消有真实 MFA PID 与组回收记录；末次覆盖准备阶段取消。 |
| 离线包 | `component-a4159343dbc44af180e2f2e361666ef9` 生成真实 ZIP／逐文件清单。过长目录首次 DLL 加载失败、未激活。修正短版本目录／路径预检后 `output/m11c-028b881d/import-report.json` 成功，321.578 秒含解包、两次环境内容核对与完整词典探针。 |
| 实际 Qt 工作台 | `qt-642b638d008c497abb576e5db63dcbe7/qt-report.json` 成功：中文／空格／子目录输入、Beam 联动、真实任务、两个 TextGrid＋溯源保存、浅／深色与 900×700、关闭保护。已查看 light／compact 截图。 |
| 实际 Qt 日志／历史 | 上述 Qt 目录 `logs-9a7b099b/report.json` 及最终构建 `logs-b34b91ca/report.json` 成功，重启后真实历史、原生日志读取、宿主路径脱敏。 |
| 真实账号／PG 网页 API | `backend/tests/test_m11_web.py` 使用新建且独占的测试 PG 集群，不触及既有库。实际认证、双账号 404 隔离、配额字节、三日期限、原队列等待、普通 worker 无法领取、重开与取消通过。运行时登记是明确的路由元数据 fixture，没有执行远程 MFA。 |

上述 `resources-*`、`wiring-*`、`qt-*` 前缀路径均位于 `output/validation/m11/`。若某次资源脚本因后续断言失败退出，表内仅引用其已完成且有独立 report 的场景，不把整个失败脚本标绿。保留全部早期失败诊断。

## 资源与体积

| 场景 | 外部任务耗时秒 | 进程组峰值 B | 临时目录采样峰值 B |
| --- | ---: | ---: | ---: |
| 小词典、2.8 秒合成输入，两个并发任务各自 | 39.172 / 38.891 | 541,126,656 / 540,508,160 | 308,464,687 / 308,464,743 |
| 小词典、100 份短文件 | 40.140 | 543,076,352 | 319,672,692 |
| TextGrid 转写输入 | 38.500 | 539,656,192 | 308,465,524 |
| 119.8 秒无分段，失败 | 43.703 | 635,133,952 | 312,677,611 |
| 完整普通话词典、离线组件探针 | 96.188 | 1,758,633,984 | 438,739,426 |
| 完整词典、Qt 正式任务，两份公开输入 | 97.906 | 1,756,790,784 | 438,892,644 |

Windows 配置：Job Object 进程组硬内存预算 2,147,483,648 B，MFA 执行期限 180 秒；临时目录约每 0.1 秒轮询 512,000,000 B **软监测上限**。采样可能漏过短时峰值，不是硬磁盘配额。环境哈希／素材准备另有时间，表中外部任务时间不包含全部安装或准备时间。首次组件检查可能需数分钟，专属桥预算 900/930 秒。

全词典峰值已超过服务器初始约 1 GiB 计算预算。服务器继续禁用；没有降低 Beam、截断语料或切换模型来强行执行。100×120 秒、自然录音准确率、更多语言／模型、其他采样率的边界尚未充分测量。

实际候选 ZIP **809,661,654 B**，展开文件 **2,698,777,844 B**；原环境含 pycache 共 **2,897,888,120 B**。模型 **92,275,957 B**、词典 **8,709,372 B**，分别管理。安装还需自检临时空间、下载缓存及保留旧版本／失败目录的额外空间，不将展开量当总磁盘峰值。

主程序本轮独占 Python 源文件约 59 KB，MFA 前端独立 JS/CSS 构建块约 18 KB，详见 `artifact-audit.json`。这不是 EXE 二进制增量。`scripts/m11_bundle.py` 给现有研究／M12 构建配方提供单一轻量 child 数据文件和 MFA／kalpy 显式排除，未收集可选环境目录，未执行构建 EXE。冻结启动仍待用户未来要求后验收。

## 平台状态

| 平台／范围 | 当前状态 |
| --- | --- |
| Windows 本地源码宿主、正式 Qt 任务、已有 3.3.8 环境 | verified，限定上述公开输入与边界 |
| Windows 可选 ZIP 离线导入 | verified，限定短目录候选及实际探针，未声明通用再分发／所有命令行入口可重定位 |
| 网页认证／PG 入队与资源政策 | verified，限定本机新建测试集群与 ASGI，计算保持 waiting |
| WSL NInfer | 可移植单元测试通过；Linux MFA 环境未建立，systemd/cgroup 可委托执行条件未满足，blocked |
| 实际阿里云服务器 | 未连接／未部署／未测试 MFA；fallback disabled |
| 实验室 Linux 节点 | 未提供已验证环境与正式连接，blocked |
| 公开组件下载／主 EXE | 组件 URL 待发布；完整许可审核与冻结运行待测，本轮未发行 |

## 命令与检查

本轮按阶段执行，未重跑无关科学模块。新 PG 仅对测试新建库应用既有 migration 文件，不修改任何既有数据库。

```powershell
$env:PYTHONPATH='backend/src;packages/phonetic_core/src;desktop/src'
& .venv/v3-dev/Scripts/python.exe -B -m pytest -c tests/pytest.ini backend/tests/test_m11.py tests/parity/test_mfa.py backend/tests/test_job_policy.py backend/tests/test_storage_policy.py backend/tests/test_p07_policy_wiring.py tests/contracts -q
$env:PTB_POLICY_FRESH_PG='1'
& .venv/v3-dev/Scripts/python.exe -B -m pytest -c tests/pytest.ini backend/tests/test_m11_web.py -q
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run contracts:check
npm --prefix frontend run build
```

后端／共享契约定向检查 **147 passed**；真实 PG 单项综合测试 **1 passed**。前端最终 **172 passed**，见 `output/validation/m11/frontend-final.log`。类型、生成 TS 检查、生产前端构建通过。构建保留其他模块的既有大块 warning；Qt offscreen 有 GPU fallback 日志，页面实际渲染、操作与截图通过。WSL 最终 **27 passed**，见 `output/validation/m11/wsl-final.xml`；Windows 定向结果为 `windows-final.xml`。未据此宣称 Linux 原生执行通过。

`verify_m11_runtime.py`、`verify_m11_alignment.py`、`verify_m11_resources.py`、`verify_m11_component.py`、`verify_m11_import.py`、`verify_m11_wiring.py`、`verify_m11_qt.py`、`verify_m11_qt_logs.py` 提供独立可复现入口，各自必须显式传入环境／模型／清单或本轮测试路径，帮助可用 `--help` 查看。

## 剩余阻断

完整 M11 尚不能收为 verified：正式 remote/1 文件与执行桥、Linux 环境和 P11 委托／配额门、实际服务器与实验室节点、自然语料准确率与更广支持边界、公开下载目录／许可发行链、冻结 EXE 均需后续独立证据。接线需求已具体写入交接文档，未新建队列或假装远程已接通。共享文件已向并行 M05 释放，未混入提交其改动，本轮没有 Git commit／push。

最终全局检查未称全绿：`generate_contracts.py --check` 为 schema_drift=[]，TS snapshot 一致；`check_architecture.py` 当时报告并行 M05 的 10 个新增资源未登记，已通知 M05 归属方；`validate_docs.py` 唯一当前缺链为既有 README 的 M10-R5 EXE，M11 文档无新增缺链。没有修改这些非 M11 归属来掩盖检查结果。
