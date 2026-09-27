# P11-LINUX 音频任务验收

2026-09-26。对应[实施计划](../plans/2026-09-26-p11-task-flow.md)和[模块接入接口](../specs/p11-linux-module-interface.md)。历史环境安装、数值精确失败与性能调查均保留，本轮不重复开展环境/BLAS 调查。

## 范围与状态

本轮执行顺序为 LPC → EGG → 声学参数提取。使用实际阿里云 Linux、现有任务解释器和独立 backend/render overlay；上传、任务持久化、计算、文件生成和下载走真实回环 HTTP。本地 token 身份、SQLite 既有合成库副本，未执行 DDL。最终逐项结果与 runtime/hash 在本报告后续收口段登记。

**最终状态：上述三模块的 Linux 受限后端任务链路 `verified`，限定本报告公开合成输入和本地 token/SQLite/回环 HTTP 范围。P11 的生产平台、网页账号及资源全局准入仍单列未完成。**

这不代表真实 PostgreSQL 网页双账号、浏览器交互、生产部署、全机并发准入或所有自然语料已验收。Windows 的原精确回归保持独立。

## 文件归属 / changed_files

修改已有共享后端：

- `backend/src/ptb_api/main.py`：沿用上一轮改动，按实际 runtime/验证凭据返回 Linux 各模块 capability。
- `backend/src/ptb_worker/acoustic_executor.py`：固定科学入口、Linux/Windows 调度分支、资源证据回传，原结果发布链路保留。
- `backend/src/ptb_worker/segmentation.py`：复用固定入口；Linux 分段 capability 未自动开放。
- `backend/src/ptb_worker/egg_runtime.py`：Linux 独立 fingerprint；Windows MKL 判据不变。LPC 原 runtime 复用此 fingerprint，无须修改 `lpc_runtime.py`。
- `backend/src/ptb_worker/font_preflight.py`、`spectrogram_preview.py`：Linux 受限字体预检和公共语谱预览。
- `backend/src/ptb_worker/local_acoustic_files.py`：Linux flock，保留 Windows msvcrt 锁。
- `backend/src/ptb_worker/native/reaper.py`、`science_child.py`、`acoustic_errors.py`：原 Windows REAPER 路径保留，Linux 固定 hash 原生输出，Linux 原生失败不发布缺失 rF0 的结果。
- 上一轮新增 `native/posix.py`、`native/capabilities.py`：补真实 MainPID 回调、按 runtime/report hash 的分模块能力门。

本轮新增：`native/linux_runtime.py`、`native/linux_reaper.py`、`linux_bootstrap.py`、`backend/tests/test_p11_task_runtime.py`、`tests/support/practical_equivalence.py`、`tests/parity/test_p11_practical_equivalence.py`、`scripts/verify_p11_task_flow.py`、`scripts/compare_p11_acoustic.py`、本报告/专项计划/模块接口文档。环境门及旧 Linux 报告仅追加当前入口。

根 README、总台账、全局 ADR、生成契约、core 科学算法、M08/M13 模块、公共 UI、desktop、V2 和用户数据不属于本轮修改。其他 agent 同时工作，保护核对按本轮认领文件与其他作者改动分别报告，不把所有工作树变化归为 P11。

## 科学检查与解释边界

Windows 原 42 项 EGG/LPC 精确回归和历史 Linux 19 项失败保留。新增 practical/1 逐项检查 shape/dtype、NaN/±Inf、事件数量、顺序、周期配对，GCI/GOI/peak 各自最多 1/fs。CQ/SQ 用一采样点事件区间的分式极值推导，不设置统一经验容差。GCI-F0 用周期两端各一采样点推导频率区间。

ROI 的 50 ms 事件路径与 100 ms CQ/SQ 路径分开验证。前者对照自身冻结事件；后者原 fixture 没有内部事件数组，捕获当前实际传入指标函数的事件，检查指标公式、冻结 mask/周期数和反向一采样点包络。此范围差异不隐藏。连续值预算见专项计划；反例覆盖事件增删、重复、NaN 和超限，以及事件合法但指标被篡改。

M01 按实际 76 个数值字段的单位比较，时轴/配置/列顺序/NaN/Inf 必须一致。rF0 为 0.01 Hz，其他 Hz 为 1e-4 Hz，dB/dB 每 decade 为 1e-4，percent 为 1e-5，比例/归一化差为 1e-6。这些是在本轮比较前按报告分辨率选择的工程预算，未由观测最大误差倒推，也不作感知等价或自然语料普遍性的结论。无输入元数据的 Lip 单位不虚构容差。

## 实际执行命令

Windows PowerShell，`PYTHONPATH=<v3>/backend/src`：

```powershell
& scripts/Invoke-M03-Python.ps1 -PythonArguments @('-m','pytest','-c','tests/pytest.ini','tests/parity/test_egg_analysis.py','tests/parity/test_lpc_spectrum.py','backend/tests/test_m04_exports.py','backend/tests/test_m03_exports.py','-q','-p','no:cacheprovider')
& .venv/m09-ui/Scripts/python.exe -m pytest -c tests/pytest.ini backend/tests/test_p11_task_runtime.py backend/tests/test_p11_capabilities.py tests/parity/test_p11_practical_equivalence.py -q -p no:cacheprovider
& .venv/m09-ui/Scripts/python.exe scripts/verify_p11_task_flow.py --template output/validation/p11-tasks-20260926/synthetic-template.sqlite3 --output output/validation/p11-tasks-20260926/windows-acoustic-run3 --phase acoustic --reaper phonetic_toolbox/core/acoustic/reaper.exe
```

科学原回归+导出 82 passed，既有重复时间 warnings 2 条。新增实用/配置检查最终 48 passed，FastAPI/Starlette 既有弃用 warnings 2 条。另 84 项共享后端/分段/REAPER/语谱/字体/协议/任务回归通过，完整命令与 XML 在本证据目录。Windows M01 实际 HTTP 两个完整原生任务成功。

Linux 任务目录 `/home/admin/ptb-p11-20260926/tasks-followup`，`P=../venv/bin/python`。原 `venv`/`numeric-followup`/`speed-followup` 保留。`PYTHONPATH` 只在当前测试进程指定 backend/render/pandas overlay，不修改 shell profile 或系统配置。

```sh
"$P" -m pip install --no-deps --require-hashes --target render-overlay -r render.lock
"$P" -m pip wheel --no-deps --no-build-isolation ./backend-p11 -w wheels-stage4
"$P" -m pip install --no-deps --target backend-stage4 wheels-stage4/ptb_api-3.0.0a1-py3-none-any.whl
"$P" build_runtime.py backend-stage4 runtime-stage4.json
"$P" scripts/verify_p11_task_flow.py --template synthetic-template.sqlite3 --output lpc-verified --phase lpc --faults
"$P" scripts/verify_p11_task_flow.py --template synthetic-template.sqlite3 --output egg-verified --phase egg --faults
"$P" scripts/verify_p11_task_flow.py --template synthetic-template.sqlite3 --output acoustic-verified --phase acoustic --faults
```

每个科学任务由产品适配器创建临时 systemd unit。pytest 科学检查另由显式 `systemd-run --user --pipe --wait` 包裹，MemoryMax=1073741824、MemorySwapMax=0、CPUQuota=100%、TasksMax=64、KillMode=control-group、三个 BLAS/OMP 线程变量为 1。进程组故障专项测试不放在同一个父测试 cgroup 内，逐项验证自己的真实组。

## 运行时与来源

CPython 3.11.14、NumPy 2.2.6、SciPy 1.16.3、Parselmouth 0.4.7、Matplotlib 3.10.8、pandas 2.3.3。本轮选用已调查过的 Linux PyPI/OpenBLAS 环境，不再安装 MKL 候选。仅补 Matplotlib/contourpy/cycler/kiwisolver/pyparsing，来自上一轮官方 hash 锁，独立 `render-overlay`，不覆盖原 venv。中文显式 Noto Sans SC、英文 DejaVu Sans、IPA 固定 Doulos SIL。

REAPER 固定 [google/REAPER 提交](https://github.com/google/REAPER/tree/1d6e9b95e6b08b500fccbc9a043989dbda747276)，源码包 SHA-256 `152af842bedc98e9d12e2f59352ba1e2487ce2689bf2fbf65c68c031ab173d02`。沿 CMakeLists 列出的 11 个 `.cc` 文件，用现有 Ubuntu g++ 13.3.0、`-std=c++11 -O2 -I.` 编译，无源码改动、无 fast-math、无全局安装。构建也限制为 1 GiB/1 核。生成二进制 119728 字节，SHA-256 `cfc89b19cbe975768fc638cada5129bdf1899a0180a8bd0a37eca9190ea40be1`。上游 fread 返回值未用警告保留。Windows 继承 EXE 及其 hash 未改，原构建提交未知，二者不称逐位同构。

## 已知限制及过程失败

- 未进行生产部署、真实账号/PG 联合验收、HTTPS、跨服务全机单槽、自然长录音/最大资源压力或浏览器 UI 验收。Linux local token 认证只能证明此 API 入口的鉴权，不能代替网页账号隔离。
- Windows M01 验证脚本第一次因该环境没有 Pillow 退出，已将图像读库延迟到实际 PNG 检查处。第二次因环境代理影响 loopback 请求失败，验证客户端改为 `trust_env=False` 后第三次完成。未安装额外 Windows 依赖，失败日志保留。
- 首次 SSH 空闲连接被重置，重新连接加 keepalive；无凭据写入文件/日志。
- Linux 默认字体名仍需要调用方提供实际可用快照；本轮没有替公共 UI 修改选择器。
- Linux 分段入口已复用公共进程边界，但尚未单独完成实际 Linux 分段联合验收，capability 保持关闭。M08 仅交付接口/示例，未改其实现、未开放能力。

## 最终分阶段收口

| 阶段 | changed_files 重点 | Linux 实际结果 | 本组合成任务的峰值 |
| --- | --- | --- | --- |
| LPC | acoustic_executor、linux_runtime/bootstrap、egg_runtime fingerprint、fonts、local flock | 7 个任务含 3 成功、部分写入失败、取消、SIGKILL、30 秒超时；PNG/JSON/WAV hash 下载通过 | 105,717,760 bytes，约 100.8 MiB |
| EGG | 复用 LPC 公共执行路径，EGG child/科学核心原样 | 7 个任务含单文件三图/CSV、批次图/CSV、inverse 双 WAV、恢复任务；部分写入/取消/SIGKILL 通过 | 225,361,920 bytes，约 214.9 MiB |
| M01 | native/reaper、linux_reaper、science_child、acoustic_errors | 8 个任务含默认/要求 native/恢复；XLSX/SQLite/JSON 回读，原生 rF0 有限值与 native hash 确认；缺失 REAPER 和登记后 exit 7 均拒绝发布 | 190,951,424 bytes，约 182.1 MiB |
| 公共收口 | spectrogram_preview、font_preflight、capabilities/main | 各阶段先实际 HTTP 语谱预览；字体三角色预检；有效凭据只开放 M04/M03/M01，失配凭据和 max_running=2 均返回空算法能力 | 每次自身进程组限制，不汇总为全机峰值 |

成功任务总耗时包括读取、运行时核对、解释器启动、计算、导出与持久发布：LPC 约 4.84–4.86 s，EGG 三图约 6.41–6.53 s、逆滤波约 4.51 s，M01 约 10.68–10.81 s。输入是约 0.8 s 的公开合成 fixture，不能与前轮 10/60 s 纯计算时间直接相除。故障注入时间不计成功耗时。

LPC/EGG MemoryMax=1,073,741,824 bytes。M01 保留原调用更低的 1,000,000,000 bytes，cgroup 按页实际为 999,997,440，未扩大到 1 GiB 以上；均 CPUQuota=100%、无 swap、TasksMax=64。三组所有任务清理成功、失败结果资产已回收、临时文件为零，原模板已有 job 行逐字段保留。

Linux 最终共享后端/真实进程故障测试 **32 passed / 1 warning**；实用科学门 **35 passed**，含 ROI 和 GCI-F0 派生预算。M01 最终 76 列配对全部通过，逐列单位/有限值数/mask/最大误差在 `acoustic-comparison-final.json`。这不是对所有自然语料的统计等价研究。

最终运行时 profile：`/home/admin/ptb-p11-20260926/tasks-followup/runtime-stage4.json`，SHA-256 `677dfce8e96370db02f0dfebaf9b758142eec851c46271728820549dec60c0e55`，301 项文件 hash。最终 backend wheel SHA-256 `a4dac6d51fa03c9bbce63801b343970beea505044b555f47152d314d19ac1b1f4`。最终 wheel 使用从开始保护清单筛出的独立 `backend-p11` 输入，未将并行 agent 新增模块混入最终 P11 运行包。

能力凭据为同目录 `validated-runtime.json`。验证时显式设置 `PTB_LINUX_RUNTIME_PROFILE` 与 `PTB_LINUX_VALIDATION_RECEIPT`，真实 HTTP 返回 algorithms `[M04,M03,M01]` 和对应三种任务操作；HTTP 服务已停止，没有后台常驻生产服务。此配置只保证当前 store 单槽，不能代替 P11-PERF 全机准入。

## 证据回收与保护核对

本地证据根 `output/validation/p11-tasks-20260926/`，服务器回收副本位于 `server/`。证据包 4,185,081 bytes，SHA-256 `54928217472089379482f63cbc0973632b074235c4931791c8a49ebe53fd1560`，113 份内部证据逐项 hash 校验通过。最终安装 backend 的 89 份 Python 源码与对应工作区文件摘要一致。`owned-units-final.log` 实测 0 loaded units，所有测试 HTTP 已停止，SSH 已退出。

实际下载的 LPC 频谱、EGG CQ/SQ、EGG 语谱/F0 PNG 已打开检查，曲线、轴和图例正常，未发现裁切；它们只证明后台输出，不作为网页 UI 截图。M01 rF0 的 154 个有限值本例跨平台精确相同；最大数值差为 pB4 的约 5.86e-9 Hz，逐列证据保留，不据此修改预算。

开始保护清单 1396 个文件无缺失。P11 自身修改与同时出现的公共 UI/浏览器测试改动在 `preservation-final.json` 分列，其他 agent 的变化未回滚或覆盖。没有 Git commit/push、数据库 DDL、全局安装、系统服务、远程节点、配额迁移、EXE 或公开发布。

文档全局检查在本次快照返回 3 处错误：两处既有 M10-R5 EXE 链接，以及并行 M08 source-map 指向尚未生成的 m08-report。均不在本轮归属，未修改。P11 自身文档链接/UTF-8、Python 语法、定向 diff 空白检查通过，不称全局文档检查全绿。
