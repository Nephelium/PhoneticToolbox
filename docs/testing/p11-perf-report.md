# P11-PERF：统一资源准入与限定服务器负载

2026-09-27。**统一准入与指定合成输入的 30 分钟负载 verified；生产、大输入与长时常驻仍未验收。** [设计与验收计划](../plans/2026-09-27-p11-perf.md)。本报告的结论仅适用于指定版本、输入、两 API 进程及同 UID 部署，不扩大模块科学验证或生产开放范围。

## 本轮实现与证据边界

Linux 科学任务、公共语谱/参数表预览、字体预检，以及科学任务内的 PNG/CSV/XLSX/SQLite 导出，共用跨进程单槽。进程组限额先于重库执行安装。取消、超时、API 崩溃和子进程失败后，须确认实际进程组清空才能交出槽。等待者使用有界 FIFO，无新数据库表、守护进程或基础设施。

`server-small` 使用一槽、最多 1,073,741,824 字节科学组预算、CPUQuota=100%、TasksMax=64、MemorySwapMax=0。**1 GiB 是配置上限，绝非测得峰值。** 更低的模块上限保留。`desktop-local` 保留既有模块内存预算，不套用云端一槽或 CPUQuota。`trusted-worker` 明确不可用，不伪装为可路由的远程节点。输入、采样率、算法参数和输出完整性保持原请求语义。

通用 ZIP/解压尚未迁入受限执行协议。最终代码对 server-small 的这两类任务在读取输入/预留输出前明确返回 `server_export_unavailable`，防止绕过预算；桌面仍走原流式路径。本轮覆盖的是科学后台导出，通用 ZIP 的服务器执行性能没有被验证。

此前 `BatchMode=yes` 的 SSH 尝试没有提交密码，其失败不能证明密码被拒绝。井井明确要求使用已提供密码后，交互式 SSH 配合既有 known-host 严格检查实际成功。密码没有写入代码、命令参数、文件、日志或报告。目标为既有授权测试主机，独立目录 `/home/admin/ptb-p11-20260926/perf-20260927`，旧测试包及服务均保留。

## 先核实的前轮状态

- P11 原有 M01/M03/M04 合成输入、受限 Linux 子进程、回环 HTTP/SQLite 副本、下载 hash 证据可以复用。此前每个调用独立开组，全机准入尚未建立。
- P07-POLICY 有候选实现与独立验证，现存库政策迁移未完成。本轮不执行 DDL，不拿旧数据库规则冒充新政策。
- M04-E 已有 Windows Chrome/Qt 范围。M08 Windows 宿主已接通，Linux capability 仍关闭；M09 Linux 未开放。M14 独立核心证据不等于正式 Linux collector/capability 已接通。本轮不绕过能力门凑齐模块。
- 根文档与总台账存在历史滞后，实际判断结合源码、专项报告与原始证据。没有覆盖它们。

## 文件归属

| 文件 | 本轮内容 |
| --- | --- |
| `backend/src/ptb_worker/resource_profiles.py` | 三种显式 profile。server-small 单槽/1 GiB 候选上限；desktop-local 保留模块内存预算；trusted-worker 明确不可用 |
| `backend/src/ptb_worker/native/admission.py` | 固定同 UID 准入目录、跨进程 flock、128 项有界 FIFO、300 秒等待、取消/死亡等待者回收、清理失败关闭准入 |
| `backend/src/ptb_worker/native/launch_guard.py` | 标准库轻量启动进程保留锁，服务启动客户端即使关闭多余文件描述符，也不能因 API 退出提前释放槽 |
| `backend/src/ptb_worker/native/posix.py` | 全部受限 Linux 入口接同一准入，清理实际组及启动客户端后释放；补启动竞态重试；OOM 转明确资源失败；输出超限检查先于流式回调 |
| `backend/src/ptb_worker/native/linux_runtime.py`、`linux_reaper.py` | profile 感知的 Linux 限制，显式桌面模式不再强套 1 GiB/1 CPU 云端条件；保留科学线程版本约束 |
| `backend/src/ptb_worker/native/capabilities.py` | 未知 profile / 未实现 trusted-worker 不能广告可用能力，旧 runtime/report 哈希门保留 |
| `backend/src/ptb_worker/acoustic_errors.py` | 准入满、等待超时、配置/恢复失败的固定公开错误，保留 M14/M08 已有条目 |
| `backend/tests/test_p11_admission.py`、`test_p11_cleanup_race.py`、`test_p11_perf_systemd.py` | 跨进程 FIFO、退出/继承锁、模拟启动竞态、需实际 systemd 的并发与崩溃恢复测试 |
| `backend/tests/test_p11_task_runtime.py`、`test_p11_posix.py` | 原 server clamp 断言保留并新增桌面断言；OOM 用明确 LimitError 断言，继续要求 oom-kill 和清理证据 |
| `scripts/benchmark_modules.py`、`scripts/p11_compute_probe.py` | 双 API/共享持久库和准入、固定合成输入、逐级短测/计算与导出分项/混合 HTTP、采样及停止条件 |
| `tests/performance/test_p11_benchmark.py` | 独立分位数期望值与停止阈值/持续时间回归 |
| 本报告与 P11-PERF 专项计划 | 复现入口、已验范围、阻断和统筹建议 |

没有修改科学 core、科研参数、输入采样率、M10/M14 模块、V2、用户语料、生产数据库或根文档。已有大量未提交成果保留，没有批量 checkout/reset/stash。未本地提交、push、部署生产、生成 EXE 或安装新的第三方依赖。`executor.py`、`process_entry.py`、`main.py` 已在 commentary 释放给 M14 串行最小接线，P11 本轮未编辑这三项。

另新增 `backend/tests/test_p11_file_gate.py`，最小修改 `file_executor.py` 与 `files.py`，实现服务器 ZIP 能力门及固定公开错误码。修改前确认这两文件无已有 diff。负载期间 `files.py` 出现并行 P07 的保留政策修改，已保留；本轮只拥有 `server_export_unavailable` 错误码行。最终服务器 wheel 不包含这段后续并行政策修改，该合并状态不能自动继承本轮服务器验证。

## 准入、公平性与故障语义

锁目录固定为 `/run/user/<uid>/ptb-resource-admission-v1`，不依赖数据库、API 工作目录或 runtime 路径。所有参与的 API/worker 必须使用同一专用 Unix 账号。不同 UID、容器命名空间和不同机器不在此锁覆盖范围。

FIFO 按到达准入层的请求排序，后来预览不能无限插队。队列最多 128 项，等待最多 300 秒，满队列、超时或不安全/损坏状态明确失败。等待中取消移除本请求，死亡等待者用 PID 加 `/proc` 启动身份回收。现有持久库继续处理 owner 单运行、租约、generation fencing 及结果发布。**没有按账号加权/轮转公平**；持久任务可先成为 running，再等待计算槽，collector 另记录 `queue_wait_seconds`。

锁描述符由轻量 `launch_guard` 保持，直到其 systemd-run 客户端结束。API 意外退出不能提前释放仍在运行的科学组。清理只操作日志中随机名称的自身 unit；客户端启动竞态未结束、cgroup 仍 populated 或清理无法确认时关闭准入。后续请求先恢复该精确 unit，再执行。

API 在排队前仍可能持有请求及输入缓冲。科学组预算不能当作整个服务器内存硬上限。本次同时测量 cgroup 聚合峰值、两个 API 各自 PSS 和整机 MemAvailable/PSI，未相加 RSS 冒充整机峰值。两 API 以外的部署规模、大输入并发缓冲及全机硬限制仍需后续验证。

## 平台与版本

连接时实测总内存 **3,669,323,776 字节（3.42 GiB）**，MemAvailable **3,031,379,968 字节（2.82 GiB）**，Swap **0**，磁盘可用 **44,029,714,432 字节（41.0 GiB）**。Ubuntu 24.04.2 / Linux 6.8.0-63-generic / x86_64，2 vCPU。起测前无运行中的 `ptb-*` 单元。

复用已有 CPython 3.11.14、NumPy 2.2.6、SciPy 1.16.3、Parselmouth 0.4.7、Matplotlib 3.10.8、pandas 2.3.3、phonetic-core/ptb-api 3.0.0a1，以及既有 render/pandas overlay 和原生 REAPER。只在新测试子目录构建并安装本项目 backend wheel 的独立 overlay，未安装新全局依赖或修改旧环境/系统配置。

第一次故障/短测 profile 为 `4f9f8f911b60f727621c5cfbc0d6b3c6203c37df55d2f6ffd92f4b6bae6118a8`。增加 ZIP 拒绝门后重新构建独立 `backend-overlay2`，最终 `runtime2.json` SHA-256 为 **`c17de6420e64f951a8f2a2e49c1841dba49f9bdd74231661f01c4754a6d6ccad`**，重新通过三阶段任务后才生成 `receipt2.json`。未直接复用旧凭据。并行 M14 的新 main/executor/process_entry 接线没有被混入本次固定运行包，当前工作区的其他修改也不能自动继承这次负载结论。

输入来自公开 `tests/fixtures/m03/EGG-SYN-PCM16.npz` 的数组，经 SciPy WAV 编码，44,100 Hz、35,280 帧、0.8 秒；文件名中的 PCM16 描述来源 fixture，不代替本轮编码格式证据。

| 输入 | 字节 | SHA-256 |
| --- | ---: | --- |
| 双声道 | 282298 | `da33a11b3270c2106fb7134df0f4c6fbdb43f98a80356c7485b0b1c5746e8357` |
| 单声道 | 141178 | `8b1ed2354023e3d05254116fd6108dd369b2f7740152c4760e668a955c3e5b7d` |

M04 `roi_end=0.05`，M03 `mode=single, roi_end=0.5`，M01 使用 AcousticConfigSnapshot 默认完整配置。OMP/OPENBLAS/MKL 线程均为 1。没有根据负载调整参数。中文 Noto Sans SC、拉丁 DejaVu Sans、IPA Doulos SIL，字体预检保存实际文件 hash。Noto `763146584cf0710223441356b4395e279021b0806c196614377a7a0174ae074a`，DejaVu `3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22`，Doulos `cc89b87c047bcc8dc00246398218bb6343e0f2372b153cf560c29a77f27068ef`。

## 定向验证

证据根为 `output/validation/p11-perf-20260927/`。

| 平台/范围 | 实际结果 |
| --- | --- |
| Windows 准入/profile/公共执行/字体/LPC 定向 | 42 passed，18 Linux/systemd 所需项 skipped，2 条既有框架弃用警告；`windows.xml` |
| WSL 原生 flock、多进程、PID 身份、FIFO、取消/超时/死亡恢复 | 27 passed，2 真实 systemd 项 skipped；`wsl.xml` |
| 实际服务器准入、父进程崩溃、启动竞态、进程组超时/取消/聚合 OOM/输出限额 | 23 passed；`linux-admission.xml`，包含 4 个独立父进程不可重叠的实际进程区间断言 |
| 新 ZIP 拒绝门 Windows / 实际 Linux | 各 8 passed；与下一行重叠，不累加 |
| Windows ZIP 原有策略及新门 | 17 passed；`archive-regression.xml`，无数据库操作 |
| 三模块实际任务故障门（第一包） | 22 个任务覆盖部分结果写入回滚、运行取消、实际 SIGKILL、M04 真实 30 秒停滞超时、M01 REAPER 缺失/退出失败及后续恢复 |
| 最终包三模块再验证 | 7 个真实任务成功，HTTP 停止、临时资源无残留，报告绑定最终 profile |
| 架构、语法、定向 diff 空白 | 通过；架构 `errors=[]` |
| 全局文档检查 | 收尾检查 803 文件、333 来源、41 任务，仅剩 README 的既有 M10-R5 EXE 缺链；未修改根文档，不称全仓全绿 |

模拟启动竞态与假客户端关闭 fd 只用于对应单元测试；真实系统组与崩溃恢复有上述实际 Linux 证据。未改变数值容差或跳过失败科学用例。

## 第一轮分级短测

`short-run1` 成功，无非预期失败、无停止条件触发。每模块 1 次 first-observed + 5 次 cache-warm，共 18 个端到端科学任务；另有每模块 6 次独立计时。每次都启动新的科学解释器，warm 仅指操作系统/字体缓存。没有清系统缓存或重启服务器，不声称获得严格机器冷缓存样本。最终运行另记录两个新 API 进程至首次健康响应的启动时间。

| 模块 | 端到端 n | 端到端 p95/p99（秒） | 纯计算 n | 纯计算 p95/p99（秒） | 导出 p95/p99（秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| M04 | 6 | 5.421 | 6 | 0.002164 | 0.796 |
| M03 | 6 | 7.410 | 6 | 0.087166 | 1.968 |
| M01 | 6 | 12.279 | 6 | 6.834 | 0.508 |

纯计算直接计时真实核心调用，M01 包括原生 REAPER。导出分别为 M04 PNG、M03 CSV/PNG、M01 XLSX/SQLite 构造。端到端包含提交、等待、子进程、持久发布、轮询及所有文件下载 hash 核对。JSON/选区 WAV 等附带准备不被错误计入上述纯计算函数计时。p95/p99 使用 nearest rank，n=6 时均落在最大值，不能理解为精确尾分布估计。

36 个 API collector 均 `cleaned=true`；加上独立诊断组的观测峰值约 215.7 MiB。整机 MemAvailable 最低 2,476,589,056 字节（2.31 GiB）；两个 API 各自 PSS 峰值为 105,340,928 与 131,201,024 字节，未将它们相加称为整机峰值。296 个整机观测，证据目录观测占用最多 7,787,635 字节。

## 最终包混合负载

**限定范围 verified：最终包混合阶段实际 1800.520 秒，短门及持续负载均通过，停止条件未触发。**

最终运行共 53,467 个 HTTP 响应，其中短门 776 个、混合阶段 52,691 个。非预期 HTTP 失败 **0/53,467（0%）**；33 次 HTTP 429 单列为预览 busy，不计作成功结果。没有对失败样本作删减。

相对原合成模板的新增持久任务为 **133 成功、76 计划取消、0 失败/中断/遗留运行**，包括短门的 18 成功任务；混合阶段本身为 115 成功、76 取消。模板自带历史失败/中断记录已从本轮任务失败率排除。19 轮批量/取消恢复完成，所有成功结果均逐文件下载并核对 hash。

| 混合阶段接口 | 成功样本 | busy | 成功响应服务端 p95/p99（秒） | 成功响应 loopback p95/p99（秒） |
| --- | ---: | ---: | ---: | ---: |
| 健康检查 | 23458 | 0 | 0.044 / 0.101 | 0.104 / 0.177 |
| 任务列表 | 23339 | 0 | 0.238 / 0.484 | 0.270 / 0.534 |
| 公共语谱 | 167 | 11 | 13.450 / 19.769 | 13.496 / 19.828 |
| 参数表预览 | 40 | 22 | 8.737 / 16.019 | 8.750 / 16.028 |

轻量 API 服务端 p95≤500 ms 的原目标在该模型下通过。首分钟轻交互阶段 1,965 个轻接口样本，合并 p95/p99 为 0.031/0.075 秒；计算并行阶段 44,832 个轻接口样本，合并为 0.166/0.376 秒。任务列表最大服务端耗时 1.497 秒，不能称每次均低于 500 ms。公网 RTT 未测。

公共预览明显受单槽排队影响：语谱 busy 为 11/178，参数表为 22/62。上述表格排除 busy 来计算成功请求耗时，避免用快速拒绝压低延迟。原 report.json 同时保留包括短门和 busy 的总体分位数，口径不同不能直接混用。

| 模块 | 混合阶段已单独计时的成功任务 n | 端到端 p95/p99（秒） | 最终短门 first-observed（秒，n=1） | 5 次 cache-warm 最大值（秒） |
| --- | ---: | ---: | ---: | ---: |
| M04 | 45 | 9.265 / 10.446 | 5.262 | 5.466 |
| M03 | 25 | 12.982 / 13.487 | 7.251 | 7.468 |
| M01 | 26 | 18.750 / 19.834 | 11.724 | 11.721 |

另外 19 个突发批次中的成功任务做了完成/下载核验，但未单独记录完整提交至下载的计时，不补造 E2E 样本。所有 358 次实际准入的等待 p95/p99/最大值为 **5.361/13.499/15.127 秒**。

最终包独立纯计算/导出计时每模块各 n=6。纯计算 p95/p99：M04 0.002894 秒、M03 0.095748 秒、M01 6.039444 秒；导出 p95/p99：M04 0.811782 秒、M03 1.947576 秒、M01 0.521715 秒。小样本 p95/p99 同为最大值。两个新 API 进程启动至首次健康响应分别为 **1.931、1.893 秒**，操作系统缓存未清空。

| 资源指标 | 实测值 |
| --- | --- |
| 观测次数 | 1,866 次整机/进程组采样，另有每个 collector 的退出证据 |
| 计算组最高峰值 | **226,177,024 字节（215.699 MiB）**，取 API collector 与独立诊断组的完整 memory.peak 最大值 |
| 两 API 各自 PSS 峰值 | **281.714 MiB、255.724 MiB**，不与 RSS 相加当整机峰值 |
| 整机 MemAvailable 最低 | **2,134,528,000 字节（1.988 GiB）** |
| 整机 memory full PSI avg10 最大 | **0.18**，未达到持续 >1 的停止阈值 |
| 本轮负载进程组 OOM/oom_kill | **0/0**；故障门的受控 OOM 另计，未混入正常负载 |
| 磁盘空闲最低 | **40.651 GiB** |
| 运行期间证据目录观测占用最大 | **42.905 MiB** |
| 最终目录占用（含结束后写入的原始 HTTP 样本等） | **54.965 MiB** |

API PSS 在混合阶段第 0/10/20/30 分钟分别约为 **103.14/275.40/278.05/280.54 MiB** 和 **126.34/226.95/229.19/254.63 MiB**。主要增长在前期，但第二 API 后段仍有增长，原因未定位。当前没有持续内存压力，**不据此宣称长时常驻内存稳定或不存在泄漏**，不外推小时/天级运行。

364 次 API collector 调用中，358 次实际创建 unit，全部 `cleaned=true`；其余 6 次取消发生在启动前，证据为空，不是遗留组。独立 18 次诊断也已清理。收尾确认两个 API PID 不存在、准入 `queue=[] / active=null`、运行中的 `ptb-*` 单元为 0。两个 SSH 会话已退出。

原始证据已回收至 `output/validation/p11-perf-20260927/server/`，657 个内部文件逐项 SHA-256 校验通过。证据包 2,892,746 字节，SHA-256 **`3c60318f09ca8944eac91d5922042d2afa40aab2a7fb84e26fdc934dda5ae531`**。最终包另做实际 Linux 定向复验 **31 passed / 0 skipped**，包含受控 OOM；它位于混合负载结束之后，不计入正常负载失败率。

负载使用两个 API 进程、共享 SQLite 副本、local token 和十个合成 HTTP 客户端。首分钟为轻交互阶段，随后一条科学任务提交链加九条交互链，交替两 API 的公共语谱/参数预览，周期性四任务突发、排队取消、running 状态取消及后续 LPC 恢复。真实运行中子进程取消的证明另来自故障门 `on_started`，不把仅观察到任务 running 等同于子进程已经启动。

这不是十个真实网页登录账号或生产 PostgreSQL 验收。原始样本分开记录服务端响应准备时间与 loopback 客户端时间，公网 RTT 保持 null，未混算。下载 hash 核对证明所测结果完整。HTTP 429 为预期 busy，另外统计，不能静默删掉。

原计划轻量 API 目标为服务端 p95≤500 ms，分别检查健康接口及任务列表；计算任务及公共预览独立报告。

约每秒采集 cgroup memory.current/peak/events/pressure/cpu.stat、两个 API PSS、整机 MemAvailable 及 memory/cpu/io PSI、磁盘空闲和证据目录占用。停止条件为 MemAvailable <700 MiB 或 memory full PSI avg10>1 持续 10 秒、OOM、磁盘空闲<10 GiB、API/worker 异常或资源归属未知。只清理本任务拥有的 PID 和精确 unit，不终止其他服务。

## 复现入口

在已有授权 Linux 测试目录，使用既有、无活动任务的合成 SQLite 模板。独立安装当前 backend wheel，按前轮方式生成包含实际源码/解释器/科学库/字体的 runtime profile，运行三个阶段的 `verify_p11_task_flow.py`，成功后生成绑定报告 hash 的 receipt。禁止以手工改 success 或旧报告绕过 capability。

```sh
P=/home/admin/ptb-p11-20260926/venv/bin/python
export PTB_RESOURCE_PROFILE=server-small
export PTB_LINUX_RUNTIME_PROFILE=/absolute/test/runtime2.json
export PTB_LINUX_VALIDATION_RECEIPT=/absolute/test/receipt2.json
# PYTHONPATH 使用对应 backend overlay 与既有 render/pandas overlay
PTB_P11_SYSTEMD_TESTS=1 "$P" -m pytest -c tests/pytest.ini backend/tests/test_p11_admission.py backend/tests/test_p11_cleanup_race.py backend/tests/test_p11_perf_systemd.py backend/tests/test_p11_posix.py backend/tests/test_p11_file_gate.py -q -p no:cacheprovider
"$P" scripts/benchmark_modules.py --template "$SYNTHETIC_TEMPLATE" --output "$NEW_SHORT_OUTPUT" --authorized-test-directory
"$P" scripts/benchmark_modules.py --template "$SYNTHETIC_TEMPLATE" --output "$NEW_MIXED_OUTPUT" --mixed-seconds 1800 --authorized-test-directory
```

输出目录必须此前不存在，失败证据保留。第二条基准也先执行完整短门和分项计时，再开始 1800 秒。脚本复制合成库，不运行 DDL，不安装依赖，不修改系统配置。停止后收集 `report.json`、`http-samples.json`、`jobs.json`、`compute-samples.json`、`system.jsonl`、`collectors/` 与 `processes/`，结合当前源码 hash 复核。

## 开放边界、未达标及未测

- 当前候选可开放范围只涉及已通过能力门的 M01/M03/M04、公共预览、字体和科学后台导出。指定输入的最终持续负载已通过。只支持同 UID、两 API 的本次合成请求模型，无生产部署授权或性能承诺。
- 通用 ZIP/解压服务器侧明确不可用。M08/M09、M14 正式 Linux collector/capability、trusted-worker 不纳入本轮，不临时绕过能力门。
- 未达标/待优化：公共预览排队可达约 20 秒且存在 busy；未给出预览响应保证。API PSS 增长原因与长时稳定性未闭合。
- 未测：真实账号/PG、新配额迁移、公网 RTT/HTTPS/浏览器、自然及长录音、所有模块最大输入、严格机器冷缓存、多 UID/容器、更多 API、大请求缓冲并发与整机硬限额。
- 本轮未实现按账号加权/轮转调度，未接通远程节点领取/科学执行/回传闭环。FIFO 已验证不能替代这些能力；并行节点组件进展见下方交接。
- 后续 trusted-worker 需节点注册/撤销、心跳及资源 profile、owner/operation/core/resource hashes 约束、HTTPS claim/lease/generation、受控输入下载/输出上传、hash/完整性验证、取消/超时/离线回收、旧节点 fencing 及原子发布。应复用现有任务/文件协议；当前没有已接通可用节点。

## P06-REMOTE-NODE 收尾接口核对

收到并只读核对 [C 节点精确交接](../specs/p06-node-handoff.md)。节点组件已有独立实现，不能继续笼统称整仓没有节点代码；本次服务器包及已验证能力仍不包含节点执行闭环。节点工作流的测试结论归其专项报告，本轮未复跑或扩大验证。

| 节点所需接口 | 当前 P11 可复用部分 | 尚缺能力 |
| --- | --- | --- |
| trusted-worker 本机预算与单槽 | 同 UID 准入、Linux cgroup 限制、先限额后启动 | trusted-worker 明确拒绝；没有节点专属可信预算配置及临时磁盘预算绑定 |
| 固定 entry / 请求 / scratch | `collect_scientific(entry, request, scratch, limits, stop, ...)`、固定入口和同版 core | 节点 operation/receipt 交集、租约/generation 到本地调用的正式 binding |
| abort / recover / 清理证明 | cancellation predicate、精确 `recover_abandoned_unit`、collector `cleaned` 证据 | 独立公开 abort/recover handle、节点实例身份持久关联与重启恢复联验 |
| 有界流式输出 | 当前代码已有 `on_chunk` 回调，并在调用回调前检查累计输出限额 | M01/M03/M04 当前仍接聚合 bundle；节点背压、断点上传及最终清单尚未接通。不能把已有回调称为完整节点流式协议 |
| capability receipt | 当前 runtime/core/依赖/字体 hash 与实际模块报告绑定 | 节点协议与服务器双方版本/能力交集、撤销和重新资格验证 |

C 报告的 WSL 缺少硬限制运行条件保持不可用，不通过设置 server-small 冒充 trusted-worker。服务器相对租约时间、冻结生成模型、凭据轮换和幂等回传等要求交由 P06 协议负责方处理。后续须联合验证节点故障、服务器接管、旧 attempt 拒收、恢复后重新领取及唯一发布，本轮不改公共 adapter 或开启 capability。

## 仅供统筹的更新建议

根 README、总台账和 ADR 本轮不覆盖。建议单列 `server_small_synthetic_load`、`same_uid_admission`、`desktop_profile_preserved`、`remote_worker_unavailable`，保留 P07/M08/M04-E 及生产/大输入未验证边界；登记通用 ZIP 服务器不可用及未来受限迁移要求。无 push、生产部署、EXE、V2/用户语料修改、数据库 schema 或系统配置变更。
