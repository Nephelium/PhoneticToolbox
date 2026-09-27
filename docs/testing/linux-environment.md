# P11-ENV / P11-LINUX 实际证据与交接

**后续进展（2026-09-26）：** 用户接受实用等价并授权完整任务接线后，LPC/EGG/M01 已完成实际服务器受限任务、结果生成和回环 HTTP 下载验证，范围与最终 hash 见 [P11 任务链路报告](p11-task-flow.md)。本页保留环境安装与旧数值门的历史事实；不再把下文“完整科学执行器未接通”作为最新状态。真实网页账号/PG、全机并发及生产部署仍未验收。

2026-09-26。任务来源：[服务器统筹任务卡](../plans/2026-09-26-server-coordination.md)。实施决策与授权变化见[本轮环境门计划](../plans/2026-09-26-p11-environment-gate.md)。本报告替代此前“未连接/未安装/Linux blocked”的阶段记录。

## results / 状态边界

| 范围 | 状态 | 实际结果 |
| --- | --- | --- |
| P11-ENV 环境基础门 | `verified`，限定下列合成服务范围 | 现有 WSL 与授权服务器均使用原生 Linux Python、安装 wheel；API/静态 JS/CSS 字节一致、中文/IPA 字形覆盖和 raster 实测通过 |
| P11-LINUX 通用进程组/参数表 | `verified`，限定实际服务器 | 本轮安装 wheel 的 42 项定向测试通过，含 8 项进程边界、参数 SQLite/XLSX/异常和认证边界 |
| P11-LINUX 完整任务 | `in_progress` | M03/M04 精确基线 23 passed / 19 failed；科学执行链路、公共语谱/后台字体、完整 capability 仍待完成 |
| Windows 原路径回归 | `verified`，限定本轮测试集 | 当前 backend wheel：39 passed / 9 Linux skips / 2 deprecation warnings；含参数读取、M03 字体真实兼容子进程、API、探针 |
| 云端负载、远程节点、统一 UI、生产 HTTPS | 未验证 | 不扩大原模块 Windows verified，不更新全局台账，不将合成 API/font raster 当作浏览器或部署验收 |

**Linux 科学 capability 保持关闭。没有改动算法、基准文件或容差来消除 19 项失败。**

## changed_files / 归属与保护

工作树 `D:/PhoneticToolbox/PhoneticToolbox_v3`，分支 `codex/v3-rebuild`，开始 HEAD `9d0283ce135f95c21c31fcc807638e4a78a4c7c8`，common Git dir 为相邻 v2 的 `.git`。原 M12、桌面、公共前端、统筹文件等已有未提交成果保留。目标后端文件开始时没有已有差异。

修改前对 Git 跟踪及非忽略未跟踪的 1,379 个现存文件记录 SHA-256，仅存摘要。最终核对：1,377 个原文件摘要不变，仅下列两项原文件变更，意外变更 0、缺失 0，见 `preservation-final.json`。本轮认领以下 15 个文件，只有前两项是修改原文件：

| 文件 | 行为/责任 |
| --- | --- |
| `backend/src/ptb_api/main.py` | 调用真实资格检查，Linux 科学不可用原因明确 |
| `backend/src/ptb_worker/parameter_preview.py` | 新增 Linux 受限 stdio 参数表读取，保留原 Windows Job 路径 |
| `backend/src/ptb_worker/native/posix.py` | user systemd 受限任务组，输入/输出/超时/取消及整组清理 |
| `backend/src/ptb_worker/native/process.py` | 固定模块 allowlist；当前仅参数表 |
| `backend/src/ptb_worker/native/capabilities.py` | M01 的批次、REAPER 实物/hash、版本与平台检查 |
| `backend/tests/test_p11_posix.py` | 实际 Linux 进程组/资源/恢复，非 Linux 明确 skip |
| `backend/tests/test_p11_parameter_preview.py` | 固定入口拒绝、真实 SQLite 中文/IPA/数字/null、错误/超时 |
| `backend/tests/test_p11_capabilities.py` | Linux 仅有 batches 时拒绝广告能力、Windows 缺原生资源拒绝 |
| `tests/support/p11_process_fixture.py` | 合成父子进程、内存/输出压力、实际子进程内部限额读取 |
| `tests/architecture/test_linux_environment_probe.py` | 20 项探针失败边界；模拟分支不当真实平台证据 |
| `scripts/verify_linux_environment.py` | 只读原生 Linux 盘点、旧脚本审计、可选 loopback HTTP |
| `scripts/verify_linux_smoke.py` | 当前安装 wheel 的无数据库统一 API/静态资源/显式字体 raster |
| `scripts/verify_linux_science.py` | 在预先受限 cgroup 内记录原基准差异和构建 fingerprint，失败返回 2 |
| `docs/plans/2026-09-26-p11-environment-gate.md` | 授权、Superseded 新 WSL 提案、边界决策、剩余门 |
| `docs/testing/linux-environment.md` | 本报告 |

不改台账/ADR/contracts/source-registry、科学 core、Windows native/MKL 校验、旧 EXE、v2 或研究数据。未 Git commit/push、reset/stash、服务数据库 DDL、系统配置、sudo/apt、持久服务或公开部署。独立合成测试自行创建的内存 SQLite 表不涉及现存/服务库。

## 平台与环境实测

| 项目 | 现有 WSL | 实际服务器 |
| --- | --- | --- |
| 位置 | NInfer `/home/ninfer/ptb-p11-20260926` | `/home/admin/ptb-p11-20260926` |
| OS / 内核 | Ubuntu 24.04.4，WSL2 6.6.87.2，x86_64 | Ubuntu 24.04.2，6.8.0-63，x86_64 |
| 资源 | 约 31 GiB RAM / 8 GiB Swap | 2 vCPU；MemTotal 3,583,324 kB；初查 available 3,002,732 kB；无 Swap |
| 磁盘 | Linux 自身文件系统，未在 `/mnt/d` 运行 Windows 解释器 | 初查总 52,448,063,488 / 可用 47,108,841,472 bytes |
| 宿主能力 | PID1 WSL init，无 systemctl、无可写 cgroup 委托；仅环境 smoke | user systemd 255.4，cgroup v2 可执行临时受限组 |
| 项目解释器 | `venv/bin/python`，CPython 3.11.14 | 同版 portable CPython 与 `venv/bin/python`；系统 Python 3.12.3 保留 |
| 核心/API import | 项目 Linux venv 的 site-packages | 项目 Linux venv 的 site-packages |

SSH 密码认证实际成功，沿用既有 known_hosts 并严格检查已知 host key。未另外完成带外指纹核验。凭据没有写入项目、日志、脚本、报告或 commit。此次使用 SSH 终端连接；域名 HTTPS/TLS 仍属 P15-STAGING。

井井选择复用现有 WSL，未创建发行版、改变默认 WSL 或编辑 `.wslconfig`。普通用户目录内准备解释器/venv和测试文件，现有 NInfer 模型与全局环境未改。服务器任务目录最终 `du -sk` 为 709,304 KiB，用户 pip 缓存目录观测为 98,484 KiB（不是运行峰值，未区分历史缓存）；pip 缓存保留。没有以 root 安装依赖。

初始只读盘点未在 PATH 找到 PostgreSQL 工具/字体工具，不据此断言未安装。实际 smoke 不配置 account/storage/job 数据库，因此不会触发原服务入口的 recover/cleanup。端口仅自身 127.0.0.1 随机临时端口，结束时验证已停止。

## 可复现输入、依赖与产物

证据根为 `output/validation/p11-env-20260926/`，该目录按项目规则忽略，不包含原始研究语料或凭据。

- `python-download.json`：官方 python-build-standalone 20260114、CPython 3.11.14、Linux x86_64 stripped 包，31,204,420 bytes，SHA-256 `7fb42e7ac220ec607c5eddbf0361e523279f8cee17fd994bf5fe521676a63950`，与官方 release asset digest 一致。
- `linux-input-manifest*.json`：最小源码/合成 fixtures/既有前端构建的逐文件输入摘要。不重新构建或修改他人前端；静态 smoke 限定这些已存在资产。
- `linux-api-candidate.in/.lock`：34 个完整传递依赖及 hash。包括现有 FastAPI 0.141.1、Pydantic 2.13.5、Uvicorn 0.52.4 等；第三方 wheel 从官方 PyPI 获取，Linux wheel 平台显式指定，require-hashes。pip check 在 WSL/服务器均通过。
- `linux-table.in/.lock`：openpyxl 3.1.5、et-xmlfile 2.0.0 两个既有 extra，require-hashes 安装通过。
- `linux-science-candidate.in/.lock`：完整候选 **未通过安装门**。PyPI 下载很慢；阿里云镜像未同步 pyparsing 3.3.3，明确失败，未改版本。保留 `science-install-mirror.log`。
- `server-final/project/linux-science-numeric.lock`：从同一官方 hash 锁提取 NumPy 2.2.6、SciPy 1.16.3、Parselmouth 0.4.7，镜像传输、官方 hash 校验，用于本轮纯数值基线。没有把未安装的 Matplotlib/Pandas 写成已验证运行时。
- 自有 `phonetic-core`/`ptb-api` 均 3.0.0a1，先在 Linux 构建并安装 wheel；最终 backend 在服务器再次构建/安装，SHA-256 `09338f28a788415704c983c8b4c0463a2dca03d3037b24ea028483b3191506e6`。Windows 当前源码另构建 wheel 安装到 `.venv/p11-windows-wheelcheck`，依赖使用 `.venv/m09-ui`，没有覆盖原运行环境。
- `server-final-evidence.tar`：52 个文件的摘要清单核对通过，整体 SHA-256 `4130aa379b8068143ad92782b8a82ff3db0ff1f47184b1b967070034664f09a3`。
- `server-supplement.tar`：修正诊断对象值摘要后的双轮诊断、安装后环境/包清单，SHA-256 `9cb5c26398d32e54f9f85f4c211bbc5ae762dd01eb173e6139c481abc0585bb0`。本地 runner 与服务器 runner 内容摘要一致。

字体显式加载项目已有 Doulos SIL 和本机 NotoSansSC-VF.ttf 的测试副本。Noto 字体 name13 为 SIL OFL 1.1，未全局安装或加入正式发行。中文字体 SHA-256 `763146584cf0710223441356b4395e279021b0806c196614377a7a0174ae074a`，IPA 字体 SHA-256 `cc89b87c047bcc8dc00246398218bb6343e0f2372b153cf560c29a77f27068ef`。两平台都检查 cmap/非空 raster，服务器 PNG 已实际查看。此证据不覆盖浏览器 CSS 回退或 Matplotlib 后台字体发现。

## commands / 实际验证

Windows（项目根 PowerShell；当前 wheel import 路径见 `windows-imports.json`）：

```powershell
$env:PYTHONPATH='D:\PhoneticToolbox\PhoneticToolbox_v3\.venv\p11-windows-wheelcheck'
& .venv/m09-ui/Scripts/python.exe -m pytest -c tests/pytest.ini backend/tests/test_p11_capabilities.py backend/tests/test_api.py backend/tests/test_p11_posix.py backend/tests/test_p11_parameter_preview.py backend/tests/test_m02_display.py backend/tests/test_m03_font_preflight.py tests/architecture/test_linux_environment_probe.py -q -p no:cacheprovider --junitxml=output/validation/p11-env-20260926/windows-final-v2.xml
```

结果 **39 passed / 9 skipped / 2 warnings，8.42 s**。Linux 专项明确跳过，不能计入 Windows 实测。另用实际登记的原 REAPER/hash 和当前包版本直接验证 Windows M01 资格返回 `(True, None)`，不把资格检查当整条科学任务验收。

服务器（`/home/admin/ptb-p11-20260926/project`，`P=../venv/bin/python`）：

```sh
PTB_P11_SYSTEMD_TESTS=1 PTB_P11_EVIDENCE_DIR=/home/admin/ptb-p11-20260926/final-process-evidence   "$P" -m pytest -c tests/pytest.ini backend/tests/test_api.py backend/tests/test_p11_capabilities.py backend/tests/test_p11_posix.py backend/tests/test_p11_parameter_preview.py backend/tests/test_m02_display.py tests/architecture/test_linux_environment_probe.py -q --tb=short -p no:cacheprovider --junitxml=/home/admin/ptb-p11-20260926/final-tests.xml
"$P" scripts/verify_linux_smoke.py --frontend frontend/dist --chinese-font test-fonts/NotoSansSC-VF.ttf --ipa-font backend/src/ptb_worker/assets/DoulosSIL-Regular.ttf --output /home/admin/ptb-p11-20260926/final-smoke
"$P" -m pip check
```

结果 **42 passed / 1 warning，13.11 s**；smoke passed，pip check 无破损依赖。API、worker、core 的 import 均来自上述 venv site-packages。先前阶段 7 项 process 测试和 20 项 probe 测试不是额外累计到 42 中。

科学原基线使用 user systemd transient unit，`MemoryMax=1073741824`、`MemorySwapMax=0`、`CPUQuota=100%`、`TasksMax=64`、`KillMode=control-group`、`RuntimeMaxSec=100`、BLAS/OMP/MKL 线程均 1，之后执行：

```sh
"$P" -m pytest -c tests/pytest.ini tests/parity/test_egg_analysis.py tests/parity/test_lpc_spectrum.py -q --tb=short -p no:cacheprovider --junitxml=/home/admin/ptb-p11-20260926/science-results.xml
"$P" scripts/verify_linux_science.py --root /home/admin/ptb-p11-20260926/project --output /home/admin/ptb-p11-20260926/science-diagnostics-3.json
```

原测试 **23 passed / 19 failed / 2 warnings，1.65 s**；诊断另以新 unit/新输出进行两轮，均返回 2。完整 argv 和 returncode 在 `server-final/final-results.json`，原始失败 traceback/XML 完整保留。

探针默认只读，禁止 Windows exe 冒充 Linux、拒绝公网/凭据 URL与覆盖旧证据。盘点故意返回 2/`acceptance=incomplete`，它不自动把环境盘点升级为任务验收。`--audit-root .` 只审计不执行旧脚本。构建/验证首次错误也保留：误用 root pytest 配置时的缺 coverage 插件参数，改为项目指定 `tests/pytest.ini`；Windows 下载 Linux wheel 首次触发 Windows marker/hash 条件，改用已完整解析 Linux 锁的 `--no-deps` 下载后在 Linux pip check 验证。没有跳过失败科学用例。

## 进程组与资源证据

`server-final/final-process-evidence/*.json` 记录每个唯一 unit、cgroup、限制、peak、返回码和清理结果。

- 子进程启动后实际读取：`memory.max=512000000`、`memory.swap.max=0`、`cpu.max=100000 100000`、`pids.max=64`。限制先于重库和业务输入处理建立。
- 父子分别申请 45,000,000 bytes，组配置 72,000,000 bytes 时被 OOM kill；systemd `MemoryPeak=71,999,488` bytes、Result=oom-kill，子进程已不能执行。这证明合计限制，不能等同科学任务峰值。
- 崩溃、留下子进程的超时、取消、输出超限均经过实际注入；各 unit `cleaned=true`。取消不影响测试另起的无关哨兵进程，输出超限后下一项正常 echo 成功。
- 取消/超时合成父子组观测峰值约 13.9 MB。极快 echo 在采样前结束时 peak 缺失，未填 0 冒充实测。
- 数值诊断一次组 `memory.peak=177,299,456` bytes，限额 1 GiB。只是短合成基线，不代表 M03 长录音/导出预算，也不是整机/API父进程峰值。
- 最后 `systemctl --user list-units --all --no-pager 'ptb-p11-*'` 为 **0 loaded units**。自己的 API 停止；没有停止现有用户服务/其他任务或关闭整个 WSL。

## 科学差异与 fingerprint

NumPy wheel 的构建配置含 OpenBLAS 0.3.29，SciPy 的配置/加载库另完整记录在诊断 JSON；`/proc/self/maps` 中实际加载的 BLAS/LAPACK 文件 SHA-256 已保存。原 Windows MKL 校验保持不变。仅能确认两个构建有差异，不能把所有数值变化的因果都直接归给 BLAS。

| 对比字段/场景 | 最大绝对差 | 结论 |
| --- | --- | --- |
| M03 8 变体的预处理 EGG，35,280 点 | 4.3863254350906544e-7 | 原精确门失败；近零参考值使最大相对差达 51.38，不解释为整体相对误差 |
| M03 Praat F0 | 3.362288225616794e-10 | 原精确门失败 |
| M03 inverse auto / explicit | 1.1757261830780408e-13 / 8.837375276016246e-14 | 原精确门失败 |
| M04 default/dynamic_2k/dynamic_48k 幅度 dB | 6.30766550102635e-11 | 原字节门失败 |
| M04 noise_order1 / noise_order200 / noise_96k | 3.9968028886505635e-15 / 5.5138116294983774e-12 / 7.283063041541027e-13 | 原字节门失败 |
| M04 constant / minimum52 | 9.933998068589744e-13 / 4.362732397567015e-12 | 原字节门失败 |

诊断记录 105 个实际比较，其中 86 个精确相同、19 个不同。另捕获 7 个 EGG 变体的事件列表断言和 2 个 LPC 动态 y_range 断言，相关测试在该断言后尚未遍历的字段不声称比较完成。各数组 shape/dtype、有限值绝对/相对差、NaN/Inf mask和实际输出摘要可逐项核查。双轮诊断 3/4 的 105 项记录相同。

早期诊断 1/2 对 scalar None 使用 object array 内存地址摘要，导致 6 项伪重复差异。已改为对象值 JSON 摘要并独立复跑 3/4；旧证据保留并标作诊断器修正历史，没有据此改科学算法或放宽原 pytest 判据。

## 旧验证入口逐项处置

完整 37 文件的 hash 和具体命中行在 `script-audit.json`，原始搜索线索在 `legacy-audit-lines.txt`。下表以人工复核的入口依赖分组，每个文件均列出。静态正则可能将注释/普通变量计为指标，不能替代被导入代码审查，也不能据无直接命中判断没有间接写操作。

| 文件（均位于 scripts/） | Linux 迁移处置 |
| --- | --- |
| verify_m01_browser.py | 固定 Windows 浏览器/解释器及后台桥接；重建 Linux 宿主入口后再迁 |
| verify_m01_contract.py | 检查数据/基线/SQLite 输出归属，不能原脚本直连服务器 |
| verify_m01_downloads.py | 本地任务库/服务与下载；需 Linux 专用测试库/目录 |
| verify_m01_failures.py | 失败注入和持久任务库；需先审核所有注入/清理目标 |
| verify_m01_legacy_downloads.py | SQLite/历史产物；需合成 fixture 与专用目录 |
| verify_m01_local_tasks.py | 本地服务/任务库；不得使用现存研究缓存 |
| verify_m01_native_io.py | 原生资源和输出 SQLite；先解决 Linux REAPER 构建/hash 与进程边界 |
| verify_m01_natural.py | 自然语料路径与基线；P11 首轮改用公开合成数据 |
| verify_m01_persistent.py | 数据库、原生/清理入口；需独立目标审阅 |
| verify_m01_segments.py | SQLite 输出与分段基线；先替换 Windows 原生执行依赖 |
| verify_m01_task_window.py | Qt/任务库/基线；桌面窗口证据不纳入本轮服务门 |
| verify_m01_web.py | Chrome/node/Windows 路径、PG/清理；禁止直接运行 |
| verify_m01_workspace.py | Qt 工作区；本轮只读，桌面另验 |
| verify_m02_m09_local.py | 本地服务/SQLite；合成数据独立宿主及平台执行器后复用 |
| verify_m02_m09_qt.py | Qt/本地基线；桌面另验 |
| verify_m02_m09_web.py | Windows 浏览器/PG/清理；禁止直接运行 |
| verify_m02_png_qt.py | Qt 实际整幅 PNG；后续模块验收复用，P11 不改绘图语义 |
| verify_m03_core.py | 科学 wheel/基线；需 Linux fingerprint，不能复用 Windows MKL 结论 |
| verify_m03_jobs.py | 兼容解释器、任务库、清理；先实现 Linux 执行器 |
| verify_m03_long_parity.py | Windows 解释器/受限进程与旧基准；重建 Linux 数值比较入口 |
| verify_m03_long_process.py | Windows Job Object/命名管道；需真实进程组等价用例 |
| verify_m03_long_qt.py | Qt/Windows 兼容解释器/任务库；桌面另验 |
| verify_m03_overview_qt.py | 同上，公共总览布局不属于当前改动 |
| verify_m03_qt.py | 同上，Linux 服务不能代替 Qt 功能证据 |
| verify_m03_web.py | Windows 浏览器/PG/清理；需 Linux 独立账号与库审阅 |
| verify_m04_core.py | 科学 wheel/导出 SQLite；Linux 构建差异单列 |
| verify_m04_jobs.py | Windows 兼容解释器/任务库/清理；先平台执行器 |
| verify_m04_limits.py | Windows 路径/任务库；迁移失败场景到 Linux cgroup 后验证 |
| verify_m04_server.py | Windows MKL/真实 PG/清理；不可盲指向服务器 |
| verify_m10_bundle.py | 资源打包检查，虽无直接风险命中也不证明 Linux 原生 ABI |
| verify_m10_geometry.py | 固定原生运行时/子进程；Linux ABI 属模块专项 |
| verify_m10_qt.py | Qt/设备页面；本轮不运行 |
| verify_m10_recording_features.py | Qt/录制；本轮不运行 |
| verify_m10_video_decode.py | 视频解码子进程；需平台 FFmpeg 来源与预算，属后续模块验收 |
| verify_m12_long_qt.py | Qt 与测试库；保留既有文件，不扩大到 Linux |
| verify_m12_qt.py | 已有未提交修改，Qt 与测试库；只读审查 |
| verify_m12_web.py | Windows 浏览器/PG/清理；需专用环境和目标审阅 |

额外接点：`ptb_api/server.py` 在配置 storage 后调用 `storage.recover()` 并启动 `run_cleanup`。即使构造账号对象不迁移数据库，启动整个服务仍有数据副作用，不能当作无条件只读盘点动作。本轮探针只对显式选择、已经运行的宿主发 GET，不自动启动它。

## remaining_risks / next_dependency

1. **M03/M04 原精确数值门未通过**。完整 P11-LINUX 不标完成。后续诊断与候选实测见 [数值兼容跟进](p11-numeric-compatibility.md)。追加证据已发现旧 Windows 默认 MKL 线程数与 Linux 单线程构成混杂因素，同机 Windows 单线程也会失败 8 项，因此上文原始失败不能全部归因于操作系统/BLAS 构建。需要通过兼容 runtime 原精确门，或经独立科学审阅明确新的跨平台等价标准；没有擅自更改标准。
2. `acoustic_executor`、`segmentation`、`spectrogram_preview`、`font_preflight`、`egg_runtime`、`lpc_runtime` 尚未完成 Linux 生产接入。只有参数表 stdio 路径通过，Linux REAPER 构建/来源/hash 仍缺。
3. capability 当前只是保守资格门。Linux 科学关闭，Windows M01 核查原生资源/版本；完整模块允许表、纯分段、M03/M04/M09、实际执行探测、资源额度与远程节点在线状态仍待接。版本/hash匹配不能代替科学结果。
4. WSL 的 user systemd/cgroup 未就绪，保留明确不可用；服务器的单任务组证据不能扩大为 WSL 进程预算或全机单槽资源准入。P11-PERF 应统一预览/计算/导出的跨进程准入，补混合负载和 30 分钟测试。
5. 完整科学依赖锁安装门、Matplotlib 字体发现/导出、真实浏览器 Linux 页面、数据库/账号/故障恢复、真实域名 HTTPS、生产服务均未验证。已有 Windows 模块状态不改。
6. P07-POLICY、P06-REMOTE、P04-UNIFY、EXE、部署未推进；本轮不把 1 GB/3 天写成已在运行库生效。下一依赖是数值运行时审阅及余下平台执行协议，详细顺序见本轮计划。

## 文档与收尾校验

最后的 `python scripts/validate_docs.py` 检查 680 文件、333 来源、41 任务，仍只有基线中两处 M10-R5 EXE 链接错误，返回 1；没有新增文档错误。日志与保护核对在 `docs-validation-final.txt`、`preservation-final.json`、`delivery-manifest.json`。定向 `git diff --check` 返回 0，15 个认领文件 UTF-8 读取及其中 Python 语法检查通过。未为消除旧链接错误重建 EXE，也不扩大为全库所有历史测试通过。
