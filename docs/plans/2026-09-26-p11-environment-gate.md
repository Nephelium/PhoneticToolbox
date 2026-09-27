# P11-ENV → P11-LINUX：执行边界与阶段门

**2026-09-26 后续授权与接线：** 用户已接受分参数小浮点误差和最多一采样点事件偏移，明确授权 LPC → EGG → M01 任务接线。下文“未通过精确门不得进入执行器”属于历史停止点；现行实施与结果见 [任务链路计划](2026-09-26-p11-task-flow.md)、[专项验收](../testing/p11-task-flow.md)。原 Windows 精确断言、Linux 失败记录与既有环境保留。

2026-09-26。任务来源：[服务器统筹任务卡](2026-09-26-server-coordination.md)。实际结果见[Linux 验证报告](../testing/linux-environment.md)。

## 状态与授权变化

P11-ENV 的原生 Linux 环境、wheel 安装、合成 API/静态资源与显式中文/IPA 字体绘制门已完成，范围分别为现有 WSL 和授权实际服务器。P11-LINUX 为 `in_progress`：通用 Linux 进程组边界、参数表接入和 capability 保守关闭已验证，M03/M04 精确基线未通过，完整科学执行器未接通。

早期新建 `PTB-P11-Ubuntu2404` 的提案已 **Superseded**。井井明确选择先复用现有 WSL，随后授权 SSH 连接并追加服务器读写权限。本轮使用现有 NInfer 中独立目录 `/home/ninfer/ptb-p11-20260926`，以及服务器 `/home/admin/ptb-p11-20260926`。没有安装新发行版、修改系统 Python、sudo/apt、编辑 `.wslconfig`、配置系统服务或开启公网端口。

项目级 Python/venv/wheel 和合成测试在上述目录执行。pip 下载另使用用户级缓存，缓存未删除。读写授权用于这项隔离测试，没有扩大到生产部署、现存数据库、系统配置、凭据变更或用户原始语料。

## 文件归属

开始时逐项核对根规则、backend/tests/docs 规则、统筹计划和 Git 差异。原有 M12、前端公共组件、桌面、统筹文档、台账、ADR、contracts、source-registry 改动不归本任务。

本轮修改既有文件仅：

- `backend/src/ptb_api/main.py`：capability 不再仅由 batches 存在推断 M01 可运行。
- `backend/src/ptb_worker/parameter_preview.py`：Linux 参数读取走新的受限 stdio 子进程，Windows 原路径保留。

新增：`native/{posix,process,capabilities}.py`、三份 `backend/tests/test_p11_*.py`、`tests/support/p11_process_fixture.py`、`tests/architecture/test_linux_environment_probe.py`、`scripts/verify_linux_{environment,smoke,science}.py`、本计划及 Linux 验证报告。完整清单和摘要见报告的 changed_files。

其他任务卡允许的执行器文件本轮保持原样。新增依赖锁和原始日志在忽略证据目录，不改根依赖锁、数据库迁移、协议模型或生成契约。总台账和 ADR 最终汇总留给统筹者串行处理。

## 已实施的进程边界决策（供统筹合并 ADR）

1. 生产入口只能选择固定模块。当前 stdio allowlist 仅 `ptb_worker.parameter_preview`，不接收客户端命令、解释器或 shell。底层 `run_bounded` 接受可信宿主 argv，性质与 Windows 原生 OwnedProcess 相同。
2. Linux 使用现有 user systemd 的唯一临时 `ptb-p11-<uuid>.service`，先设置 MemoryMax、MemorySwapMax=0、CPUQuota=100%、TasksMax=64、KillMode=control-group、OOMPolicy=kill，再启动目标。没有 user manager 时明确不可用，无无限制回退。
3. 输入/输出字节、墙钟超时、取消和组内内存均受限。只停止该任务拥有的 unit；异常后核实组已空/消失以及 unit 已停止，清除自己的 failed unit。Windows Job Object 和原有 MKL 校验未修改。
4. 单个任务组的 512 MB/1 GiB 证明不代表整个服务器总预算。API 父进程输入缓冲、跨 worker 并发和统一单槽准入属于 P11-PERF，当前没有声称实现。
5. Linux 科学 capability 保持关闭，给出 `linux_scientific_runtime_unverified`。Windows M01 额外核对登记 REAPER 的绝对路径/hash和已安装包版本。节点在线/远程能力来自 P06-REMOTE，将来接真实状态，当前不虚构。

## Linux 环境和锁提案

- CPython 3.11.14 为 python-build-standalone Linux x86_64 独立解释器，官方发布资产 digest 与本地下载一致，原系统 Python 保留。
- `linux-api-candidate.in/.lock` 固定后端依赖、构建/测试和字体 raster 工具的全部 34 项传递依赖及 hash。已在 WSL、服务器原生 venv 安装并执行 pip check。
- `linux-table.in/.lock` 固定既有 optional extra 的 openpyxl 3.1.5、et-xmlfile 2.0.0，服务器实际表格测试通过。
- `linux-science-candidate.in/.lock` 为完整科学候选，镜像缺 pyparsing 3.3.3，未完整安装，不能作为生产运行时验收。
- 从官方解析锁提取的 `linux-science-numeric.lock` 只固定 NumPy 2.2.6、SciPy 1.16.3、Parselmouth 0.4.7。实际 hash 校验安装后，纯数值门得到 23 passed / 19 failed。完整依赖与原始 fingerprint 在报告中逐项列出。
- 当前候选使用 OpenBLAS，不覆盖原 Windows MKL 结论；同版本不同构建的差异已经实测。原科学源码、fixtures、容差和 Windows locks 不改。

锁文件、官方解释器来源/digest、自有 wheel SHA-256、安装路径和构建输入清单都保留在 `output/validation/p11-env-20260926/`。后续将通过数值门的 Linux lock 正式合入依赖管理，当前保持候选身份。

## 剩余实施顺序与退出条件

1. **数值门**：调查 Linux OpenBLAS 与原 Windows MKL 的滤波/事件/F0/LPC 差异，另建 Linux 兼容构建候选进行同一精确基线，不直接放宽容差。诊断报告的重复性不等于与原基准等价。
2. **协议接入**：在数值与 runtime 门通过后迁移 `acoustic_executor`、`segmentation`、`egg_runtime`、`lpc_runtime` 的命名管道边界；补 Linux REAPER 来源/构建/hash。公共语谱与后台字体也仍需独立接入和运行证据。当前参数表成功不证明这些模块成功。
3. **真实 capability**：与模块允许表、资源准入、执行端状态对齐，补 M03/M04/M09 和纯分段路径；当前 Windows M01 资格检查仍非完整执行证明，远程节点未接入。
4. **实际服务联合门**：独立账号/任务库与故障场景、完整字体发现和真实浏览器页面另验。实际数据库建库/迁移、系统配置或部署仍须相应明确授权。本轮未启动配置了 storage 的服务，避免 recover/cleanup 碰现存数据。
5. 后续每次更换候选先固定版本/hash/fingerprint，再跑 Windows 原回归和 Linux 定向测试。P11-PERF 的共享单槽/混合负载、P06-REMOTE、P07-POLICY、统一 UI 和 HTTPS 部署分属后续任务，不在本轮代做。

## 数值兼容跟进（井井已授权继续）

2026-09-26 本次只处理上一轮 19 项数值失败与一个 Linux 兼容候选。先在原 Windows MKL / Windows PyPI / Linux PyPI 环境捕获相同公开合成输入的中间值，再验证 Linux Conda/MKL 候选。候选使用任务用户目录内的新环境和缓存，不替换既有 API/Windows/NInfer 环境。禁止修改算法、fixtures、原精确断言、Windows MKL 校验或开放科学 capability。

文件边界：新增 `scripts/diagnose_numeric_compatibility.py`、必要的诊断器定向测试与 `docs/testing/p11-numeric-compatibility.md`，更新本计划及 `linux-environment.md` 的后续状态。临时安装器、版本/hash锁、快照、数组、日志均放忽略证据目录。其他上一轮 15 个文件和用户已有改动保留。

验收：原 42 项 M03/M04 精确基线在原 Windows MKL 和 Linux 候选分别执行；采集归一化、detrend、滤波系数/结果、事件/CQ/SQ、Praat、LPC 中间数组与实际加载库。输入摘要相同，独立轮次可复现。候选未通过时报告剩余数值及语义影响，不将诊断用固定中间输入实验用于生产处理。

本次跟进已完成诊断与候选实测，详见 [数值兼容报告](../testing/p11-numeric-compatibility.md)。Windows 原默认/显式 24 线程均为 42 passed，同一环境 1 线程为 34 passed / 8 failed，证明旧跨平台比较混入线程因素。Linux Conda/MKL、混合 PyPI NumPy+Conda SciPy/MKL、混合 24 线程三组均为 23 passed / 19 failed，各组两轮 534 数组精确复现，但仍未与原 Windows 基线一致。41 份 Conda 归档 hash 和回收证据包校验通过。P11-LINUX 数值门仍 in_progress，生产科学能力继续关闭，不进入下一执行器接入门。
