# P11-LINUX 数值兼容跟进

2026-09-26。任务范围为 [P11 环境计划](../plans/2026-09-26-p11-environment-gate.md) 的数值门，承接 [Linux 基础报告](linux-environment.md)。本报告仅对实际公开合成输入、指定运行时和命令负责。

## 状态与归属

**本次数值诊断与候选实测已完成；P11-LINUX 数值门仍为 in_progress。三组 Linux 候选均为 23 passed / 19 failed，不能开放科学 capability。** 原科学源码、冻结 fixtures、精确断言、Windows 运行时与先前 P11 生产文件不改。没有修改容差或将固定中间结果实验接入生产。

工作前记录 `codex/v3-rebuild`、HEAD `9d0283ce135f95c21c31fcc807638e4a78a4c7c8`、1392 个已有文件摘要及已有 dirty 状态。原 M12、前端、统筹文档与前轮 P11 改动均属于本轮保护范围。此次认领：

- 新增 `scripts/diagnose_numeric_compatibility.py`、`tests/architecture/test_numeric_diagnostics.py`、本报告。
- 更新 `docs/plans/2026-09-26-p11-environment-gate.md`、`docs/testing/linux-environment.md`。
- 原始证据保存在忽略目录 `output/validation/p11-numeric-20260926/`，不把实验安装锁当作生产已通过锁。

## 实验与判据

原验收命令始终为：

```text
python -m pytest -c tests/pytest.ini tests/parity/test_egg_analysis.py tests/parity/test_lpc_spectrum.py -q --tb=short -p no:cacheprovider
```

独立诊断命令：

```text
python scripts/diagnose_numeric_compatibility.py capture --root . --output <新目录> --label <运行时标签>
python scripts/diagnose_numeric_compatibility.py compare --actual <实际目录> --reference <参考目录> --output <新JSON>
```

诊断捕获 534 个数值数组、48 个独立 None 字段、4 个预期 LPC 错误案例。涵盖 EGG 原始/预处理、detrend、两级滤波系数/初值/输出、8 变体的事件/CQ/SQ及各 ROI、Praat 时间/F0、逆滤波和 LPC 中间步骤。输入文件 SHA-256 必须一致，数组集合必须一致；精确比较含 shape、dtype、原始字节，另外报告 NaN/正负 Inf mask 与有限值误差。误差统计只描述差异，没有隐含通过容差。

`frozen_preprocessed` 将同一原基准预处理波形传给原事件代码，用于定位上游影响；它没有证明实际 Linux 预处理通过。诊断器禁止 object 数组摘要；Windows/Linux 输入文件键的目录分隔符归一化，文件内容/hash不变。上述语义有 4 项定向测试。

Linux 科学子进程通过 user systemd 临时 `ptb-p11-*` unit 启动，重依赖导入前检查 cgroup：MemoryMax=1,073,741,824、MemorySwapMax=0、CPUQuota=100%、TasksMax=64、RuntimeMaxSec=100。24 线程实验也不提高 CPU/内存预算。这是一个任务组的实验边界，不代表 P11-PERF 全机单槽已实现。

## 已验证：旧基线包含线程设置影响

Windows 兼容环境实际为 **PyPI NumPy 2.2.6/OpenBLAS + Conda SciPy 1.16.3/MKL 2025.3.0**，不能将整个环境笼统称作 Conda NumPy/MKL。Windows CPU 为 Core Ultra 9 275HX；原进程 `MKL_Get_Max_Threads()` 返回 24。Linux 服务器为 Xeon Platinum，2 vCPU；OS、CPU 和二进制构建均存在差异。

| Windows 实验 | 原 42 项结果 | 证据 |
| --- | --- | --- |
| 原兼容环境，默认线程 | 42 passed | windows-mkl-baseline.log/xml |
| PyPI NumPy/SciPy，默认线程 | 34 passed / 8 failed | windows-pypi-baseline.log/xml |
| 原兼容环境，OpenBLAS/OMP/MKL 全部 1 线程 | 34 passed / 8 failed | windows-mkl-threads1.log |
| 原兼容环境，OpenBLAS=1、OMP/MKL=24 | 42 passed | windows-mkl-threads24.log |

针对 EGG 变体 0 和 LPC default 的拆分实验：只设 OpenBLAS=1，两项通过；只设 MKL=1 或 OMP=1，EGG 失败、LPC 通过。见 thread-matrix.json 及三份 thread-*.log。默认与显式 24 线程的 534 数组全部逐字节相同；1 线程与默认相比 448 相同、86 不同。

因此，上一轮 Linux 23 passed / 19 failed 的实际失败证据仍有效，但它同时包含 Linux 单线程与旧 Windows 默认线程不一致的影响。不能据此把 19 项全部归为 Linux/OpenBLAS 的独立影响。线程控制在 Windows 同机上已经形成可重复的干预证据；尚未证明所有剩余差异来自哪个 CPU 指令或底层函数实现。

## 已验证：PyPI 候选差异的位置与影响

与 Windows 默认兼容环境比较，Windows PyPI 为 424 数组相同、110 不同；Linux PyPI 为 362 相同、172 不同。Linux 与 Windows PyPI 比较为 366 相同、168 不同。534 数组比原 42 项 pytest 覆盖更多中间字段，其计数不能等同测试通过率。

| 比较 / 步骤 | 实测最大绝对差或结果 |
| --- | --- |
| Windows PyPI 的 float32 detrend | 5.960464477539063e-8；3,493/35,280 有限值改变 |
| Windows PyPI 的滤波 a/b 与高低通初值 | 与 Windows MKL 默认精确相同 |
| Linux PyPI 的高通 a/b | 与 Windows MKL 默认精确相同 |
| Linux PyPI 高通 lfilter_zi | 3.2166444530190574e-7 |
| Linux PyPI Praat F0 | 3.362288225616794e-10 |
| Linux PyPI inverse auto | 1.1757261830780408e-13 |
| Linux PyPI LPC default 预加重 | 精确相同 |
| Linux PyPI LPC default Hamming 窗 | 1.1102230246251565e-16，当前分解中首个不同步骤 |
| 同案例自相关 / LPC 系数 | 2.220446049250313e-16 / 9.824849267481284e-13 |
| 同案例 freqz 实部 / 最终 dB | 1.0488692137045064e-9 / 6.30766550102635e-11 |

Windows 和 Linux PyPI 两组固定原预处理波形实验中，8 变体的全部 48 个事件/global CQ/SQ 字段都与 Windows 原环境精确相同，支持该公开样本的事件差异由上游预处理传入这一定位。

Linux PyPI 实际处理下，手动 prominence 的变体 0、2 各有 8/287 个 GCI 移动一采样点（44.1 kHz 下约 22.676 µs）。全部变体的 global CQ/SQ 最大绝对差分别为 0.0036314612584355532、0.007288816758746819。原工作台默认 slope/scale/auto-prominence 对应变体 3，96 个 GCI 全部精确相同，GOI 最大差 4.126548733794644e-11 秒、CQ 5.013245152341028e-9、SQ 3.2312307329807055e-8。手动参数最坏值与默认参数结论分别报告；都没有转为新的科学容差。

## Linux 候选来源与安装边界

依据 [micromamba 官方安装文档](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)，只提取官方 conda-forge Linux micromamba 2.9.0-0 的自包含二进制，在 `/home/admin/ptb-p11-20260926/numeric-followup/` 放置工具、独立 prefix 和缓存；没有 shell init、sudo、apt 或系统 Python 替换。下载资产与官方 API 摘要核对。Conda [BLAS 变体机制](https://conda-forge.org/docs/maintainer/knowledge_base/) 仅用于构造待验证候选，不作为精确兼容证明。

固定解析输入为 python=3.11.14、scipy=1.16.3、numpy=2.2.6、libblas 的 MKL 变体、mkl=2025.3.0，使用 conda-forge 严格优先级和独立 root；将解析得到的 41 包、230,865,469 字节 URL/SHA-256 写成 explicit 文件后安装。主要 Linux 构建为 SciPy py311hbe70eeb_2、NumPy py311h5d046bc_0、MKL h0e700b2_463、libblas 3.11.0 build 5。它们与 Windows 构建号不同，不能称为相同二进制环境。

第一组使用 Conda NumPy+SciPy/MKL，第二组仅在新候选 prefix 内增加 PyPI NumPy 2.2.6 独立 overlay，并由实验进程 PYTHONPATH 选择，匹配 Windows 混合来源形式。两组均保持 Parselmouth 0.4.7、同一已安装自有 core wheel；PyPI 安装使用带 hash 的最小依赖锁。不会覆盖旧 API 环境或 Windows 环境。

## Linux 候选实际结果

| 候选 | 原测试 | 与 Windows 默认的 534 数组比较 | 独立双轮 |
| --- | --- | --- | --- |
| Conda NumPy + SciPy/MKL，1 线程 | 23 passed / 19 failed | 361 相同 / 173 不同 | 534 全部精确相同 |
| PyPI NumPy overlay + SciPy/MKL，1 线程 | 23 passed / 19 failed | 362 相同 / 172 不同 | 534 全部精确相同 |
| 同一混合候选，OMP/MKL=24、OpenBLAS=1 | 23 passed / 19 failed | 362 相同 / 172 不同 | 534 全部精确相同 |

三组失败集合相同：8 个 EGG 变体、1 个 Praat、2 个 inverse、8 个 LPC。不存在测试收集错误或跳过。原测试的两个重复时间警告保留；诊断构建配置提示可选 pyyaml 缺失，未为了该展示提示增加依赖。三个实验内部的 None 字段和预期错误同样一致。混合候选的 1/24 线程采集也全部精确相同，不能将 Windows 的线程效应直接外推到此 Linux 硬件/构建。

六次采集实际 cgroup memory.peak 快照为 229,404,672–237,592,576 字节（约 219–227 MiB），memory.max=1 GiB、cpu.max=`100000 100000`、pids.max=64。快照限定当前小型公开 fixture，不能作为自然长录音或并发容量结论。所有采集返回 0，三个原测试返回 1；完整命令和状态见 candidate/candidate-results.json。最后 owned-units.log 为 0 loaded units。

实际 imports 证明混合组 NumPy 来自专用 overlay，SciPy/core 来自新 prefix。18 份 Linux EGG/LPC 已安装源码与 Windows 兼容环境、工作区源码 SHA-256 一致。实际映射包含 libmkl_rt、libmkl_core、libmkl_avx512；混合组另含 PyPI NumPy 的 libscipy_openblas64。原采集的 library_hashes 因路径包含 conda-mkl 而额外收录了该 prefix 其他映射库，数值数组不受影响；最终诊断脚本已把筛选收窄到文件名。原始采集和执行版本 runner-v2.tar 保留，不能将最终脚本 hash 冒充原执行 hash。

41 份 Conda 下载归档逐份实际 SHA-256 与解析元数据一致，记录于 candidate/verified-conda-archives.json。服务器证据包 `candidate-evidence.tar` SHA-256 为 `5699704a04d8a55070d8f74a9331c62a77a8e83c12e59399828af34869b0ddcc`；回收后对 46 份内部证据逐份校验通过。初始安装助手的两个锁/缓存路径已按服务器实际目录修正，最终本地与服务器 candidate-run.py 摘要一致。旧传输包作为过程证据保留。

### 候选对科学结果的实际影响

Conda 组 EGG 预处理最大差降至 3.48279570114296e-7，混合组为 4.0385149546739996e-7，均未精确一致。手动变体 0、2 的 GCI 分别有 6/287（Conda）或 5/287（混合）移动一采样点；默认变体 3 的 96 个 GCI 仍全部精确相同，混合组默认 CQ/SQ 最大差为 3.4020659900324546e-9 / 2.1927633431229054e-8。三组固定原预处理波形后的 48 个事件/global CQ/SQ 字段全部精确相同。

Conda 组 LPC default dB 最大差为 2.1540991212987137e-11，混合组为 6.30766550102635e-11。Hamming 窗、高通初值和 Praat 差异在候选内依然存在。混合 Linux 与同为 1 线程的 Windows 比较仍有 168/534 数组不同，EGG detrend 与滤波均不同。**匹配包版本、BLAS 家族和线程设置仍不足以复现原 Windows 字节基线。** 现有证据定位到具体运算层，但尚未分离 CPU 指令选择、编译器/libm、BLAS 构建等剩余因素，不把它们任一项写成已确认单一根因。

## 未完成项与下一依赖

1. Linux 兼容运行时仍未通过原科学门。后续优先围绕现有三个最小差异点（detrend、lfilter_zi、Hamming）做 CPU/构建受控对照，避免无依据地继续堆叠完整环境。Praat/逆滤波也仍需独立解释。
2. 如果跨平台目标改为科学等价，需要独立审阅连续值误差、离散事件时间/数量、CQ/SQ 与边界输入标准，扩充公开/授权自然语料后再批准。当前报告没有批准新标准、更新 fixtures 或放宽断言。
3. `egg_runtime`/`lpc_runtime` 等 Linux 生产执行器接入、字体/导出、真实账号任务、长录音/多任务资源验收仍未完成，按原 P11 后续顺序推进；本轮没有扩大 capability。
4. 新候选及缓存只保留在任务用户目录，没有生产服务、系统配置、数据库迁移、密钥修改、push、EXE、P07-POLICY、远程节点或 HTTPS 部署改动。

## 收尾证据

诊断器定向测试 4 passed。Windows 原 42 项在默认和显式 24 线程下通过。三组候选相对 Windows 的全部数组 NaN/正负 Inf mask 一致。文件保护记录确认 1392 个已有文件中 1390 个逐字节保留，仅本轮两份认领文档变化，无缺失或无关变化。新增文件仅上文列出的三份。

`validate_docs.py` 检查 682 文件、333 来源、41 任务，返回 1，仅保留两处既有 M10-R5 EXE 链接错误，无新增错误。读取该校验的子进程输出时曾遇 Windows 默认编码与 UTF-8 不一致，已给子进程显式设置 PYTHONUTF8=1 后重新核实，无源文件编码改动。定向 diff/空白/UTF-8/Python 语法检查通过。文件保护、文档校验和最终交付摘要见本证据目录 preservation-final.json、docs-validation-final.log、delivery-manifest.json。
