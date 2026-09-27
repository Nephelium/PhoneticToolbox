# M06 语音合成迁移验收

2026-09-27。**M06 整体 in_progress；Windows 开发态功能已完成定向验证，Linux 能力关闭。** 已完成六组功能、正式任务/文件接线与 UI，不将本轮扩大为生产部署、远程实机、真实声卡或任意长度输入通过。

## 平台与能力状态

| 范围 | 状态 | 实际证据与边界 |
| --- | --- | --- |
| Windows 科学核心 | verified | V2 独立双轮 167 数组/991816 值一致；源码 28 项、安装 wheel 28 项通过 |
| Windows 正式宿主 | verified | LocalService 真 HTTP/持久队列/Windows 有界子进程/工件哈希；实际 Qt QWebChannel + 构建前端 + 原生目录保存 |
| Windows Chrome UI | verified | 正式 AppShell/desktop adapter/TaskBridge/LocalService；仅 QWebChannel 传输由测试通道替代，八组流程通过 |
| Windows 服务存储 | verified（新建隔离 PG） | 双账号与工程隔离、真实合成输出配额核算、1 GB 满额拒绝、3 天新结果、下载不续期、cancel/fencing；现存库未操作 |
| Linux 核心精确门 | in_progress / closed | NInfer 专用环境 21 passed / 7 failed；失败保留原 exact 断言，逐字段如下 |
| Linux 任务门 | blocked / closed | 公共 posix 资源边界真实返回 linux_process_boundary_unavailable，无替代无界启动 |
| 云端/实验室远程 | planned / closed | 只提供固定 child、能力判断、输入/内存/时间预算；未创建远程协议、未验证远程实机 |
| 统一 UI | verified（Windows 范围） | 公共 Frame/Toolbar/Section/Status、波形/播放器/任务/关闭；浅深主题与 900×700 布局检查，IPA 使用统一 Doulos SIL |

## 可重复命令和结果

工作目录为仓库根。Windows `PYTHONPATH` 明确指向 `packages/phonetic_core/src;backend/src;desktop/src;scripts` 的绝对路径。解释器 `.venv/m09-ui/Scripts/python.exe`（CPython 3.11.14，NumPy2.2.6、SciPy1.16.3、Parselmouth0.4.7、pandas2.3.3、soundfile0.13.1）。以下均实际执行，未运行不存在的 test:e2e。

| 命令 | 结果 |
| --- | --- |
| `python tests/support/m06_baseline.py`，另一次 capture 输出到 `output/validation/m06/v2-second` | 两次独立 V2 expected 完全相同；V2 源文件最终 hash 保持 |
| `python -m pytest -c tests/pytest.ini tests/parity/test_speech_synthesis.py backend/tests/test_m06.py backend/tests/test_p07_policy_wiring.py backend/tests/test_storage_policy.py backend/tests/test_p11_capabilities.py backend/tests/test_p11_task_runtime.py tests/contracts -q -p no:cacheprovider --junitxml=output/validation/m06/windows-final.xml` | 169 passed；原 V2 HNR 空 slice 警告及现有 TestClient deprecation 保留 |
| `python -m pytest -c tests/pytest.ini backend/tests/test_m06_pg.py -q -p no:cacheprovider`，`PTB_POLICY_FRESH_PG=1` | 2 passed；复用受控新集群测试 fixture，只在全新随机测试库建表，未连现存 DB |
| `python scripts/verify_m06_wiring.py` | 生成、合成、提取、提取曲线再次提交及错误任务通过真实 HTTP |
| `node tests/e2e/m06-host.cjs` | 八组流程通过，真实生成/播放节点/导出/提取、曲线绘制与缩放、旧结果/迟到/取消、关闭保存失败恢复 |
| `python scripts/verify_m06_qt.py` | 实际 Qt offscreen 宿主生成、合成、原生目录 WAV/CSV/JSON 保存通过 |
| `npm --prefix frontend run typecheck` / `test` / `build` | 类型通过；145 passed；构建通过，已有较大 chunk 警告未隐藏 |
| `python scripts/generate_contracts.py --check`；`npm --prefix frontend run contracts:check` / `ui-data:check` | 无契约/来源目录漂移 |
| `python scripts/generate_m06_catalog.py --check` | 23 参数/默认值/元音/五类预设与核心一致 |
| `.venv/v3-dev/Scripts/python.exe -m build --wheel --no-isolation --outdir output/validation/m06/wheel-final packages/phonetic_core` + `pip --no-deps --target output/validation/m06/installed-core-final` + 源码同一核心专项 | wheel 安装目录 28 passed，MIT LICENSE/NOTICE/presets 已打包；未替换现有环境的项目包 |

Qt offscreen 出现 GPU context 回退日志，实际页面/链路通过。此测试未观察物理声卡声压输出、多屏 DPI 或 GPU 渲染性能；Chrome 公共播放是 WebAudio 操作证据。无新 EXE。

## 六组正常与边界

详细操作和源码定位见 [来源映射](../modules/evidence/M06-source-map.md)。

- F01：应用时长/F0、淡入淡出/平滑与 IPA；0/NaN/反向范围/非法字符明确拒绝，未应用的文本编辑阻止提交，并使在途任务失效。
- F02：常数 Override、分段曲线、Shift 手绘与缩放/重置；越界、覆盖锁定、清空确认、图外坐标夹取。CSV 保留被覆盖的原始轨迹。
- F03：波形与原版 scipy 语谱、四窗长、同步时间范围；无结果空态、源音频独立预览、切换不计算。深色 SVG 刻度填色修正后加入实际颜色断言。
- F04：生成曲线、合成、试听、导出独立；无结果禁用、失败保留旧音频、实际 WAV 回读、原生导出三工件。同一随机种子下的 WAV PCM 与 V2 soundfile 输出精确比较。
- F05：合成公开源音频提取 23 项、提取后再提交；坏 WAV 和预算超界失败；预设确认/取消、元音表、原辅音移除说明。
- F06：完整 CSV/JSON 与旧 CSV 读取、非法表头/字段/值；真实下载再导入、快照往返与 Override 原曲线；帮助、草稿失败阻止关闭、恢复。

额外：实际保存的 m06.ptb.json 可重新导入；提取后源文件二次读取校验 audio hash，变化则拒绝应用参数。参数修改后旧结果状态，真实任务在途修改后的迟到拒绝，取消后的零部分发布，旧 generation 不能写入；输入/输出 SHA-256；真实 PG owner/project、额度 writer、回收和最终政策。

## 正式接线与政策

完整参数是已有 table/CSV 资源，源音频是 audio 资源，任务请求仅含 ID/hash/action。解决 16 KB DB snapshot 限制，无 schema 改动，也没有扩大公共 JSON 入口限制。上传与结果均走公共 owner/project/配额系统，worker 输入读校验、ManagedScratch、硬进程预算、心跳租约和 generation fencing 不绕过。

三个 operation action 都是新计算，`speech_synthesis` 纳入公共 independent-result 分类；新结果发布时独立最多 259200 秒，输入原截止保留。M06 没有新增复制任务，公共 M08 copy/ZIP 继承期限的规则保留。桌面目录保存与服务器内容下载不创建续期；完整参数的上传是本次计算的独立输入资源。

新 PG 测试实际证明输入剩余一小时仍不决定新合成结果的三天截止，下载前后截止不变，二账号不能读取结果，满额写入失败，取消后的迟到输出被拒绝。现存旧政策库仍须由 A 的授权流程迁移，本轮没有执行该迁移。

## Windows 资源实测与准入

真实公共 Windows 受限子进程（含科学 import），进程组限制 1,000,000,000 字节、120 秒，任务 deadline300秒；输入请求16MB、child bundle24MB、输出预留64MB。下表总耗时含准备/发布，峰值为公共 Windows Job 记录的内存提交峰值。源音频为公开正弦合成，未读研究语料。未据此推断 Linux/云服务器耗时或物理工作集。

| 操作 | 输入秒/采样率 | 耗时 s | 峰值字节 | 输出字节 | 结果 |
| --- | --- | --- | --- | --- | --- |
| synthesize | 0.1 / 16000 | 1.891 | 247685120 | 18858 | succeeded  |
| synthesize | 10 / 16000 | 4.469 | 449724416 | 879160 | succeeded  |
| synthesize | 10 / 48000 | 5.859 | 635252736 | 1519159 | succeeded  |
| extract | 10 / 48000 | 11.047 | 296505344 | 2706130 | succeeded  |
| synthesize | 10.01 / 16000 | 0.906 | 150228992 | — | failed m06_admission_budget |

仅开放不超过 10 秒且 480000 样本的 Windows 任务，提取源 WAV<=8MB、参数 CSV<=2MB、最多8声道。核心配置文件可表示0.1–100秒/480万样本，但这不构成正式任务准入。复杂轨迹、极端参数仍受1GB/120秒硬限制，未声称穷举所有组合或生产并发。

## Linux 逐字段差异

用户本轮明确允许创建 M06 专用 Linux 环境，位置 `/home/ninfer/ptb-m06-20260927`，复用既有 CPython3.11.14。相同 NumPy/SciPy/Parselmouth 版本；复用 M01 提取所需 pandas2.3.3 与 V2 导出所需 soundfile0.13.1，在专属环境安装。来源为官方 PyPI wheels及已有项目 wheel缓存；本轮下载13份的官方文件 SHA-256 已核验。pip 下载 soundfile 时连带下载的 NumPy2.4.6 **没有安装**，实际始终2.2.6。未改系统 Python、WSL 配置、M08/M14 环境。

执行 `wsl -d NInfer -- env PYTHONPATH=<core-src>:<backend-src> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/ninfer/ptb-m06-20260927/bin/python -B -m pytest -c /dev/null <repo>/tests/parity/test_speech_synthesis.py -q -p no:cacheprovider --tb=short --junitxml=<repo>/output/validation/m06/linux-exact.xml` 得到 **21 passed / 7 failed**。环境变量只作用于本次测试。最终核心 wheel 也安装到 M06 专属 Linux 环境，使用该安装包重复 exact 门仍为21 passed / 7 failed，见 `output/validation/m06/linux-wheel-exact.xml`。使用同解释器运行 `tests/support/m06_linux_probe.py` 输出完整219字段报告，197字段 exact。以下单位继承原参数，不使用统一容差：

| 合成样例 | 浮点波形最大绝对差 | PCM16 | 23 项输入数组 |
| --- | --- | --- | --- |
| 0 | 3.33066907388e-16 | exact | 全部 exact |
| 1 | 1.34459110512e-12 | exact | 全部 exact |
| 2 | 1.03850261723e-12 | exact | 全部 exact |
| 3 | 8.881784197e-16 | exact | 全部 exact |
| 4 | 1.19859677739e-12 | exact | 全部 exact |
| 5 | 7.77156117238e-16 | exact | 全部 exact |

提取逐字段，所有字段时间坐标、shape、NaN/非有限掩码相同：

| 参数 | 数值最大绝对差 | exact |
| --- | --- | --- |
| F0 | 1.20811876059e-08 | False |
| AV | 0 | True |
| Jitter | 0 | True |
| Shimmer | 0 | True |
| SHR | 2.47198095327e-17 | False |
| HNR | 1.00328634289e-11 | False |
| Slope | 0 | True |
| H1H2 | 0 | True |
| F1 | 1.9025662823e-08 | False |
| F2 | 0 | True |
| F3 | 6.14577402303e-08 | False |
| F4 | 0 | True |
| F5 | 0 | True |
| A1 | 0 | True |
| A2 | 0 | True |
| A3 | 0 | True |
| A4 | 0 | True |
| A5 | 0 | True |
| B1 | 4.22161576807e-08 | False |
| B2 | 1.11931015567e-08 | False |
| B3 | 1.05316644294e-08 | False |
| B4 | 0 | True |
| B5 | 0 | True |

这些差异与跨平台二进制数值行为相容，但根因未进一步隔离到具体函数/库构建，不能凭幅度小宣称过门。没有修改 expected、放宽容差或更换算法。正式 Linux capability 始终 false，提交返回 m06_platform_unverified。公共资源边界实际探测为 linux_process_boundary_unavailable，未生成合格运行时 receipt，也未绕过门运行持久科学任务。

## 共享模块回归与开发入口

M08 两文件回归和 M06 PG 两项在 m09-ui 环境通过。第一次将 M14 一起放入 m09-ui 检查时，4 项因该环境无 python-docx 失败（混合运行 37 passed / 4 failed），没有改 M14 或安装文档依赖到 M06。改用 M14 已有专属 `.venv/m14` 环境后原14项全部通过，见 `output/validation/m06/m14-existing-runtime.xml`。这是环境范围差异，混合运行报告保留为失败记录，未伪报全绿。

新增 `scripts/Start-M06-Workbench.ps1` 只为已有共享宿主设置本次进程的源码路径，复用 m09-ui 科学环境与 start_m01_workbench.py，退出恢复环境变量。没有覆盖已安装旧 wheel 或修改 M08/M14 启动入口。

## 证据入口与精确交接

- [独立 fixture](../../tests/fixtures/m06/v2.json)，[第二轮捕获](../../output/validation/m06/v2-second/v2.json)，[源码测试](../../output/validation/m06/windows-final.xml)，[wheel测试](../../output/validation/m06/windows-wheel-final.xml)，[新 PG 测试](../../output/validation/m06/pg.xml)。
- [正式 HTTP](../../output/validation/m06/host/b62bd38256864ca0ae1b83795e619438/report.json)，[最终 Chrome 流程](../../output/validation/m06/host/84a3fedadea34b60a4a3f17e8ea05065/browser-report.json)，[最终 Qt](../../output/validation/m06/host/b52360371ebf4d0e97b4a3daa8a2b2bb/qt-report.json)。
- [资源原始值](../../output/validation/m06/host/87133dfa3c314586b9f55b9545b16eb8/resources.json)，[Linux exact 原始失败](../../output/validation/m06/linux-exact.xml)，[Linux全字段](../../output/validation/m06/linux-fields.json)，[wheel来源](../../output/validation/m06/linux-wheel-provenance.json)。
- [浅色](../../output/validation/m06/host/84a3fedadea34b60a4a3f17e8ea05065/light.png)、[深色](../../output/validation/m06/host/84a3fedadea34b60a4a3f17e8ea05065/dark.png)、[窄窗](../../output/validation/m06/host/84a3fedadea34b60a4a3f17e8ea05065/compact.png)、[实际语谱](../../output/validation/m06/host/84a3fedadea34b60a4a3f17e8ea05065/spectrum.png)。截图已实际查看，深色刻度已复核。

后续 B/Linux 接手只需围绕 `native/linux_runtime.py` 固定 m06 入口、`m06_task.capability`、`m06_executor` 的公共 collector、已登记科学版本/输入预算建立平台资格；先隔离上述7项数值差异，再由统筹确认逐字段等价标准或匹配构建。当前禁止只把 sys.platform 判断移除来开放能力。随后验证合格 Linux 系统的真实任务取消/超时/峰值/配额/fencing，再验远程领取/上传/发布。M06 不定义节点身份、租约或远程传输协议。

共享接线已串行完成：AppShell、research/desktop adapters、TaskBridge、main/jobs/job_models、store/executor、acoustic/local文件、固定进程入口、生成契约与存储 operation 分类。未删除 M08/M14 入口和结果分支。没有 Git push、现存库 DDL、公开部署、系统依赖修改、V2/语料修改或 EXE。
