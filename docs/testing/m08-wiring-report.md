# M08 正式接线与限定验收

2026-09-27。Windows 正式开发宿主链路 **verified（仅下述范围）**。完整 M08 仍为 **in_progress**，真实 PostgreSQL 网页验收及 Linux 正式宿主受阻。未改算法、原默认值或跨平台精确断言。此前独立模块证据保留在 [原报告](m08-report.md)。

## 实现与范围

- `M08Port → desktop/server transport → 认证 API → 持久任务 → 公共受限子进程 → fenced writer → manifest → 历史` 已接通。AppShell 使用既有 M08 页面，无另造页面或计算实现。
- 正式 operation 为 `pitch_manipulation`，schema 为 `m08/1`。实际 WAV、MP3、FLAC 经 Praat 解码，选文件、文件读取和任务入站均支持三格式。测试 MP3/FLAC 由已有 FFmpeg 生成，产品没有新增 FFmpeg 运行依赖。
- preview 使用解码后的 float64 WAV，合成仍由原 handler 生成，再用原 Praat WAV 保存得到 PCM16；历史 F0 来自该 PCM16 回读。原科学核心四文件和 handler 与上轮交付 SHA-256 一致。
- 计算子进程逐文件发出有长度、hash、结束标记的受限流。公共 writer 先分配额度、逐块写入并 seal，完整结果经公共 generation/worker fencing 才发布。取消、超时、崩溃、写失败不发布残缺 manifest。
- 输入 owner/project/hash/expiry 走既有正式 adapter。复用额度预留、临时回收和最终提交，不改 quota/storage 政策源、ProjectStorage 或迁移。
- 保存是另一条持久复制任务，PCM 字节不重算，继承原结果截止。管理只收明确结果 ID，并核对 owner/project/source hash。受管编号在事务内分配，本地导出使用授权目录、扫描最大尾号及独占创建。现有文件不覆盖；目录实际编号回写受管显示名。
- 明确删除仅处理选定 ID；删除和重命名不扫描前缀推断目标。部分删除失败逐 ID 返回；本地副本操作失败不报整批成功。下载重新核对内容 hash。
- Windows 仅在真实 adapter、子进程入口及匹配 NumPy/Parselmouth/SciPy 版本可用时开放 M08。Linux 当前既不公布能力，也拒绝直接科学任务提交，不借 API 绕过未通过的精确门。

## changed_files

本轮认领文件的完整路径及 SHA-256：[`delivery-manifest.json`](../../output/validation/m08-wiring/delivery-manifest.json)。这是相对本轮的归属清单，不能把工作区全部 Git diff 归入 M08。

| 文件组 | 本轮改动 |
| --- | --- |
| 新增 M08 后端 | `m08_task.py`、`m08_child.py`、`m08_stream.py`、`m08_results.py` |
| 公共后端接点 | `jobs.py`、`job_models.py`、`main.py`、`m08_models.py`、`acoustic_executor.py`、`executor.py`、`acoustic_files.py`、`local_acoustic_files.py`、`store.py`、`process_entry.py`、`acoustic_errors.py` |
| 公共进程接点 | `native/reaper.py` 流式回调与证据、`native/windows.py` 实际 Job 峰值与清理确认、`native/posix.py` 流式预算、`native/linux_runtime.py` 固定入口。原资源隔离边界保留 |
| 桌面 | 新增 `m08_bridge.py`，更新 `task_bridge.py`、`file_provider.py`；新增源码开发入口 `scripts/Start-M08-Workbench.ps1` |
| 前端 | 新增 `platform/m08.ts`，更新 `platform/research.ts`、`platform/desktop.ts`、`app/AppShell.vue`。原模块页面与状态算法不改 |
| 生成契约 | `openapi.json`、`generated/api.ts`、`schemas/resultmanifestenvelope.json`、`schemas/acousticfilemanifest.json`。最后一项刷新并行 P07 模型字段，非本任务制定政策 |
| 验证 | `test_m08_wiring.py`、三份 verify 脚本、PG 只读 gate 脚本、fixture 脚本、真实 host bridge、`m08-host.html`、`m08-host.cjs` |
| 文档 | 本报告及 M08 计划、接线合同、原报告入口、使用说明 |

根 AGENTS/README、总台账、政策源与现存库均未由本任务修改。其他任务的工作保留。原页面/核心/handler 的保持证据和可比较的并行文件变化单列在交付清单。

## 实际验证

环境为 Windows，项目既有 `.venv/m09-ui`，NumPy 2.2.6、SciPy 1.16.3、Parselmouth 0.4.7；源码 PYTHONPATH 为三个绝对目录。没有安装新依赖。数据库使用已有 P06 测试库的 SQLite backup 副本与新合成缓存，没有 DDL。删除验证全部发生于本轮合成结果。

| 验证 | 结果与证据 |
| --- | --- |
| Python 集中回归 | **154 passed**，M08 原 handler/科学精确对照、正式接线、P11 进程和 capability、contracts。`regression-final.xml`。2 个既有 FastAPI/Starlette 弃用 warning |
| 资源设置后的精确复验 | **41 passed**，11 项正式接线与 30 项原科学精确对照，`resource-single-thread.xml`。未扩大容差 |
| 最后能力检查 | **14 passed**，`final-followup.xml`；另新增直接 Linux API 拒绝用例 **1 passed / 11 deselected**。记录属性的 xunit2 warning 不影响断言，资源专门记录使用 legacy XML |
| HTTP 正式宿主 | **8 组通过**，三格式预览、实际变速变调、幂等、两个持久保存副本/PCM hash、历史 rename/delete、重启回读和预留/临时回收。`cb4fdd234cbf452fbe60c158c779de53/report.json` |
| Chrome 正式 adapter | **5 组通过，pageerror=0**，`e4497806fa2c445a906d0e5897ca31ff/browser-report.json`。真实 TaskBridge/LocalService/worker，仅 QWebChannel 传输与目录选择替换为测试传输 |
| 原生 Qt 最终 bundle | **通过**，`38cb604750174126964df946193dee15/qt-report.json`。实际 Workbench、QWebChannel、生产 bundle、原 F0、合成、保存、历史。测试 offscreen，目录选择限定合成目录 |
| 前端 | **137 passed**；vue-tsc、Vite build 通过。最终 bundle 已包含 AppShell M08。仍有既有 M13 大 chunk 提示 |
| 生成契约 | Python generator `--check`、frontend `contracts:check` 通过 |

收尾：44 个认领文件 UTF-8/语法检查与定向 `git diff --check` 通过；11 个原 M08 核心、handler、页面/状态与冻结测试文件哈希保持不变。`python scripts/validate_docs.py` 检查 761 文件、333 来源、41 任务，退出码 1，仅两条既有 M10-R5 EXE 缺链，无新增 M08 缺链。未为此生成 EXE 或改根 README。

正式测试还覆盖错误 owner/project/hash、原生运行中取消后新任务恢复、原生崩溃、极短受控超时、结果写入失败回收、旧 worker generation 拒绝发布、明确 ID/名字冲突、排队保存编号及实际四组合 PCM16 文件。资源/超时注入仅在测试中进行。

Chrome 验证了目录已有 `_9.wav` 时生成 `_10.wav`、历史显示实际名称，随后选中结果 rename/delete 保留该未选文件和原始输入；实际下载落盘，草稿写失败保留标签和 dirty，成功后关闭重开恢复参数。浅/深主题及失败截图均保存。原 Qt 两次测试脚本定位/时序失败保留；最终修正测试等待后通过，未降低产品约束。offscreen GPU 告警保留于 `qt-final-render.log`，不据此声称显卡、声卡或多屏验证通过。

截图：[Chrome 浅色](../../output/validation/m08-wiring/e4497806fa2c445a906d0e5897ca31ff/light.png)、[深色](../../output/validation/m08-wiring/e4497806fa2c445a906d0e5897ca31ff/dark.png)、[保存失败](../../output/validation/m08-wiring/e4497806fa2c445a906d0e5897ca31ff/close-failed.png)、[Qt 实际合成](../../output/validation/m08-wiring/38cb604750174126964df946193dee15/qt.png)。

主要复验命令（PowerShell）：

```powershell
$env:PYTHONPATH='D:/PhoneticToolbox/PhoneticToolbox_v3/packages/phonetic_core/src;D:/PhoneticToolbox/PhoneticToolbox_v3/backend/src;D:/PhoneticToolbox/PhoneticToolbox_v3/desktop/src'
.venv/m09-ui/Scripts/python.exe scripts/prepare_m08_fixtures.py
.venv/m09-ui/Scripts/python.exe scripts/verify_m08_wiring.py
.venv/m09-ui/Scripts/python.exe -m pytest -c tests/pytest.ini backend/tests/test_m08_jobs.py backend/tests/test_m08_wiring.py backend/tests/test_p11_task_runtime.py backend/tests/test_p11_capabilities.py tests/parity/test_pitch_manipulation.py tests/contracts -q -p no:cacheprovider
npm --prefix frontend run test
npm --prefix frontend run typecheck
npm --prefix frontend run build
node tests/e2e/m08-host.cjs
.venv/m09-ui/Scripts/python.exe scripts/verify_m08_qt.py
.venv/m09-ui/Scripts/python.exe scripts/generate_contracts.py --check
npm --prefix frontend run contracts:check
.venv/m09-ui/Scripts/python.exe scripts/verify_m08_pg_gate.py
```

154 项集中回归之后新增了直接 Linux 拒绝测试，当前上述集合会多出 1 项。不能将不同轮次的通过数相加当独立覆盖数。

## Windows / Linux / 资源 / UI 状态

| 维度 | 状态及边界 |
| --- | --- |
| Windows 正式开发宿主 | **verified**，限定 SQLite 副本、短公开合成、HTTP/Chrome/实际 Qt。旧安装 wheel 和 EXE 未更新。源码入口使用统一宿主，不建另一台服务 |
| Windows 网页 PG | **blocked**，只读 gate 实查旧 policy=1、运行时要求=2。未运行写入/迁移，未伪装双账号、满额下载删除、实际服务器过期联合通过 |
| Linux 科学精确门 | **in_progress**，现有 NInfer 独立 venv 原精确断言 **25 passed / 5 failed**；本轮未改 fixture/assertion |
| Linux 正式宿主 | **blocked**，现有环境无 `systemd-run`/`systemctl`，不能验证 P11 cgroup 受限链路；未安装/改系统，M08 能力继续关闭 |
| 资源 | **部分 verified**，Windows 子进程 Job 内存硬预算 1,000,000,000 字节、120 秒、逐文件输出总流 64,000,000 字节；故障/取消/超时进程组回收已测。未进行长录音/极限组合/10 人吞吐或云端 4 GiB 压测 |
| 公共 UI | **verified**，正式 AppShell、参数/F0/波形/历史、草稿、保存失败、标签关闭、主题；没有重做模块页面 |
| 远程节点/生产/EXE | 未实施，非本轮交付 |

Windows 资源证据是 Job 的 **峰值提交内存**，不是 RSS。默认科学库线程数时曾达约 930–934 MB，现仅在 M08 子进程导入科学库前固定 OMP/OPENBLAS/MKL 为 1，不改全局环境。单线程后 1 秒 16 kHz 合成预览实测：WAV 157,712,384 字节/2.141 秒，MP3 157,220,864 字节/2.110 秒，FLAC 159,694,848 字节/1.875 秒。时间为子进程段，清理通过 Job ActiveProcesses==0 确认。不能外推为最长材料预算。

Linux 实际命令：

```powershell
wsl -d NInfer -- sh -lc 'cd /home/ninfer/ptb-m08-20260926/project && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/ninfer/ptb-m08-20260926/bin/python -m pytest -c /dev/null tests/parity/test_pitch_manipulation.py -q -p no:cacheprovider --junitxml=/mnt/d/PhoneticToolbox/PhoneticToolbox_v3/output/validation/m08-wiring/linux-exact.xml'
```

原 F0 最大绝对差 1.084364953385375e-8 Hz，4 组 float64 合成波形最大差 5.551115123125783e-17。此为精确门失败，不能按“很小”放宽标准。上轮 Linux 同平台原 V2 64/64 精确证据仍仅支持其原范围，不替代这道跨平台门。

## 已知限制与精确交接

1. **P07-POLICY 串行接线**：本轮结束后释放 M08 公共文件占用，接手前再读 diff。特别注意 `snapshot.saved_copy=true`、`source_ref`、`copy_result` 的 M08 保存任务只有字节复制。P07 计划将 `pitch_manipulation` 纳入独立结果到期集合时，必须按 snapshot 排除 saved_copy，继续继承源结果截止。只判断 operation 会让复制续期，禁止照此直接接线。公共 `files.py` 当前未由 M08 修改，政策 owner 须补此回归。M08 科学新结果与复制结果分别验收。
2. 实际 PG gate 文件 `pg-gate-998045119954497ebc1b3f3013a20cdf/gate.json`：只读、schema_applied=[]、write_gate_ready=false。后续由已授权的 P07/统筹落实政策与数据库门，再做真实双账号、额度并发、到期、满额下载删除和迟到 worker 联验。本任务没有迁移权限，不执行该步骤。
3. **P11-PERF 接点**：公共 collector 新增可选 `on_chunk`，累计总量仍计入限制，Windows evidence 增加实际提交内存/进程回收。固定 m08 入口只提供适配，未建立 Linux M08 验收 receipt。后续须兼容这些流式与 fencing 语义，不能退回聚合整个批次。
4. **M14/公共 UI 接点**：`AppShell.vue`、`platform/research.ts`、`platform/desktop.ts`、`task_bridge.py`、`jobs.py`、`job_models.py`、`main.py` 及生成契约可在本轮结束后串行最小接线。保留 M08 可选 port、关闭失败路径和 operation/manifest 联合类型；重新生成契约，不能覆盖 P07 同时已有的字段。未自行派发或实施 M14。
5. 本地导出副本的目录授权/文件身份关联当前限**本次宿主会话**；跨会话历史持久保存的是受管结果。重启后管理旧受管结果不会重新扫描并删除外部目录文件。当前会话外部文件被改动即拒绝管理。跨会话外部目录重新授权/关联未实现，不能声称已验。
6. 本地导出失败时受管保存副本仍可能已成功，不把其隐式删除。提示失败后可再次保存；复制任务通用 retry 返回明确错误，须重新保存，科学任务仍走通用 retry。不能把已保存受管副本与外部导出文件数量混为一谈。

下一步为 P07 合并政策接点与明确数据库授权后的网页验收，以及 P11 提供合格 Linux 运行边界并解决精确门。此报告可供统筹更新台账；本任务不更新根文档/总台账、不 push、不部署、不生成 EXE。
