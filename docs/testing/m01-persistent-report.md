# M01-F2 持久任务与双端实际操作

2026-09-10，Windows 定向验收。**F2 的已列明本机持久流程已实现；完整 M01/P08 仍为 in_progress，进入 G 联合收口。** 本文区分实际操作、故障注入和未测范围，不以窗口截图代替科学或数据库证据。

## 用户现在可以做什么

- 桌面选择 WAV 目录、关联 TextGrid/安全唇形 JSON，保存 80 参数和 14 设置的批次快照；“开始全列表分析”处理整个列表。
- 在左列勾选切分范围、选择同名 TextGrid 层，保存实际 WAV 片段；未勾选时处理当前音频。可选择同步切分相同原音频的最近一次完整参数结果，保留原帧网格和来源时间；没有父结果时明确仅切分 WAV。
- 查看持久处理记录、部分成功/失败/中断、取消后续处理和单项重试。取消后尚未开始的文件仍为“未开始”，完整批次必须每项成功。
- 桌面结果先进入受控本地缓存，再保存至已授权目录；同名同内容复用，同名不同内容另起文件名。保存后刷新输入目录；网页保存为本账号项目资产并可下载。应用重启后需重新选择桌面结果目录授权。

开发启动使用 [Start-M01-Workbench.ps1](../../scripts/Start-M01-Workbench.ps1)，它只打开现有已审阅 SQLite，新建/复用 `output/validation/m01/workbench-cache-*`；不执行 DDL、不启动 PG、不要求登录。仍是本机开发入口，完整 EXE/安装器属于后续发行阶段。

## 数据和执行边界

005 两份 SQL 按井井对具体审阅的“好，继续”授权实际应用。证据 `output/validation/m01/persistent-7d7a6a1a5818455da1de88cbadcfb33b/report.json`：SQLite 的 3 张旧表、PG 的 13 张旧表逐行保留，新表/外键/版本核验通过。SQL hash 与 [005审阅](m01-migration-review.md)相同；后续验证不重复 DDL。

新批次接纳与未开始容量预留在同一事务，沿用 P06 每账号/全局运行槽、租约与代际校验。PG 输出、请求临时文件和 native 暂存先经过 P07 配额预留；完整产物与任务成功同事务公开。参数结果从成功起至多 7 天，切分不延长输入截止。桌面以 SQLite 成功状态加完整 manifest 为可见门，缓存 metadata 在 SQL 提交前即使已落盘也不可下载。

桌面原文件通过原生目录能力读取，复制后的输入以 hash 固定；renderer 不提供任意磁盘路径。目录导出每个新文件经过尺寸/hash/fsync，再使用 Windows 不覆盖 rename。中途失败只回收本次创建且身份仍匹配的文件。多文件导出不是跨文件系统断电事务，已经在受控缓存成功的结果可重新保存。

科学计算在内存/时间受限的自有 Windows Job 中运行；调度与 API 不导入 NumPy/SciPy。实际排查曾发现调度进程卡在 NumPy 动态库加载，堆栈证据为 `output/validation/m01/worker-stack.log`；已移除该调度路径的科学库导入，并新增子进程 import 回归。这里只确认 PhoneticToolbox 调度阻塞已在复验中消失，不将其认定为此前 Codex 退出根因。

## 已执行验证

工程解释器 `.venv/v3-dev` 保持无 NumPy；API/desktop wheel 安装至项目内 `.venv/m01-ui`。本轮没有新增第三方依赖、升级 v2 环境或修改全局设置。

```powershell
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/run_m01_validation.py --approved-m01-schema-and-synthetic-tests --verify-persistent --verify-web
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/run_m01_validation.py --approved-m01-schema-and-synthetic-tests --verify-web --verify-legacy-files
& '.venv/m01-ui/Scripts/python.exe' -X utf8 scripts/verify_m01_local_tasks.py --approved-m01-synthetic-tests
& '.venv/m01-ui/Scripts/python.exe' -X utf8 scripts/verify_m01_task_window.py
& '.venv/m01-ui/Scripts/python.exe' -X utf8 -m pytest -c tests/pytest.ini backend/tests desktop/tests tests/contracts tests/security -q
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
```

| 范围 | 实际证据与结论 |
| --- | --- |
| 两个真实数据库 | 005 已应用，原行保留；仅自行启动的固定测试 PG 被正常停止 |
| PG 17 项 | 顺序、全请求幂等、跨 owner 拒绝；17 份实际切片均与原双声道整数样本逐点相同 |
| PG 异常 | 并发同键仅一批、真实子进程启动后取消并核对所属句柄退出、租约过期拒绝旧提交、显式新任务重试；中间文件坏层不阻止其他项 |
| 最终PG与网页复验 | `persistent-1b3a7e4481b448df9eaf0607c0b24eeb`：另含计算期间实际删除输入、认领前到期、普通P06实例也计入17项预留、真实网页分析/切分TTL；全部通过且PG正常停止 |
| 成组失败 | 第二结果预留注入 quota 错误：无 manifest，全部本次输出物理删除后释放占用；注入不同于真实填满 5 GB |
| SQLite / native | `local-tasks-838ddfe4ce244230bb4206e6dc9ad032/report.json`：实际 REAPER、双声道 WAV/参数切分、幂等不覆盖导出、第二下载失败回收、服务重启和缓存结果再保存 |
| 原 v2 科研基线 | 同一次真实持久计算的 GUI-ALL：全部目录参数、列序、时间轴、有限值和非有限 mask 与独立冻结原 v2 捕获一致；没有修改基线或容差 |
| 实际 Qt | `task-window-8a13a23f113b48ea9c9604902503fb76/report.json`：7 步实际点击与原生目录选择、完整计算/同步参数切分/文件刷新、浅深截图、缩小窗口和正常关闭 |
| 实际网页 | `persistent-1c0b653726414cccafeb33595e0ba57a/web-report.json`：真实 PG/worker、登录、全参数分析、XLSX 下载、刷新恢复、父结果切分、换账号清空与越权拒绝；pageerror 为空 |
| 原 P07 回归 | 同目录运行报告及 `output/validation/p07/jobs-validation.json`：11 组任务/ZIP/配额/到期/取消/旧 worker/重试通过；只清理该验证生成的文件 |
| 定向测试 | 237 passed，两个既有 Starlette/anyio 弃用提示；前端 16 passed，typecheck/build 通过 |

最终共同预览布局另以原 E 的14步Qt脚本回归：`workspace-12ea69804a7c445b833d920826bf74b3/report.json`通过（含语谱图/时间轴初始可见、17项列表及正常关闭）。最后的架构、契约生成检查、来源生成检查、文档链接检查和`git diff --check`均通过。[机器证据](../modules/evidence/M01-persistent.json)记录源码/wheel hash及各验证结果。

首次验证中的测试问题原样保留：空 cache 目录初始化遗漏、local JSON 边界误拦音频、测试读取 wire 字段错误、将纯正弦 REAPER 结果预设成 200 Hz、Qt 对话框路径分隔符比较、结果完成与下载按钮渲染的等待竞态。已按实际契约修正测试；REAPER 改用原 v2 冻结全参数基线逐项验证，没有放宽数值检查。迁移只读保护测试不得与实际 SQLite 写入测试并行；分开运行后通过。

## 保留范围与下一项

M01-G 继续逐项审阅 39 项验收和说明书，尤其是任意旧版 XLSX/SQLite 的导入兼容、自然语料、所有窗口缩放和科学异常说明。当前父参数来源为本任务存储内完整分析生成的 `.ptb.json`；不能声称已支持任意历史文件自动配对。

科学解码仍限 2,000,000 采样值（多声道合计）、单次 240 秒、成组结果 64 MB；长音频的 64 MB/3200万采样值显示和切分预算是另一条路径。必要时先按 TextGrid 切分，不能把显示降采样说成已经解决任意长度全参数计算。未验收硬断电、全平台桌面、完整 EXE/生产负载；没有公开发布或 push。

本轮 Git 根核对为 v3 目录。`output/validation/m01/f2-preservation.json`的7项均与F1记录相同：v2 HEAD/index/status、427份基线、包元数据和环境路径。保留 v2、旧未提交文件和冻结基线；没有把 Users 文件夹纳入提交。所有浏览器测试使用独立 Chrome/Qt，未操作或关闭 Codex 内置浏览器，也不作“绝不再闪退”的保证。
