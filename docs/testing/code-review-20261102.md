# 2026-11-02 外部代码审查复核与修复

输入为井井提供的桌面文件 `PhoneticToolbox_v3_代码审查报告.md`，报告日期 2026-11-01。报告中的建议作为待核证据，不作为执行指令。当前工作树包含此前三语等未提交改动，本轮仅增量修改，未提交或推送。

## 结论与范围

报告的大部分具体故障判断成立，但性能限制不都属于功能错误，部分科研判断仍缺参考输出证据。Windows、macOS、Web 共用源码已同步修正，Mac 平台适配的进程/会话逻辑也已修改；没有 Mac 实机或打包证据。本轮仅后台测试与前端本地构建，没有启动可见应用、截图或线上部署。

井井已确认三项方案：修正 Intensity/无效 Iseli 带宽并标记数值版本，新增默认关闭的无引用导入副本清理，更新签名分阶段实现且暂不配置正式公钥。2026-11-03 井井明确授权继续 A5 专项改造，已完成共享实例锁、账户/资产锁和有界连接复用，并通过独立 PostgreSQL 回归，见文末补充。

## 逐项核查

| 报告项 | 判断与处理 | 主要源码 |
| --- | --- | --- |
| N1 | 成立。高频搜索区间完全超出 Nyquist 时返回 NaN，保持有效频段数值和首帧窗口契约。覆盖 8/11.025/16 kHz。 | core/acoustic/spectral_batch.py |
| N2 | 排队消耗运行预算成立。初次认领才开始原运行预算，排队单独限 24 小时；远程网络重试不续期。单账号串行是资源公平策略，保留。 | worker/store.py、remote_scheduler.py |
| A1 | 成立。1000 限额只计活动任务和未展开批次项，终态历史保留但不再耗名额。 | worker/store.py |
| A2 | 成立。元数据保存压缩 deleted 墓碑，容量排除墓碑，旧任务引用仍返回不可用。没有修改现存开发缓存。 | worker/local_acoustic_files.py |
| A3 | 成立。单任务异常经租约栅栏提交失败，存储异常重试；本机 API 监管自己的 worker，重启失败退避。存储完全不可用时靠租约恢复。 | worker/executor.py、api/cli.py |
| A4 | 成立。实现 Ed25519 验签和离线签名工具，签名覆盖整份发行信息；配置公钥后不允许 GitHub digest 回退。信任根当前为空，尚未启用签名保护。 | desktop/update_signatures.py、updates.py、release/sign_update_manifest.py |
| A5 | 成立，2026-11-03 经授权完成。普通读写共享实例 OS/PG 锁，按账户/资产协调，维护与跨资源发布保留独占锁。账户/工作区/任务/存储复用每进程每 DSN 上限 8 的连接池，worker 块 I/O 使用同一分级锁。鉴权仍逐块执行，未声称生产容量达标。 | api/storage.py、storage_concurrency.py、postgres_pool.py、account_store.py、workspaces.py；worker/store.py、files.py |
| A6 | 成立。每块不再重写元数据，未封存字节数在锁内从文件大小恢复；封存输入先 fsync 再发布 ready，结果块原数据持久化保留。长输入哈希移出全局锁并复核文件身份。 | worker/local_acoustic_files.py |
| N3 | 成立。空队列先轻量只读探测，不解析资产 JSON、不进写事务；轮询退避至 1 秒。 | worker/store.py、executor.py |
| N4 | 成立。独立默认关闭选项只清理应用复制件；保护活动任务、未完成批次、存活结果和完整父结果输入链。原始目录不在清理范围。 | worker/local_retention.py、api/local_retention.py、StorageSettings.vue |
| N5 | 成立。新锁写入 PID，确认进程退出后按文件身份恢复。活进程、无法确认所有者及旧版空锁保留，旧空锁不自动猜测删除。 | desktop/annotation.py |
| N6 | 部分成立，报告的“即 CI”不准确。当前 SQ 为 `(去接触时长−接触时长)/接触总时长`，与常见 Orlikoff CI 的符号相反；说明书已说明该定义。修正误导性函数注释，保留列名及科学数值。 | core/egg/metrics.py |
| N7 | 默认启用属既有产品设定，界面显示一致；过期的 opt-in 模块注释已修正，不改默认策略。 | worker/local_retention.py |
| A7 | 成立。目录列表、读文件和 TextGrid 解析改走异步 task 通道；仍有有界 base64 的内存开销，启动 hello 的小型探测仍同步，未宣称所有 GUI 同步工作已消除。 | desktop/task_bridge.py、frontend/platform/desktop.ts |
| A8 | 成立。文件上传/下载独立有界通道，不占交互与恢复通道。保留数量限制，不无限提高服务器并发。 | api/web_admission.py |
| A9 | 请求体占计算槽成立，已改为读完有界请求体才取槽。单计算槽仍为资源保护策略。 | api/preview.py、jobs.py |
| A10 | 成立。账号限流键改为账号+IP，成功登录重置该对的失败计数，保留全 IP 上限。未改数据库 schema。 | api/account_store.py、auth.py |
| A12 | 常数偏差成立且已获科研确认。新值同均方输入降低约 5.0309 dB，数字静音 NaN。窗口/网格不变，不宣称完全复制 Praat Intensity 对象算法。 | core/acoustic/energy.py、models/acoustic.py、worker/acoustic_result.py |
| A13 | 保持待验证。镜像代码存在值得核对的索引，但尚无可信原始 VoiceSauce 输出证明正确替代值，未改 HNR。官网 ZIP URL 在本次读取中仅返回 Git LFS 指针，不能当成已验证源码压缩包。 | core/acoustic/hnr.py 未改 |
| A15 | 单实例属显式容量限制；其他用户挤掉现有会话的问题已在 Windows/Mac 拒绝，Linux 原有保护保留。未建立无界每用户原生进程。 | worker/egg_interactive.py、native/darwin_egg.py |
| A16 | 成立。本机 EGG 上传磁盘检查/写入移到线程池。 | api/preview.py |
| A11/N16 | 底层 IRAPT 吞资源异常成立，MemoryError/BackendAborted 直接抛出；报告指称 services/acoustic.py 同样吞掉它们不成立，该层已有 raise_stage_failure。 | core/acoustic/jitter_shimmer.py |
| A17 | 代码路径成立。宿主退出保留中断恢复，不再把 pipeline_check 标作用户取消；历史 17 条取消记录不能单凭数量归因。 | worker/executor.py |
| A18 | 指定路径成立。terminate 超时后 kill、有限等待并关闭管道，覆盖 worker、本机服务、EGG 进程。 | worker/executor.py、egg_interactive.py、desktop/local_service.py |
| A19 | 成立。SQLite 配 storage_root 返回明确配置错误，说明应使用 local_files_root。 | worker/cli.py |
| A20 | 成立。Windows 读取 ACL、校验当前所有者及授权主体；不修改实际权限或读取输出配置内容。策略测试通过，未修改任何现存配置 ACL。 | api/private_config.py、server.py |
| A21 | 保持待验证。当前从左到右合并峰的实现与按高度选择的算法不等价，但尚未完成原始 VoiceSauce/MATLAB 科研对照，不据此更换 CPP 算法。 | core/acoustic/cpp.py 未改 |
| A22 | 成立。队列等待超时使用非空原生错误提示。 | desktop/vocal_tract/client.py |
| A23 | 成立。task 通道固定日志记录异常类型，不记录私有路径、请求或凭据。 | desktop/host.py |
| A24 | 成立。文档隐藏暂停反馈轮询，可见时恢复，卸载移除监听。 | frontend/components/FeedbackThread.vue |
| A25 | 成立。维护循环单次普通异常记录并继续，取消仍传播。 | api/main.py |
| N8 | 成立且已获确认。无效 Iseli 频率/带宽/采样率返回 NaN，合法公式不变。既有“缺失共振峰时跳过部分校正”分支未另行重定义。 | core/acoustic/corrections.py |
| N9 | 成立。修正 Hanning 注释为实际矩形窗，没有偷偷换窗。 | core/acoustic/spectral_batch.py |
| N10 | 重解析点拒绝是现有路径安全边界；未取得 OneDrive 实机兼容性证据，不放宽全部重解析点检查。 | 原安全检查保留 |
| N11 | 竞态可用受控管道时序复现。退出被观察到后继续排空，覆盖空读后出现最后字节的回归。 | worker/native/reaper.py |
| N12 | 全局锁内身份检查和进程等待成立，已移出；保留两会话/单计算槽的容量策略。 | worker/vocal_sessions.py |
| N13 | 成立。清理失效保留记录，状态无变化时不落盘。 | worker/local_retention.py |
| N14 | 成立。超大 deferred snapshot 留失败子任务，不回滚整个认领；提交共用配置超过 16 KiB 明确返回 413，原批次请求保留。 | worker/acoustic_batches.py |
| N15 | 成立。安装器等待限 1800 秒，返回 INSTALL_TIMEOUT，不强杀正在写入的安装器。 | desktop/update_apply.py |

## 非功能错误与未采纳建议

`legacy_parameters.py` 经 Python AST 实际解析成功，不能据报告中的 ruff 一条结论认定语法错误。`tdklatt.py` 的默认列表在现有路径只读后转换为新数组，未见跨调用修改，未借审查改变科学参数。文件体积、if 分派及未使用导入是维护建议，本轮不做无目的大规模重构。`backend/build/lib` 等旧构建目录未作为源码入口，未删除。native_messages_data.py 为生成文件，不能手工拆改。

SQ 判断参考 [Orlikoff 原文摘要](https://pubmed.ncbi.nlm.nih.gov/1749236/)与[接触时段定义](https://pmc.ncbi.nlm.nih.gov/articles/PMC6966776/)。Intensity 的依据和兼容边界见[科研 ADR](../decisions/ADR-acoustic-review-20261102.md)。更新签名使用 [cryptography 官方 Ed25519 API](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/)，依赖来源已登记，正式随包许可审计留待启用。

## 验证与证据

验证根目录为 `output/validation/review-20261102/`，每次运行保留 pytest.txt、run.json、scratch 清单和清理回执。使用项目 m14 Python、明确 v3 源码/PYTHONPATH 与 tests/support；`-o addopts=` 仅关闭根目录遗留 v2 覆盖率参数，未放宽断言、数值容差或添加跳过。Qt 使用 offscreen。

| 最终有效检查 | 结果 | 运行目录或命令 |
| --- | --- | --- |
| 工作队列、文件存储、清理、认证、准入、桌面锁、安装等待及科研解析值 | 116 passed | final-focused-afa63fa732074a0a9c600108a640d512 |
| 更新原有行为与新增签名、篡改/未知密钥/缺签拒绝 | 70 passed | signatures-8ce4353f2a044aa0ad2d1b0bb23e1dd1 |
| 远程协议、真实声道会话 | 53 passed | integration-corrected-73a14ba37c4f4e3a833197bc2e969c66 |
| 待处理父结果依赖保护、毒快照、管道退出竞态 | 33 passed | edge-final-fa62aa13ed6c4f639e9860d83d289d84 |
| 输入封存持久化、真实有界 M01 工作进程与结果读回 | 2 passed | seal-and-owned-job-97bbd3fe7acf446182318e9efe23cca4 |
| EGG 原始 CSV/频谱/逆滤波/导出冻结基准 | 27 passed | egg-bootstrap-607aa5ca84b84303b316c5ad3d4ed250 |
| 声学实际 REAPER/Python 结果往返、M06 初始化、M01 有界算法等 | 已通过所在集成运行的相应用例 | integration-cba08a1ce8504a039810b1717f710a48，113 passed；其余 2 项失败后来在上列 corrected 运行通过 |
| 前端可见性/更新/生命周期 | 21 passed（4+17） | node --test visible-polling-sfc.test.ts；updates.test.ts、release-lifecycle.test.ts |
| 前端类型与构建 | 通过，存在既有大 chunk 提示 | npm run typecheck；npm run build:workbench |
| OpenAPI/TS、三语、UI 来源生成一致性 | 通过，11348 messages、pendingReviews=0 | generate_contracts.py --check；generate-contracts.mjs --check；generate-i18n.mjs --check；generate-ui-data.mjs --check |

上述不同运行包含重复用例，不相加宣称唯一测试总数。前期失败记录保留：缺 tests/support 或遗留覆盖率参数导致收集失败、测试错误预期、主环境直接执行 EGG 导致 runtime_mismatch、声道缺 PTB_M05_PYTHON。EGG 直接 conda 调用还发生一次 MKL 原生异常；按生产 bootstrap 的 PATH、DLL 目录及单线程变量重跑后 27 项通过。未通过安装全局依赖或降低基准解决环境问题。

11 月 2 日初轮没有 PostgreSQL 实库并发验证，11 月 3 日已补充，见下节。生产容量、Mac、Safari、实体音频设备及最终 EXE 未新增验证。API worker 自动重启已实现，尚无单独完整崩溃重启集成验收；异常任务继续执行有回归证据。Windows ACL 有策略单测，尚无多 Windows 用户身份的端到端验收。

## 临时产物与后续

常规 scratch 由任务 finalizer 清理。regression-two-05bb87a732ac4310bd74e2295155d205 中的测试联接使自动清理保守保留；人工确认两处联接及目标均在该运行 scratch 内后，清理动作仍被工具自动审批拒绝，仅返回 blocked by policy。未改写命令绕过，目录保留，见该目录 cleanup.json。不影响源码测试结果，但清理尚未完成。

A5 已按 11 月 3 日授权完成，设计见 [存储并发 ADR](../decisions/ADR-storage-concurrency-20261103.md)。没有现存库 schema 变更，也未去掉鉴权或锁检查。

签名正式启用按已批准的分阶段方案后续确认公钥与发布流程。HNR、CPP、OneDrive 均保持待验证，不能归类为已修复。Mac 源码随本轮修改，未来有环境后再做原生回归和打包。

## 2026-11-03 A5 专项实库验收

实现新增 `api/postgres_pool.py` 与 `api/storage_concurrency.py`，调整 Storage、PostgresAccountStore、WorkspaceService、PostgresJobStore、FilePipeline 及 API lifespan。OS 锁在 Windows 使用 LockFileEx 共享/独占模式，POSIX 使用 flock；相同实例锁文件和 PG 锁 ID 与既有独占调用协调。scope 锁按实例→账户→资产顺序，新增磁盘预留另序列化，worker 先结束 owner 查询事务再取得存储锁并重新 fence。恢复/回收/跨资产发布继续独占，任务调度写事务仍短时串行，不等于所有请求无锁。

连接池每进程每 DSN 最多 8 条，等待 5 秒，SQL 10 秒/锁 5 秒超时保持。最多登记 4 个 DSN，仅淘汰无活动借用的池；关闭中的活动池保持登记直至归还，不能并发另建同 DSN 池突破上限。连接归还 rollback+DISCARD ALL，禁用自动 prepared statement 缓存。身份、workspace、资源检查和逐块重新鉴权保持，不跨网络 send 持有资源。复用已有 psycopg，无新增全局或环境依赖。

此次实库测试发现并修复此前 A10 成功登录重置计数为 0 与数据库 `attempts > 0` 约束冲突的问题，改为删除该账户/IP 的临时失败计数，其他 IP 桶保持。另一个旧测试预期与 HEAD 已提交的策略冲突：已预留额度允许结算，即使管理员之后降限。按实际既有策略更新测试，同时断言新增预留仍被拒绝、账本总量与原到期时间保持，没有放宽配额保证。新增事务回滚测试的第一版用了不合法的 quota_bytes=1，已改为合法项目名称变更后故意触发除零，验证真正回滚。

执行方式：m14 Python，显式 backend/core/scripts/tests/support PYTHONPATH，`PTB_POLICY_FRESH_PG=1`，`QT_QPA_PLATFORM=offscreen`，pytest `-q -o addopts= --basetemp=<owned scratch>`。`web_postgres_support.cluster` 使用本地 PostgreSQL 17.11-3、随机回环端口、独立 initdb、max_connections=20、shared_buffers=16MB。测试迁移只作用于新建随机库；全程 CREATE_NO_WINDOW，没有接触在线服务或现存研究库。

| 检查 | 结果及证据目录（均在本复核 validation 根下） |
| --- | --- |
| 跨账户读写、同账户最后字节配额争用、维护排他、重入块、资源到期、实际发布/取消栅栏、登录重置、连接重用 | `a5-postgres-final-71293355d193447ba80aef23fdba06ea`：26 passed；剩余 1 项事务回滚测试夹具错误后在下一行通过 |
| 事务回滚不污染下次借用、独立进程跨账户锁、下载中途撤销真实会话 | `a5-fences-57e7f6f06a2148ba82393b91476a8ec5`：3 passed |
| 池关闭排空不重建重复池、连接上限/等待超时、8 账户并发复验 | `a5-pool-lifecycle-1db1e47fb20144fea56c75ed47332e1a`：3 passed（其中 2 项与前面重复） |
| 工作区刷新/多设备隔离、过期、退出/worker fencing、结果期限、心跳关闭竞态、预览释放 | `a5-workspaces-e3b025ec133d4fd198f190554bc510b9`：6 passed |
| HTTP 认证、下载逐块边界、Web 准入及本机非 PG 回归 | `a5-boundaries-122760e1ecab46a6b9e6759cbcc66308`：26 passed |

合计 36 个不同实库用例得到通过证据，另 26 个定向用例通过。没有重跑前端构建或科学算法全套，因为本次未改这些层。受影响 Python AST 和 git diff --check 通过。

并发样本：8 个独立账户同时上传，每份 53,248 字节，再并发读取 80 次；实际新建连接 8 条，峰值 8 条。数据字节与 owner 隔离、账本总数均校验。精确耗时见 final 与 pool-lifecycle 运行的 concurrency.json。该样本证明复用和并发边界，不是生产服务器容量或旧版速度的对照基准。

五轮 fresh PostgreSQL 集群均正常停止，新 scratch 均有 deleted 清理回执。此前 11 月 2 日被工具拒绝清理的联接目录仍按原决定保留，没有重试绕过。Mac/Linux 仅共享源码覆盖，本次 OS 锁实测平台为 Windows，没有 Mac 打包或生产部署。

## 2026-11-03 3.1.3 文案与当前源码复验

按井井指定的 shaozi-writing-style 技能整理 `release/notes/3.1.3.md`，包括当前未发布的三语功能及审查修复。文案 SHA-256：`b5b4c03759c0e6801e22d7479b77c0967f4e92856c3a8de286bb607f673c8360`。等待井井审阅，未改变正式版本、更新渠道或发布任何文件。

- `node frontend/tests/updates-browser.mjs --notes-file release/notes/3.1.3.md`：独立 headless Chrome 真实更新弹窗7项通过，浅深色、窄窗口及150%缩放可读。证据 `output/validation/updates-ui/check-14ee0868-d343-40f8-bbcd-58bda5965bb9/report.json`；显示夹具的版本/包大小不代表正式发行元数据。
- 前端全部 `frontend/tests/*.test.ts` 使用 Node `--test --test-concurrency=2`：875 passed，0 failed/skipped。证据 `output/validation/review-20261102/313-frontend-b8c637e8649e4bf086f97344c2dfc350/node-tests.txt`。`npm run typecheck`、契约和三语生成 `--check` 通过。
- m14 Python、offscreen、独立 fresh PostgreSQL 和项目科学解释器，pytest `-q -o addopts= --basetemp=<owned scratch>`：存储并发、工作区、审查回归、保留清理、认证、准入、配置、远程协议、科学集成、真实M01、桌面与更新应用、声学及M06合并复验200项通过。证据 `output/validation/review-20261102/313-recheck-0fc1dc194d214d7daeb236dfed92973d/pytest.txt`。
- 同轮新增监管集成首次失败：Windows虚拟环境Python启动器增加一层进程，测试只查询直接子进程，未找到worker，也未发送终止信号。调整为仅在本次API进程后代中识别实际worker叶进程，保持重启及后续任务断言。定向复验 `desktop/tests/test_desktop_review_20261102.py::test_local_api_restarts_its_crashed_worker_and_runs_next_job`：1 passed，5.94秒，实际终止自有worker、宿主拉起新PID、后续任务成功且API健康。证据 `output/validation/review-20261102/313-supervisor-b8460dc73c8a44b0b51b7d23b8aa4a04/pytest.txt`。

本轮201个不同Python用例得到通过证据，初轮失败与修正如上，未将初轮写为全通过。所有新增检查均后台运行，没有打开应用窗口或抢占焦点。当前已测范围未发现新增回归；这不能替代最终3.1.3冻结EXE/安装升级/Web成品验收，也不能证明所有输入均无缺陷。Mac原生运行、生产容量和既有待验证科学项仍按前述边界保留。上线前需对最终冻结成品再次验收并取得该版本文案确认。
