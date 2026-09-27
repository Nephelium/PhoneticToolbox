# P07-POLICY / 006 增量迁移审阅

## 2026-09-27 具体目标与最终授权门

**代码公共接线、契约生成和独立合成验证已完成，见[本轮报告](p07-policy-closeout-report.md)。现存目标库未迁移，未启动、停写或恢复其服务。** 以下为可审阅执行方案，覆盖本文件较早的“等待 M08 接线”和最大编号描述。

| 项目 | 本次唯一候选目标/边界 |
| --- | --- |
| 实例/数据库 | Windows 本机回环专用测试集群，`ptb_p05_test_20260909`；不含云服务器/生产库 |
| PGDATA | `D:/PhoneticToolbox/PhoneticToolbox_v3/output/validation/p05/postgres-data` |
| 存储根 | `D:/PhoneticToolbox/PhoneticToolbox_v3/output/validation/p07/storage` |
| 存储实例标记 | `efe95c3e-708c-4c00-b9f3-4ec19f7d49b1`；在线 state 必须完全相同 |
| 软件/政策 | 当前源码 backend/core 3.0.0a1、API 1.1.0，目标 policy 1 → 2；磁盘 PG_VERSION=17 |
| 当前在线状态 | 本轮检查无 postmaster.pid，没有启动现存集群；不能据此断言当前 SQL schema 版本已实查。上轮 M08 只读 gate 为 policy 1，本轮在线确认仍待执行 |
| 在线前置版本 | accounts.schema_version=1、jobs.schema_version=1、storage.state.version=1、job_files_version=1、acoustic_batch_version=1，policy_version 缺失/1；这些是独立版本表，不是一个全局整数 005 |
| 006 文件 hash | `bf5eb0d18eb3d242db98f026c9a32c55a90bf57b21589efe36365a974866f247`，本轮 SQL 未改 |
| 后续编号 | 当前已有 B 的候选 007_remote.sql，未执行、未纳入本申请；其 hash 可能随 B 开发变化，实施前再次枚举 |

### 停写、预检、执行和恢复

1. 确认仅授权上述库/根及 006。冻结新上传/chunk/finalize、任务提交/重试/保存复制/结果管理、worker claim/发布、定时清理/恢复和手工脚本写入。维护期间服务写者全部退出，下载可短暂停止；不能误停其他目录/账号/端口实例。没有已知持有者时停止执行并核对，不按进程名批量停止。
2. 首选排空 running/cancel_requested 后窗口。queued 与未完成上传保留。runner 遇到 running/cancel_requested 会拒绝，不能自动取消、清零 reserved 或强行恢复。若需保留仍运行的 worker 跨窗口，另审具体进程与 lease，而不复用本 runner 的排空假设。若目标已经运行，runner 拒绝启动/接管，需要井井明确指定该实例的停写和进程权限。
3. 由已有私密配置在内存提供 DSN，禁止输出配置。runner 使用现有 `owned_postgres`：核对固定 PGDATA/项目内运行时，选择随机回环端口，记录启动身份，只正常停止自身新启动实例。先持存储 OS/advisory lock，要求目标无其他 DB 会话，再开启 READ ONLY 预检。schema/instance、冻结、账本、资产尺寸/未知文件、used/reserved、旧 manifest/snapshot 指纹须通过。未知文件或 fsync 未结算等差异要求人工核对，不在预检中自动恢复或删除。
4. 预检通过且仍符合本授权后，在持锁连接只执行 hash 对应 006；SQL 内再次有界锁定并核对账本。SQL 异常显式 ROLLBACK，不修改约束重试。实际迁移不执行 001–005，也不执行 007。
5. 提交后在 READ ONLY 事务验证 policy=2、quota=1,000,000,000、存量 assets.policy_version=1，原资产字段/used/reserved/任务 snapshot/manifest 指纹一致。任何失败保持关写，先确认提交状态，再按下方回滚边界处理。runner 不自动恢复 API、worker、清理器或系统服务。
6. 独立验证获准的合成账号/目录，记录 1 GB、259,200 秒、旧资产原期限、超额读删/禁增、已写完在途结算、到期及迟到 worker。新科学结果与 M08 复制分别验。之后才按明确恢复授权启动当前版本 API/worker/清理器，重新验证 capability；旧写者不得混跑。M04/M08/M14 联验入口和覆盖缺口见[交接文档](2026-09-27-prerequisites-handoff.md)。

默认只读展示，已实际执行：

```powershell
& '.venv/v3-dev/Scripts/python.exe' scripts/p07_policy_database.py show
```

以下命令**尚未执行**，仅在井井对本具体目标、窗口和恢复边界明确同意后使用。标记本身不能替代会话授权：

```powershell
$env:PYTHONPATH='D:/PhoneticToolbox/PhoneticToolbox_v3/backend/src;D:/PhoneticToolbox/PhoneticToolbox_v3/packages/phonetic_core/src'
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/p07_policy_database.py apply --approved-dedicated-p07-006 --reviewed-sha256 bf5eb0d18eb3d242db98f026c9a32c55a90bf57b21589efe36365a974866f247
```

该工具的 apply_checked 已在本轮独立随机库测试：错误目标/hash 拒绝，006 通过且原元数据指纹不变。未借此迁移本候选目标。执行记录新建于 `output/validation/p07-closeout-20260927/migration-<UUID>/`，只记录脱敏报告，不打印 DSN/密码。

### 现有恢复手段及备份缺口

- 可用：006 单事务在 COMMIT 前完整 ROLLBACK，已有 PGDATA/WAL 的 PostgreSQL 常规崩溃恢复，以及提交后保持关写、保留新版兼容读取器并另审向前修复。独立合成测试已经验证账本失败时整事务回滚。
- 未证实：本轮未在 p05/p07 证据目录发现 dump/bak/命名 backup，pg_wal/archive_status 为空，未发现显式 archive_mode/command 设置。**没有经过还原验证的目标库备份或 PITR 能力证据**。当前 PGDATA 约 71.9 MB，仅为当前数据目录，不是备份。元数据 SHA 指纹不能还原内容。
- 本轮未创建额外整库备份，也不以数据库迁移授权兼作备份授权。若井井要求补恢复点，单独申请：仅上述专用集群在正常停机后的 PGDATA 与对应存储根、一个项目内忽略目录、预估空间为实际目录总字节加 20%（当前至少约 86.3 MB，执行前复测），保留至迁移验收后 24 小时。创建、恢复验证和期满删除都需明确范围授权，不备份 V2/整个工程或云端库。默认本方案不执行该候选备份。

本轮最终授权依据是[根 AGENTS.md](../../AGENTS.md)的数据库实际迁移、服务/进程所有权边界及井井本次明确要求。需要确认的动作是：启动上述当前停止的专用集群完成在线预检，满足前置条件后执行 **006**，独立验证，再启动本任务拥有的 API/worker 做合成网页联验并在结束后正常停止。任何身份/版本/恢复缺口需要改变本方案时重新审阅。云库、007、生产部署及长期服务恢复不在本申请中。

---

以下保留 2026-09-26 的原兼容/SQL设计说明，状态与编号以上方最新记录为准。

2026-09-26。**候选方案待井井确认。现存/服务数据库未执行迁移，新政策尚未在运行库生效。** 本轮只在随机新建、本轮独占的 PostgreSQL 合成测试集群执行 001/002/003/004/006，验证 SQL 与兼容行为。该测试不构成任何现存库的 apply 授权。

入口：[实施及共享文件清单](../plans/2026-09-26-p07-policy.md)、[测试报告](p07-policy-report.md)、[006 SQL](../../backend/migrations/006_storage_policy.sql)、[只读预检](../../scripts/p07_policy_preflight.py)。003 原文件保持不变。当前最大既有编号为 005，006 为本轮候选，实际迁移前须再次检查并行任务是否新增编号。

## 需要确认的存量兼容规则

| 对象 | 候选行为 |
| --- | --- |
| 已完成上传、旧科学结果、旧 ZIP | 原 expires_at 不变，保持原有到期访问和删除机制，不因降额提前删除 |
| 旧 manifest、任务 snapshot | 不改 JSON、不重新生成历史结果。缺政策字段的 AcousticFileManifest 按版本 1 读取，其他历史 manifest 保持绝对 expires_at 读取语义 |
| 账号额度 | 将 quota_bytes 改为十进制 1,000,000,000，保留所有 used/reserved 原值。超额允许存在，available=0，下载/删除仍可用 |
| 旧在途上传/任务 | 不在迁移 SQL 内取消、清空预留或删除文件。到期/租约/取消规则继续适用。超额时禁止物理追加，即便之前有预留；恢复可将已写字节从 reserved 转入 used，合计不增长 |
| 已全部写完的在途上传 | 可最终确认，释放未用预留，从本次确认起 259,200 秒，标政策 2；下载旧 ready 资产不触发该过程 |
| 已全部写完的在途结果 | 输入及租约仍有效时可成功提交，按新发布期限；迁移前输出的版本字段更新待共享 files.py 串行接线完成 |
| 新上传/新结果 | 按政策 2：三天上限，派生/复制/ZIP/切分取输入更早截止，科学重算从完成起计时 |

已超额且无法继续的任务会在尝试写入时收到 quota_exceeded，随后沿既有任务失败清理路径处理其暂存输出。实际迁移前需要明确停写/恢复顺序及是否先让在途任务完成。SQL 本身不代用户选择或删除任何数据。

## 006 的数据库变化

1. 在事务内取得既有存储 advisory lock（577707）及三张存储表的排他锁，设置有界锁等待和 SQL 时限。必须停止全部 API 写入、worker、清理器之后才可执行；不能仅依赖迁移进程的锁隔离旧二进制。
2. 在锁内再次校验 state 版本、旧 quota 值、账本与资产汇总一致性；不一致则整个事务失败。
3. state 增加 policy_version 和 policy_switched_at，写入切换时间。assets 增加 policy_version：存量填 1，后续默认 2，原期限、尺寸、预留、hash 均不变。
4. quota_accounts 移除旧 `quota_bytes=5000000000` 及总量不可超额 CHECK，改额度默认值/现有值和固定 CHECK 为 1 GB。非负 CHECK 保留。
5. 新增原子增长 trigger：新账号不能超额，已有超额行允许合计保持/减少，拒绝超额增长。既有行锁序列化并发变更。应用层进一步禁止超额账号继续物理追加；数据库允许原预留转实际，以支持崩溃恢复。
6. 保留历史 assets.expected_bytes 的 5 GB CHECK 供存量回读/结算，新 INSERT trigger 限制版本 2 和新 expected_bytes≤1 GB。没有为过量旧数据添加清理 SQL。

SQL 常量是这一不可变迁移的政策快照。应用运行时新常量只定义在 storage_policy.py。将来换政策应新增下一号迁移，不回写本次已应用的 SQL。

## 预检和实际执行门（本轮未对现存库执行）

只读工具没有 apply 分支，也不调用 Storage.recover/cleanup。以标准输入接收 `{"dsn": ...}`，不把连接串写入命令行、日志或报告：

```powershell
# 由获准的私密配置传递程序把一行 JSON 送入 stdin，不在命令行粘贴凭据。
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/p07_policy_preflight.py
```

工具的事务显式 READ ONLY，输出聚合账号数、超额数、占用/预留、在途资产/任务、删除失败/到期数、账本差异及约束定义，不输出账号/文件名。`ready_for_review=true` 只表明基础检查可进入审阅，不是授权或生产准备完毕。

实际 apply 前还须人工/运行手册完成：

- 确认目标实例、库、存储根和 instance_id 的对应关系，当前迁移编号及 SQL hash；不将测试 DSN 沿用为目标库。
- 确认上述存量方案和停写窗口。完成 M08/P07 共享文件整合、生成契约、新旧客户端兼容及 Linux PostgreSQL/磁盘锁验证。
- 记录旧 state/quota/资产期限/任务 manifest 摘要与 counts，确认无未知文件、账本差异及未处理恢复异常。不得为了通过预检直接清零预留或删除文件。
- 停止所有写者，确认没有旧 worker 能发布结果；在同一数据库会话执行已审核 SQL，出错显式 ROLLBACK，禁止自动改约束重试。
- 用合成账号验证新额度/期限、旧账号读删、输入期限、配额竞争和清理。逐项回读存量 expires_at、manifest、used/reserved 未变后再开放新写入。

## 回滚边界

- **事务提交前：** 任意错误 ROLLBACK，恢复原 schema/值。本轮已用账本不一致验证事务全回滚，没有 policy_version 残留。
- **提交后尚未开放写入：** 优先保持新版兼容读取器并关闭新写入，调查后向前修复。若确需恢复 5 GB，须另审增量补偿迁移：确认所有账号 used+reserved≤5 GB，再调整默认值/约束，保留原 expires_at 与所有数据。不能简单重复 003。
- **已有政策 2 新资源后：** 回退额度不等于延长新资源到期时间。不得把三天改回七天来模仿旧环境，不能删除新资源或恢复旧快照覆盖新写入。
- **应用回滚：** 旧模型可能拒绝新增 policy_version 字段，旧 worker 还可能写七天期限。保留读兼容补丁、关写并核对任务后才能回退二进制；禁止新旧写者混跑。
- 本次不提供自动 down/apply，不删除 schema/数据/目录；任何补偿 DDL 和生产部署需要独立确认。SQL 提案和独立集群通过不能代替实际目标库验证。
