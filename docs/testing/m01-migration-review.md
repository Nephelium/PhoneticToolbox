# M01-F：005 持久批次表的具体审阅

2026-09-10。**状态：applied，2026-09-10两库已按原SQL应用。** 井井对本份具体审阅及限定合成验证/清理回复“好，继续”，该授权已核实并执行；不是复用003/004授权。证据见[F2报告](m01-persistent-report.md)及`output/validation/m01/persistent-7d7a6a1a5818455da1de88cbadcfb33b/report.json`。后续不得重复DDL。下文保留实际应用时审阅的精确范围。执行准备证据见 [M01-F1报告](m01-execution-preparation-report.md)。

## 拟操作对象

| 对象 | 固定目标 | 变更 |
| --- | --- | --- |
| PostgreSQL | `127.0.0.1` 的 `ptb_p05_test_20260909`；PGDATA=`output/validation/p05/postgres-data` | `ptb_jobs` 增加3张批次表和3个索引；插入1条新版本记录 |
| 桌面SQLite | `output/validation/p06/local-state.sqlite3`，必须已经存在 | 增加3张批次表、3个批次索引及旧jobs的1个复合唯一索引；插入1条新版本记录 |
| 配额/文件目录 | 复用已有 `output/validation/p07/storage` 标记根 | 建表过程只核验目录实例身份及锁，不增加/删除用户文件 |

三个新表为 `acoustic_batch_version`、`acoustic_batches`、`acoustic_batch_items`。批次保存有序输入、参数快照、幂等键、取消标记和子任务引用；子任务继续使用P06状态/租约/代际校验，不另造状态机。每批1–1000项，未创建子任务的预留项也应在F2接入时原子计入既有1000任务上限。

PG外键要求批次、项目、子任务和音频资源属于相同owner/project；SQLite固定local身份和本地项目。关联TextGrid/唇形的hash保存在输入快照内，资源引用/到期检查仍须F2事务接入；新表存在本身不能证明隔离和恢复已经通过。

## 精确SQL与保护

- [PostgreSQL SQL](../../backend/migrations/005_acoustic_batches.sql)，归一化UTF-8文本SHA-256：`3d1006b267d3db69d6e0b9bc3419dd7062e261aba24f21712c5bf7ec169dedf6`。
- [SQLite SQL](../../backend/migrations/005_acoustic_batches_sqlite.sql)，归一化UTF-8文本SHA-256：`d66c08a7885f78676651afe1f08c0c9e459aafae901fc4bfe5fde8d8e6c47847`。
- [执行工具](../../scripts/m01_database.py) 默认不做任何初始化；`show`只读SQL，不打开数据库。`apply`同时要求具体授权标记和上述hash，SQL变更后必须重新审阅。
- PG只接受固定名称的loopback数据库和已有P07存储实例；验证001/002/004版本，锁定相关表，在事务内比较13张旧表的逐行摘要。SQLite以`mode=rw`打开固定路径，拒绝链接/reparse、缺失库、错误版本、重复005对象，比较3张旧表并检查所有外键。
- 所有DDL在单库事务内完成；异常回滚。PG锁等待5秒/单语句30秒，SQLite写锁等待5秒。两个库之间没有分布式事务：一库成功一库失败必须分别报告，不能自动删除成功库的新表来回退。
- 不改旧行、旧表字段、账号额度、P07 ZIP上限或配置文件；不删除旧历史或文件。不执行旧库整库备份，遵循井井不要额外备份的要求。

只读审阅命令（已执行）：

```powershell
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/m01_database.py show --kind postgres
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/m01_database.py show --kind sqlite
```

已获具体授权并执行的入口（以下为历史应用命令，不应重复执行）：

```powershell
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/m01_database.py apply --kind sqlite --approved-m01-batch-schema --reviewed-sha256 d66c08a7885f78676651afe1f08c0c9e459aafae901fc4bfe5fde8d8e6c47847
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/m01_database.py apply --kind postgres --approved-m01-batch-schema --reviewed-sha256 3d1006b267d3db69d6e0b9bc3419dd7062e261aba24f21712c5bf7ec169dedf6
```

PG入口另从标准输入接收既有私有连接配置，禁止把DSN放命令行/日志。受控启动流程复用P07已有原则：只在固定PGDATA没有运行实例时，用项目内既有PostgreSQL临时绑定loopback端口；记录自己启动的实例，只对该PGDATA与实例身份匹配的进程正常停止。不能直接执行整套 `run_p07_validation.py`，其旧测试动作不是本次建表的一部分。当前F1没有启动该集群，也没有执行上述apply命令。

## 已授权的联合验证范围

1. 先应用两份005，核实旧行摘要完全保留、新表外键和版本正确。
2. F2继续实现提交/认领/取消/租约与原子发布；使用本轮自造的合成音频和TextGrid，验证17项批次、第二项失败、取消、进程退出、输入删除/到期、旧代结果拒绝及重启恢复。
3. 新测试文件只在 `output/validation/m01/` 的新UUID子目录，以及已有P07标记根中新登记的M01测试资源内生成/清理。测试记录只操作本次随机owner/project/batch/job ID并保留此前记录；不依据泛化路径或进程名删除/停止对象。清理失败保留证据，不扩大范围。
4. F1的WAV/XLSX/SQLite样例是合成导出文件，不是任务schema迁移；随后F2已按本次具体授权执行005，并验证PG并发、持久恢复和UI保存，结果以[F2报告](m01-persistent-report.md)为准。
