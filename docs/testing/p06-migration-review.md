# P06 任务表操作审阅

2026-09-09，状态 executed / verified（限定已审阅测试范围）。19:31 井井在具体审阅后回复“继续，刚刚不小心闪退了”，原任务随后执行并验证。本轮核对原始会话、已有表与测试证据，再次复验通过；下文保留审阅时的目标及命令，不构成对其他数据库的授权。

## 已审阅并执行的唯一范围

1. 启动并复用 P05 已验证、仅本机监听的测试 PostgreSQL，数据库仍为 ptb_p05_test_20260909，PGDATA 仍为 output/validation/p05/postgres-data。在其中新增独立 ptb_jobs schema：jobs、events、schema_version 三张表及索引。P05 users/sessions/projects 原表不修改、不迁移旧数据。
2. 在 v3 忽略目录新建 output/validation/p06/local-state.sqlite3，建立等价本机任务表，验证桌面同一 API。目标文件已存在时拒绝覆盖。
3. 创建随机测试账号、项目及有限任务元数据，执行并发认领、取消、杀死自有 worker、超时、重试与重启验证。只管理此次创建的进程；测试结束停止服务，保留测试记录。数据库自身正常事务日志管理属于此次测试，不手工删除用户文件或测试库。

已准备完整 SQL：[PostgreSQL](../../backend/migrations/002_jobs.sql)、[SQLite](../../backend/migrations/002_jobs_sqlite.sql)。PG 依赖 P05 schema version 1；工具拒绝其他库或已存在的 ptb_jobs；本机工具使用独占文件创建。没有 DROP、ALTER、旧数据导入或自动启动迁移。出错保留证据，回退方式是停止新任务服务，继续使用 P05；不自动回滚删除新表。

## 首次执行入口（已完成，勿重复建表）

使用现有 .venv/v3-dev/Scripts/python.exe；不增加全局依赖，不修改 .env、系统服务或 v2。

```powershell
python scripts/p06_database.py show --kind postgres
python scripts/p06_database.py show --kind sqlite
# 仅在此方案获授权后：PG 私密配置由父进程通过 stdin 提供，不写命令参数。
python scripts/p06_database.py apply --kind postgres --approved-p06-test-schema
python scripts/p06_database.py apply --kind sqlite --approved-p06-test-schema
python scripts/verify_p06_jobs.py --approved-test-data
```

最后一个工具分别接收 PostgreSQL/SQLite 私密配置，验证同一状态机的实际行为；不使用测试替身替代数据库证据。已有纯政策、进程和接口边界测试不能证明真实 SQL 并发通过。此门已通过；结果见 [P06 报告](p06-jobs-report.md)。恢复复验使用 `python scripts/run_p06_validation.py --approved-p06-test-data`，不带 --apply-reviewed-schema；再执行 `python scripts/verify_p06_local_service.py --approved-test-data`。

不含 P07 文件上传、配额/删除/下载表，不含生产、公开部署、正式账号、邮件、原生设备或语音算法迁移。
