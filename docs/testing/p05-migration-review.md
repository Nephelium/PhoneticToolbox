# P05 实际数据库操作审阅

状态：authorized / executed。2026-09-09，井井对本方案回复“允许”；已在下述专属空库执行审阅的 SQL 并通过真实 PostgreSQL 验证。此授权不覆盖其他数据库、后续 schema 或部署。

## 已授权并执行的具体目标

仅为 P05 新建隔离 PostgreSQL 测试实例，专属数据库 `ptb_p05_test_20260909`，数据位于本工作区忽略目录 `output/validation/p05/postgres-data`。实测监听 `127.0.0.1:10255`，使用 SCRAM-SHA-256，端口在启动前检查可用；未复用相邻 v2、旧网站或其他项目数据库。三个随机账号及其数据仅供本轮验证。凭据保存在当前 Windows 用户专用 ACL 的忽略目录 postgres-private，通过子进程私有标准输入传入工具，不进入命令行和报告。

本机及现有 NInfer WSL 未找到可用的 PostgreSQL，Docker 引擎未运行。已通过 PostgreSQL 官网指向的 EDB 下载页获取 Windows 17.11-3 二进制压缩包，运行 postgres --version 返回 17.11。运行时位于 .venv/postgresql-17.11-3，仅提取 bin/lib/share/doc 和根许可。未运行安装器或注册 Windows 服务。来源、本地计算的哈希及许可证据见 [运行时清单](../../third_party/p05-postgres-runtime.json)。下载页未提供独立 SHA-256，postgres.exe 未签名，不将本地哈希误称为厂商认证。

## 已执行的数据库变更

[001_accounts.sql](../../backend/migrations/001_accounts.sql) 在单个事务中创建专属 `ptb_accounts` schema 与五张表：版本、账号、会话、项目、登录计数。包含 UUID 主键、登录名唯一性、owner 外键、会话截止检查、项目 owner 索引和双字段唯一性。不包含 DROP 或旧数据导入，不自动执行清理。

```powershell
# 只读查看，现已可执行。
& '.venv/v3-dev/Scripts/python.exe' scripts/p05_database.py show
# 获授权且专属空库已建立后才运行；DSN 在隐藏提示中输入，不放参数或日志。
& '.venv/v3-dev/Scripts/python.exe' scripts/p05_database.py apply --approved-empty-test-database
& '.venv/v3-dev/Scripts/python.exe' scripts/verify_p05_postgres.py --approved-test-data
```

迁移工具会拒绝名称不符合 `ptb_p05_test_*` 或已有业务表的数据库。完整 SQL 必须先审阅。脚本不创建数据库本身，不自动运行 migrations，也不更改 `.env`。后续账号创建通过受控命令与隐藏密码输入，不提供默认管理员密码。

## 已补齐的验证

真实 PostgreSQL 下，两账号项目隔离、不同 adapter/app 重建后的会话与项目恢复、退出撤销跨连接生效、事务回滚，以及 12 个并发连接争用同账号登录预算时恰好 10 个通过。另以真实 HTTP 验证 API 与数据库进程停止再启动后会话/中文项目恢复、退出后旧 Cookie 失效。重复执行迁移会拒绝非空库。测试生成随机凭据，不打印、不写提交文件，测试数据留在专属测试库供复核；所属 API 和 PG 进程已停止。完整命令和首轮 Windows 输出句柄问题见 [验收报告](p05-accounts-report.md)。

此时仍不等于 P06/P07 已验收：任务/日志/下载/取消/配额路径要在对应接口实现后做跨用户验证，不能以当前缺少接口就宣称通过。
