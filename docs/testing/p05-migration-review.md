# P05 实际数据库操作审阅

状态：prepared / 未执行。账号与项目代码、SQL 和显式 PG 验证脚本已准备；本文件不是执行授权。

## 请求确认的具体目标

建议仅为 P05 新建一个隔离 PostgreSQL 测试实例，专属数据库命名 `ptb_p05_test_20260909`，数据放在本工作区忽略目录 `output/validation/p05/postgres-data`。服务只监听本机回环地址，测试端口在启动时检查占用；不复用相邻 v2、旧网站或其他项目数据库。创建测试账号及其数据仅供本轮验证。

本机发现 Docker CLI，但引擎未运行；PATH 中未发现 postgres/initdb/psql。因此实例供应方式须在获授权后落实：优先检查是否已有可用的本机/WSL PostgreSQL；若没有，则在 v3 隔离目录安装官方 PostgreSQL 运行时。启动 Docker Desktop 或修改全局配置不在此方案中，不自动执行。下载版本、来源、校验和与包体许可在实际选择时登记。

## 将执行的数据库变更

[001_accounts.sql](../../backend/migrations/001_accounts.sql) 在单个事务中创建专属 `ptb_accounts` schema 与五张表：版本、账号、会话、项目、登录计数。包含 UUID 主键、登录名唯一性、owner 外键、会话截止检查、项目 owner 索引和双字段唯一性。不包含 DROP 或旧数据导入，不自动执行清理。

```powershell
# 只读查看，现已可执行。
& '.venv/v3-dev/Scripts/python.exe' scripts/p05_database.py show
# 获授权且专属空库已建立后才运行；DSN 在隐藏提示中输入，不放参数或日志。
& '.venv/v3-dev/Scripts/python.exe' scripts/p05_database.py apply --approved-empty-test-database
& '.venv/v3-dev/Scripts/python.exe' scripts/verify_p05_postgres.py --approved-test-data
```

迁移工具会拒绝名称不符合 `ptb_p05_test_*` 或已有业务表的数据库。完整 SQL 必须先审阅。脚本不创建数据库本身，不自动运行 migrations，也不更改 `.env`。后续账号创建通过受控命令与隐藏密码输入，不提供默认管理员密码。

## 将补齐的验证

真实 PostgreSQL 下，两账号项目隔离、不同 adapter/app 重建后的会话与项目恢复、退出撤销跨连接生效、事务回滚，以及 12 个并发连接争用同账号登录预算时恰好 10 个通过。测试生成随机凭据，不打印、不写提交文件，测试数据留在专属测试库供复核。

此时仍不等于 P06/P07 已验收：任务/日志/下载/取消/配额路径要在对应接口实现后做跨用户验证，不能以当前缺少接口就宣称通过。
