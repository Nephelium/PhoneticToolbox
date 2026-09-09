# P07 存储表与合成文件清理审阅

2026-09-09，prepared / 未执行。P07 第一批代码、完整 SQL、验证脚本与无数据库的边界测试已准备。本文是具体操作范围，不把 P07 实施授权自动扩大为实际建表/文件删除授权。

## 请求确认的范围

1. 复用已验证且已停止的本机 PostgreSQL，数据库仍为 ptb_p05_test_20260909，PGDATA 仍为 output/validation/p05/postgres-data；只在运行验证时启动，只监听 127.0.0.1，结束停止。
2. 执行 [003_storage.sql](../../backend/migrations/003_storage.sql)，新增独立 ptb_storage schema 的 state、quota_accounts、assets 三张表及索引。P05/P06 表结构不修改、旧数据不迁移，没有 DROP/ALTER。
3. 新建 **output/validation/p07/storage**，放置根目录标记与锁文件；创建少量随机测试账号和项目，写入确定性的测试字节。单次测试数据约数 MB，无个人语料。5 GB 边界通过预留与独立计量测试验证，不为测试复制 5 GB 音频。
4. **允许程序删除这个专属目录中由本轮测试生成的随机 UUID.bin 文件**，用于测试主动删除、到期删除、删除失败后的重试、异常恢复，以及后续 P07 ZIP/生成联合验证。允许对这些测试行调整截止时间和注入故障；只处理程序已登记的测试资源，不递归删除目录、不扫描清理其他 output 内容，不碰 v2、电脑原文件、旧研究数据或测试根之外的路径。
5. 已审阅测试范围内允许反复修复和复验；控制本轮创建的 API/清理/worker/独立浏览器进程。没有全局安装、系统服务、.env/凭据改动、公开部署、推送或对外发送。

## 操作入口

```powershell
# 只展示 SQL，可在授权前使用。
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/p07_database.py show

# 仅在上述范围获确认后首次执行：
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/run_p07_validation.py --approved-p07-schema-and-test-files --apply-reviewed-schema

# 后续复验不得重复建表：
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/run_p07_validation.py --approved-p07-schema-and-test-files
```

私密配置从既有私有文件由父进程经 stdin 传送，不放命令行/URL/报告。初始化同时检查固定库名、P05/P06 前置版本、schema 不存在和测试目录不存在；根标记与数据库 instance_id 必须一致。

## 失败与恢复

初始化任一步失败时保留证据，不自动删除文件或回滚已有 schema。回退到 P06 的方式是不配置 storage_root 并停止新服务；不做 DROP。删除失败的文件仍计量占用，未知文件/大小异常冻结新增写入并留下待核对状态。服务恢复先核对再开放下载；正常运行与进程中断验证不冒充真实断电或全部跨平台持久化。

当前脚本覆盖初批单文件场景；完整 Q01–Q20、生成文件的 P06 fencing/manifest、ZIP 与输入到期联合验收仍按 [P07 计划](../plans/2026-09-09-p07-storage.md)继续。没有实际执行报告前保持 in_progress。
