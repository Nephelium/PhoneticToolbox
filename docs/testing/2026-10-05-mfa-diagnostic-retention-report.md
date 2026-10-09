# MFA 诊断临时目录保留与清理

日期：2026-10-05。范围：Windows 源码、明确登记的当前本机实例 MFA attempt 临时目录。井井已授权应用临时文件的定期清理。本轮沿用现有本机保留服务，未修改数据库 schema、模型登记、手册作者章节、更新器或构建脚本。

## 实现与策略

- `m11_executor` 为每次任务生成 `registry_root()/attempts/<job UUID>-<generation>-<random UUID hex>`。新目录创建后、复制输入和模型前，在资产锁与 SQLite transaction 中核查当前任务的租约、worker/generation fencing、MFA 操作及 desktop-local 路由，再登记归属。
- 登记写入本机实例 `.ptb-retention.json` 的 `mfa_attempts`，并在目录内创建 `.ptb-mfa-attempt.json`。两份记录共同绑定实例 ID、任务 UUID、generation、attempt 名和目录设备/文件 ID。清理从当前配置的固定 `registry_root()/attempts` 重建路径，不采用登记 JSON 提供的任意删除路径。
- 只有执行器心跳线程结束，并且未调用原生 runtime，或已收到 owned 原生进程组 `group_cleaned=True` 证据，才登记 `completed_at`。runtime 已调用但清退证据不完整时保持未完成。完成登记异常同样保持保护，避免把已发布的对齐结果反标成失败。
- 临时目录保留固定 7 天，跟随现有 `enabled` 开关。期限取 `max(first_seen, created_at, completed_at)`，保留从首次完成登记开始的完整宽限，不采用目录或文件 mtime。重复完成登记不会重置日期。
- 未完成 attempt、任务缺失、非终态任务，以及同一任务任何当前 queued/running/cancel_requested generation 均受保护。历史未登记目录不扫描、不补猜归属，mtime 再旧也不删除。
- 清理只遍历当前实例的登记项。注册模型、词典、`kernel-cache` 和用户原件均位于该删除范围之外。登记身份不符、注册根变化、目录被替换、任意祖先或树中符号链接/Windows junction/reparse point、文件 hardlink 或特殊文件均拒绝。
- 删除前检查完整 tree，最多 200000 个成员；删除时再次核查路径及文件 link count。归属 marker 最后删除。若最终根目录删除失败，且根身份仍匹配，则恢复原 marker，保留登记以便下次重试。前缀删除、访问拒绝等失败不会标成完全成功。
- 清理仍由既有 worker 保留线程执行，常规最多每日一次；显式清理可立即重试。API 继续使用既有本机 token 和 Origin 鉴权，无新路径选择参数。

## 记录与返回值

`.ptb-retention.json` schema 保持 `ptb-retention/1`。新增 `mfa_attempts` 是 attempt 名到记录的映射，包含 `registry`、`job_id`、`generation`、`directory_id`、`first_seen`、`created_at`、`completed_at`、`size_bytes`、`removed`，清理后记录 `removed_at`，失败记录 `last_failure`。字段和 timestamp 在读取时校验。

目录内 marker schema 为 `ptb-mfa-attempt/1`，包含 `instance_id`、`job_id`、`generation`、`attempt`、`directory_id`。marker 需为单链接普通小文件，读取内容必须逐字段与登记一致。

既有状态接口增加 `diagnostic_days`、`diagnostic_count`、`diagnostic_bytes`，count/bytes 表示登记中尚未清理的数量与最后一次已知大小。尚未完成的目录大小未扫描，数值不等于实时磁盘总量。cleanup 增加 `diagnostic_failed_count`、`diagnostic_protected_count`，其 count/bytes 表示本次实际清理数量/释放字节。总 `count`、`bytes`、`failed_count`、`complete` 包含 MFA 诊断清理结果；部分删除按可验证的剩余大小计算已释放字节。

## 实际验证

使用 `.venv/v3-dev/Scripts/python.exe`，加入 `backend/src` 与 `packages/phonetic_core/src` 搜索路径后运行：

```python
pytest.main([
    'backend/tests/test_mfa_retention.py',
    'backend/tests/test_local_retention.py',
    'backend/tests/test_m11.py',
    'backend/tests/test_m11_web.py',
    'backend/tests/test_m11_r2_probe.py',
    'backend/tests/test_m11_bundled_registry.py',
    '-q', '-rs', '-o', 'addopts=',
])
```

结果：**73 passed，1 skipped**，8.73 秒。跳过项为原 M11 web 测试要求显式启用的新建合成 PostgreSQL 集群，未创建或访问该集群。两个 warning 为现有 Starlette/httpx 与 anyio 弃用提示。

新增 19 项测试使用临时目录和副本 SQLite，执行实际登记、持久状态读取、文件删除和重试，覆盖：

- 完成后的精确 7 天边界、创建时间很旧仍有完整完成宽限、重复完成不重置日期。
- 历史未登记目录及其旧 mtime、另一实例同一注册根、三种当前活动状态跨 generation、未完成终态保护、关闭/开启保留开关。
- 实际 Windows junction 目标不枚举/不删，移除 link 后成功重试；实际 hardlink 阻止任何前缀删除；用户原件、模型、词典与 kernel-cache 哨兵保持。
- 目录被替换且复制 marker、marker 实例篡改、注册根变化、越界路径及非 MFA 操作拒绝。
- 中途文件删除失败保留 marker 与登记，最终根 rmdir 失败恢复完全相同 marker，下一次正确重试，失败汇总保持 `complete=False`。
- 执行器失败收尾与真实登记/SQLite 链路；原生进程组清退证据为 true/false/缺失三种情形，只有 true 进入完成状态。

## 边界

本轮未重新运行真实 MFA 数 GB 语料、物理录音、PostgreSQL 或冻结成品中的定期清理。执行器测试中的 runtime 是诊断收尾适配器，不能记为真实 MFA 对齐验收。已登记但无法证明完成的 attempt 与未登记历史 attempt 会继续保留，保守保护是明确策略。主线最终包需重新构建，并核查新模块和登记接线进入冻结产物。本轮无公开发布、push、全局安装或用户既有文件清理。

改动文件：`backend/src/ptb_worker/mfa_retention.py`、`local_retention.py`、`m11_executor.py`、`backend/tests/test_mfa_retention.py` 与本报告。
