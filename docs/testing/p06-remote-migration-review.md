# P06-REMOTE 007 迁移提案与公共串行清单

2026-09-27。**review only，未操作现存或服务数据库。** 井井确认 A 尚未释放公共文件。本轮没有修改 store/files/storage/main/job_models/resource_profiles 或公共契约生成器。

`backend/migrations/007_remote.sql` 新增六张表和一个当前 attempt 唯一索引，不更改旧 jobs 行、旧期限或 P07 配额。nodes 只存凭据 hash；jobs 扩展保存准入预算；attempts 保存代数/位置/运行时/摘要/失败与幂等回执；uploads/chunks/reads 保存受控传输进展。输出实际字节及额度继续归 P07，不存入这些正式表。

独立测试对该 SQL 仅移除 ptb_jobs namespace 后，在每测试新建 SQLite 中实际执行；没有据此声称 PostgreSQL DDL/并发已通过。006 政策迁移、目标实例/库/schema、备份与回滚窗口、迁移执行身份须在后续具体授权中逐项列出。SQL 不使用 IF NOT EXISTS 掩盖不一致，不在服务启动时迁移。编号待串行整合时复核。

公共文件释放后的顺序：

1. 完成 A 的 FileConfig、独立科学结果期限、policy_version 与 M08 保存副本接线，冻结 P07 的最终 policy。旧结果保持读取规则。
2. 实现 RemoteFiles 的正式 PostgreSQL/P07 bridge，统一 Storage → scheduling 锁顺序。validate_inputs 每次校验账号 active、owner/project、输入当前状态/hash/期限及新政策写入门。reserve/append 使用既有原子额度和 fsync 恢复；seal 核 hash；publish 在传入事务完成资产/额度/manifest，禁止另开连接。
3. 旧 JobStore._claim/_recover、FilePipeline.claim/_fence、所有后台入口串行转交 remote jobs；新 coordinator 不能与原样旧 claimant 同时启用。远程 I/O 不沿用本地 10 秒续租，服务器自身仍 10 秒。
4. 服务器能力回调返回实际验证过的 runtime hash；claim_server 后仍走 P11 一槽及 cgroup。接入内部服务器执行、完成/失败及 attempt 记录收口，不能把普通 JobStore.finish 当文件原子发布。
5. 新路由接统一 main，挂管理员登记/撤销/issue 的已有认证能力，限制实际 TLS 代理信任与请求并发。定时调用 recover。生成公共 OpenAPI/TypeScript 时保留其他模块改动。
6. 公共 JobView/UI 读取 reason 和 location，区分 waiting_node、taking_over、server_busy/capability/over_budget；节点正常繁忙和健康失联分别显示。
7. 独立新 PostgreSQL、实际文件/额度/清理失败、服务进程崩溃、两 API 并发测试后，再授权目标库迁移。随后 C 与 P11 联验，M06 公共接线独立串行。

回退须先停新 claim、等待或明确撤销远程租约，确认有效 attempt 数为零并保留记录。已发布结果保留，不回写旧 TTL，不删除节点/attempt 表或回滚已结算字节。本提案没有自动 rollback/drop 脚本。
