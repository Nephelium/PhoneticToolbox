# P06-REMOTE remote/1 · C 接线协议

2026-09-27。协议实现候选；A 尚未释放公共文件。此文档供 C 复用，不能另造协议。模型源 `backend/src/ptb_api/remote_models.py`，独立生成入口 `scripts/generate_remote_contracts.py`；不修改公共 OpenAPI 和 TypeScript 快照。

## 接口

统一宿主的 `/api/v1/worker` 路由工厂，尚不挂 main.py。所有节点请求只接受 HTTPS Authorization Bearer 短效凭据，禁止 query token。管理员登记/撤销由服务器侧方法提供，公共管理员认证接线前不暴露登记 HTTP。

| 方法/路径 | 请求 | 响应 |
| --- | --- | --- |
| POST /poll | request_id UUID、runtime_hash SHA256 | Claim 或 null；Claim 带 server_time |
| POST /health | runtime_hash | server_time、健康连续计数 |
| POST /attempts/{attempt_id}/heartbeat | generation、phase、node_bytes | LeaseResponse；仅传输接口实际新增字节推进传输时钟 |
| GET /attempts/{attempt_id}/inputs/{asset_id} | generation、offset、size≤1 MiB | 字节块，SHA256 响应头 |
| POST /attempts/{attempt_id}/outputs | generation、key、name、size_bytes、sha256 | upload_id、offset |
| PUT /attempts/{attempt_id}/outputs/{upload_id} | generation、offset、X-Chunk-SHA256、≤1 MiB 原始块 | offset；相同块幂等 |
| POST /attempts/{attempt_id}/complete | generation | `{result_manifest: ...}`，重试不重复发布；内部 coordinator 返回 manifest 本体 |
| POST /attempts/{attempt_id}/fail | generation、code | accepted；network_error/transfer_stalled/node_shutdown 可有界重排 |

领取响应包含 protocol、attempt_id、job_id、generation、lease_until、deadline、location、operation、runtime_hash、参数/字体 hash、输入清单、预算和不可变参数快照。输入/输出 UUID 由服务器控制；没有下载 URL 或机器路径。请求模型 extra=forbid。

Claim/LeaseResponse 提供同事务 `server_time`、`lease_remaining_seconds`、`deadline_remaining_seconds`；InputAsset 提供 `expires_remaining_seconds`。客户端以请求开始时的 BOOTTIME 加 remaining 再减安全余量，不用客户端墙钟。相同 request_id 领取重试不会续租，返回当时剩余值。租约不超过凭据、输入和总 deadline。Claim 的 snapshot 仅包含 operation/core_version/adapter_version/config/input_refs，去除服务器项目/批次 metadata，科学 analysis 复用现有 LpcTaskConfig/EggTaskConfig/AcousticConfigSnapshot。参数 hash 为 config 的 canonical JSON SHA256，字体 hash 为 config.analysis.font（无字体时 null）；input_hash 为输入清单在添加 remaining 字段之前的 canonical SHA256。示例由模型生成在 `contracts/remote-v1.json` 的 examples。

POST 控制体上限 16 KiB，PUT 块上限 1 MiB；两者接收超时 10 秒。超过限制返回 413，接收停滞 408。成功响应 no-store。公开协议允许 generation/offset/size 数值 query，不允许凭据 query。TLS 反向代理信任范围由后续部署配置明确，本路由不自行相信任意 Forwarded header。

管理员登记明确 scopes=[owner_id,project_id] 配对、operation allowlist、服务器审核的 runtime_hash、slots 与 memory_bytes。凭据只返回一次，库中存 hash，期限默认 15 分钟；轮换由已有安全运维入口调用 register/issue，不在节点协议中放长期密钥。撤销立即使后续请求无权，撤销租约后才能重排。

C 交接决议：管理员通过既有安全通道更新节点 owner-only 凭据文件，调用 issue 后旧 token 立即无效；不自造刷新端点。空闲正常退出无需请求，活动退出使用新增 `fail(node_shutdown)`，明确撤销本 attempt 租约后有界重排。`fail(cancelled)` 表示取消任务，不能拿它表示普通退出。退出请求丢失时由原租约到期回收。node_shutdown 在历史中独立记录，不冒充网络错误。凭据撤销本候选结束受影响任务，不将鉴权错误自动无限重试；管理员重新授权后的人工重试另走既有任务入口。

## 状态与错误

保留 queued/running/cancel_requested/succeeded/failed/cancelled/interrupted。路由 reason 为 waiting_node / waiting_server / taking_over / server_capability_unavailable / server_over_budget / server_busy / account_busy。attempt 历史保存 location、runtime、输入/参数/字体 hash、失败码、起止时间及 node_bytes。

401 node_unauthorized；403 node_scope_denied；409 stale_attempt/cancelled/idempotency_conflict/transfer_stalled；410 input_unavailable/deadline_exceeded；413 quota_exceeded/output_budget_exceeded/invalid_chunk；422 invalid_request/hash_mismatch；503 remote_files_unavailable。无原始异常、凭据或路径进入响应。

complete 已提交后的相同 attempt/generation 重试只读取原 manifest；仍验证节点身份、scope 和当前 generation，不重新开放输入/写入，也不延长输入/结果 TTL。后来重试的不同 generation 永远不能读取此前完成回执。

## 文件适配与公共交接

RemoteFiles port 的 validate_inputs/read/reserve/append/seal/publish 必须复用同一 `tx`。publish 只准备并提交 P07 manifest/额度/资产可见性，任务 succeeded 与 attempt receipt 由调度器在同一事务更新。事务回滚必须回滚全部可见状态。写盘崩溃保留 P07 reservation；cleanup 仅在确认物理回收后释放。禁止把现有另开连接的 FilePipeline.complete 原样塞入 port。

目前独立测试使用明确标注的合成 SQLite 文件适配，仅验证接口事务与故障语义。A 释放后再实现生产桥：按 Storage → scheduling 锁顺序；修复 FilePipeline._fence 目前所有 I/O 都续本地 10 秒租约的行为，使远程按 attempt 规则续租；所有旧 worker claim/recover 路径必须排除/转交 remote_jobs。不得同时运行旧 claimant 与未接线的新调度器。

server capability 回调必须复用 P11 实际 runtime receipt 与资源 profile，不接收节点自报能力作为服务器能力。执行前仍须 P11 Admission，不以数据库一槽替代 cgroup。M06 后续串行加入 allowlist；实时设备/桌面操作永不加入。

该回调返回实际经验证的服务器 runtime SHA256，关闭时返回 None，不能返回布尔 True。若服务器与节点 runtime 不同，必须由公共能力适配验证对应科学等价证据，attempt 记录实际服务器 hash。远程 operation 初始限 M01/M03/M04，登记仅供管理员；独立实现没有开放任何当前 capability。每账号默认 1 个运行任务、服务器 1 槽、节点登记槽数及节点合计内存、最多 32 个被接纳 remote 活动任务分别检查，绝不调大 store.max_running。

`remote_jobs.reason` 已持久保存等待原因；公共 JobView/事件/UI 尚未接线，当前不能声称页面已显示它们。生产桥须在相同事务把原因暴露到既有任务界面，并保留现有终态兼容。

## 最低验收计划

独立临时数据库：两节点/节点服务器竞争，claim 回包丢失，下载中断，断网租约，上传停滞但心跳正常，complete 回包丢失，迟到节点，取消竞争，撤销，输入到期，超额，重启，抖动，超预算，以及恢复后新任务节点优先且服务器旧任务不迁移。逐个确认唯一 generation/有效租约/manifest 与结算次数。

后续真实门：公共接线后的独立 PostgreSQL/P07 合成测试 → 双方已有证据的科学任务数值及文件对照 → 真实 HTTPS 故障 → 实验室 Linux。WSL 初查无 systemd、无 cgroup 委托，不关闭检查。个人电脑测试不能替代实验室验收。
