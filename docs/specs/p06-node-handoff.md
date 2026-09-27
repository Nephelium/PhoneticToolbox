# C 节点 → B / P11 精确接口交接

2026-09-27，P06-REMOTE-NODE。本文件仅提出接口需求，不修改公共文件/生成契约。

本轮后续复核：B 已在候选源码增加 LeaseResponse（服务器时间/剩余租约/deadline）、
InputAsset（输入剩余期限）和其他响应模型，另增加 node_shutdown 错误码。
以下第 1、2 项是原始交接需求，代码层已开始解决；仍以 B 最终冻结、生成契约与联验为准。
节点 Session 已据此将输入剩余期限纳入截止，并在快速计算前后显式按序发送 phase 转换，
避免只依赖周期心跳导致跳过 compute。实际 wire binding 仍未开放。

## B remote/1 候选现状与接线门

本轮已读取 `remote-protocol-v1.md`、`ptb_api/remote_models.py`、`remote.py`、
`ptb_worker/remote_scheduler.py`。这是 B 的唯一协议源，C 没有创建替代 API。

必须冻结/补齐后才能开启领取：

1. `_manifest()` 当前只有 `lease_until`/`deadline`，`heartbeat()` 只有 `lease_until`。
   文档 poll 表中描述的 `server_time` 尚未随当前 claim 返回。
   请在同一事务时点返回 `server_time` 与 `lease_until`，或直接返回精确 `lease_remaining_seconds`。
   同时提供 `deadline_remaining_seconds` 与输入 expiry 的服务器相对时长。客户端使用
   `request_started_boottime + remaining - safety_margin`，不以本机墙钟减服务器时间。
   领取回包丢失时，相同 request_id 重试仍须返回**剩余**租约，不能返回初始 60 秒。
2. 请为 claim/heartbeat/health/输出预留/offset/complete 响应及 inputs 元素提供生成模型。
   目前 `inputs: list[dict]` / `snapshot: dict` 没有冻结字段；C 不把它们写成第二份 schema。
   `protocol`/generation/operation/runtime_hash/参数 hash/字体 hash/预算均须严格校验。
3. GET/PUT 的 generation/offset/size 为非秘密 query 参数，C transport 将按冻结路由单独编码，
   凭据只能进 Authorization。目前通用 transport 禁止所有 query，待 binding 增加受限数值参数支持。
4. B 当前登记凭据 15 分钟有效，只有服务器管理员可 issue/轮换。请明确人工安全更新文件流程或
   新授权的续期机制，C 不自造永久密钥/刷新端点。当前组件可读取 owner-only 文件，未实际登记。
5. 没有 goodbye endpoint。正常退出有活动 attempt 时拟调用已有 fail(cancelled)，网络失败让租约回收。
   空闲退出无需编造接口。请确认停机重排是否希望使用另一批准码，避免用 network_error 谎报退出。
6. 完成后的 complete 重试、poll request_id 重试、upload offset/块 hash 以 B 的幂等语义为准。
   不使用新 generation 回传旧结果。服务器已执行任务不会由 C 请求迁回。

## P11 公共 adapter 需求

当前 `selected_profile()` 明确拒绝 trusted-worker，C 保留该门，不改环境变量冒充 server-small。
请由公共文件负责人串行提供以下能力，具体命名以其实现为准：

负责人本轮已核对并记入 P11 报告：既有固定入口、stop 回调、清理证据及 on_chunk
可复用；节点预算、正式绑定、独立恢复接口和端到端流式回传仍待接通。
本次没有把该交接扩大为 P11-PERF 以外的实现授权，trusted-worker 继续关闭。

- 本机可信配置传入 process-group memory_bytes、CPU quota、TasksMax、swap=0、单槽身份和临时磁盘预算。
  对所有科学/导出/原生子进程合计施加限制，安装限制后才加载科学库。
- 固定 entry + 已校验版本化请求文件 + 本地独占 scratch + cancellation predicate，复用同版 core。
  不接收远程 argv/python/module/path；operation allowlist 与双方 receipt 求交集。
- 单独返回 abort/recover handle 与清理证明，记录精确随机 unit/进程组，重启只回收本节点拥有对象。
  强杀前要求可验证身份，清理失败阻断下一槽。P11 当前 journal/admission 机制应复用。
- 输出需要有界流式 sink / 固定输出文件清单，使节点不把整个 bundle 聚合到内存。
  当前 `collect_scientific -> bytes` 不能直接证明大批结果的节点流式回传预算。
- capability receipt 包含 core/adapter/version、BLAS 构建、字体文件和模型 hash、实际模块报告 hash。
  缺字体/版本不匹配/无硬限制均关闭对应能力，纯数据与图形任务按原语义区分。

## 已有组件

`ptb_node.lease.Lease/Heartbeat`：绑定 identity，独立 watchdog/heartbeat，CLOCK_BOOTTIME，
休眠偏移变化直接弃本 attempt，不复活旧结果。`transport`：TLS 校验、固定 origin、
有界分块/hash/offset/重试、独立传输中止。`storage`：私有目录/锁/本地随机文件/残留清理。
`service`：AF_UNIX 控制与健康重连 port。当前 CLI 固定 UnavailableBinding/P11Runtime，
并不 poll/执行/上传实际任务。移除门之前需 B/P11 联合用例。

## 必须保留的真实联验

节点完成 → 节点自有进程/连接故障 → 租约失效 → 合格服务器接管 → 节点恢复 →
后续任务回到节点，服务器执行中任务不迁移。旧 attempt 的读、写、complete 全部拒收，
最终恰好一份结果与一次额度结算。超预算/无科学能力时保持队列。
组件 fake 和本机 TLS 测试不能证明这些服务器事务行为。
