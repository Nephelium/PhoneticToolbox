# A 前置交接：政策、测试库与远程准入

2026-09-27。交付状态：**政策接线/契约/合成验证完成，目标库迁移与网页联验待具体授权**。本轮不替 B/C/D 宣告完成，不依据文件时间戳释放其他任务。

## 版本与证据入口

- 工作分支 `codex/v3-rebuild`，审阅基点 HEAD `ad165a618754b2a8354e92048ad756ec999a950a`。工作区包含多任务未提交成果，不能用 HEAD 单独代表本轮可执行版本。
- backend/core `3.0.0a1`，API `1.1.0`，政策 `2`。具体交付文件 SHA-256 见 `output/validation/p07-closeout-20260927/delivery.json`。
- [本轮代码与测试报告](p07-policy-closeout-report.md)、[006 具体迁移方案](p07-policy-migration-review.md)、[政策说明](../manual/storage-policy.md)。新 manifest 显式版本 2，旧缺字段读作 1，所有服务器期限仍受实际库版本写门约束。
- `006_storage_policy.sql` SHA-256：`bf5eb0d18eb3d242db98f026c9a32c55a90bf57b21589efe36365a974866f247`。
- 本轮重新枚举已出现 B 的候选 `007_remote.sql`，观察 hash 为 `7cf890d73426129f940ca7d1298cf171b05ea8daefd3d7a3200de98f5d0a1c80`。它仍在并行开发，**不是冻结 hash，不属于本轮 apply**。下一任务不得复用 006 编号，也不得从 006 获准推导 007 获准。

## P11 最终结果复核

初读报告仍写运行中，随后读取 P11 chat 当前状态和本机 `server-evidence.tar.gz` 中的真实 `mixed-run1/report.json`。该报告 SHA-256 为 `3230e4198660742a6d69d5c4e4ff335aaac70c6e76d3da79b23aa789e152c8a7`，`success=true`、持续 **1800.520 秒**、`thirty_minute_gate_passed=true`、API 已退出。P11 专项文档随后更新为限定 verified。没有再次连接服务器、启动负载或修改其固定测试包。

最终 profile SHA-256 `c17de6420e64f951a8f2a2e49c1841dba49f9bdd74231661f01c4754a6d6ccad`。53,467 次 HTTP 响应，非预期失败 0，33 次预览 busy 单列；新增任务 133 成功、76 计划取消，实际启动 358 个 API 计算组均清理。资源观测与限制详见 [P11-PERF](p11-perf-report.md)：整机最低 MemAvailable 2,134,528,000 字节，观测组峰值 226,136,064 字节，两个 API 各自 PSS 峰值 295,398,400 / 268,145,664 字节。这些测量均不等同于配置上限。

服务器配置为同 UID/同命名空间单槽，科学进程组最多 1,073,741,824 字节，CPUQuota=100%、TasksMax=64、Swap=0。保留更低模块限制。公共预览仍可能排队约 20 秒并返回 busy；API PSS 后段仍有增长，长时常驻稳定性未闭合。没有真实账号 PG、公网 RTT、长录音/最大输入或生产部署结论。

**P11 固定包未包含本轮政策/manifest 与后续远程改动。** 新部署必须按整合后实际文件重建 runtime profile 并运行真实任务生成匹配 receipt，不能把旧 receipt 的 success/hash 字段手改后使用。

## 逐模块矩阵

V=报告范围内 verified，P=planned/尚无相应证据，B=阻断，I=in_progress。历史网页验证不代表新政策已生效。所有远程资格仅指接入候选，当前无已验真实节点。

| 模块 | Windows | Linux 正式任务 | 小服务器预算/证据 | 网页账号 | 新政策 | 远程可接入资格 |
| --- | --- | --- | --- | --- | --- | --- |
| M01 参数估计 | V，原功能收口 | V，P11 固定包合成任务 | 单槽/模块限额与 30 分钟合成负载 V | 历史限定 V，新政策待验 | 公共发布合成 V，目标库 B | 候选，需新 receipt 与 B/C 联验 |
| M02 参数显示 | V，图窗/PNG | 独立科学任务不适用，公共预览有 P11 证据 | 公共预览单槽，整模块未验 | 历史限定 V | 上传/派生经公共存储，目标库 B | 无独立重算入口 |
| M03 EGG | V，开发功能 | V，P11 固定包合成任务 | 单槽/模块限额与合成负载 V | 历史限定 V，新政策待验 | 公共发布合成 V，目标库 B | 候选，需新 receipt 与 B/C 联验 |
| M04 LPC | V，Chrome/Qt/自然短 ROI | V，P11 固定包合成任务 | 单槽/模块限额与合成负载 V | 当前 B，上传被政策门拦截 | 新生成三天合成 V，目标库 B | 候选，网页与新 receipt 待验 |
| M05 唇形提取 | P，未见正式迁移交付 | P | 未验，关闭 | P | 公共底座，模块未接 | 不可接入 |
| M06 语音合成 | D 正在实施，本交接未验 | 未验 | 未验，关闭 | 未验 | 由 D 后续接本轮政策 | 待 D/B/C 明确交付 |
| M07 发声类型合成 | P | P | 未验，关闭 | P | 模块未接 | 不可接入 |
| M08 变速变调 | V，正式本地链路，本轮 20 项回归 | B，精确门 25 pass / 5 fail，能力关闭 | Windows 1,000,000,000 字节/120秒；Linux 未准入 | B，待专用 PG 迁移 | 科学/复制分开，合成 PG V | Linux 不可接入，不放宽精确门 |
| M09 语谱图转音频 | V，限定开发态 | B，Linux 未开放 | 未验，关闭 | 历史限定 V | 公共发布合成 V，目标库 B | 不可接入 |
| M10 声道工作台 | V，最新 R5 定向开发验证 | P，原生 ABI/设备未验 | 未验，关闭 | 生产未验 | 服务器任务未接 | 不可接入，保留设备边界 |
| M11 MFA | P | P | 未验，关闭 | P | 模块未接 | 不可接入 |
| M12 标注 | V，R6 限定开发/旧包 | 服务正式模块联验未完成 | 交互/文件链路，服务器负载未验 | 历史限定 V | 新上传走公共政策，目标库 B | 无已验远程任务入口 |
| M13 普通话 IPA | V，Chrome/Qt | 前端计算，不需要后台任务；WSL 静态服务 V | 无科学组任务 | 账号任务不适用 | 浏览器本地 PNG 不套服务器 TTL | 无需远程重算；生产部署另验 |
| M14 音系归纳 | V，本轮真实本地 8 组通过 | B，fixed entry/receipt 未登记 | 核心受限短测 512 MB/60秒，峰值约69.5 MB；正式任务未准入 | B，待专用 PG 迁移 | preview/export 重新分析后三天，合成 V | Linux 不可接入；核心短测不代替正式门 |

矩阵依据当前模块专项报告及实际 capability/entry 代码。M10 以[最新 R5 报告](m10-r5-report.md)为准，M13/M14 不沿用旧总表的 planned。未重测的历史模块仅引用原证据，不扩大 verified。

通用 ZIP/解压在 `server-small` 保持 `server_export_unavailable`，在读取输入和预留前拒绝；本轮 8 项原门包含在 20 项回归内。桌面原 ZIP 流程保留。通用 ZIP 性能与 Linux 受限迁移未验。`trusted-worker` 当前仍明确不可用，B/C 的新文件出现不等同于调度可用。

## 授权后的网页联验顺序

1. 仅对迁移审阅中的本地专用 PG 执行 006，独立核对 preserved 指纹、实际 policy/quota、实例及磁盘。授权不包含 007 或任何云服务器库。
2. 用新建合成账号执行上传后三天、两个账号/project/hash 隔离、超额保留下载删除/禁止增量、到期禁读、删除失败不释放账本、旧 generation 发布拒绝。受控时钟/到期注入需标明，不称自然三天。
3. M04 复用 `scripts/verify_m04_web.py` / `tests/e2e/m04-web.cjs`：真实双账号、LPC、历史和三文件下载。该现成脚本未覆盖的超额/到期/迟到 worker 由公共定向检查追加，不能仅脚本 exit 0 就全项通过。
4. M08 复用 `scripts/verify_m08_pg_gate.py` 的只读门与 `verify_m08_wiring.py` 的正式链路步骤，适配到获准 PG/认证账号；当前没有完成的双账号 PG 验收脚本，需在该授权阶段补齐。重点比较科学新结果与保存副本 PCM/hash/截止；Linux 仍关闭。
5. M14 复用 `scripts/verify_m14_wiring.py` 的五格式与三文件回读，适配 PG/认证账号，验证新政策及取消/旧 worker。Windows 实际环境可验；Linux 只验明确不可用响应，直到 P11 完成 fixed entry、collector 和真实 receipt。

本轮没有运行会启动/停止目标专用集群并写入的上述网页脚本，没有将本地 SQLite 测试冒充账号网页验收。

## 公共文件归属与释放

本轮公共变更限政策模型、`files.py`/`acoustic_files.py`/`local_acoustic_files.py`、`store.py` 的旧预算 retry 错误保护及生成协议。`main.py`、executor/process_entry、P11 资源实现、科学核心、M08/M14 路由保留；本轮未改这些接点。

收尾观察到 D 已向公共模型/发布器加入 M06，A 已保留并从当时最终模型重生成。两侧契约 check 与 124 项政策/契约回归通过，但全局 typecheck 有 D 新代码的 3 项 TS2352（speech-synthesis/state.ts:4、:6，platform/m06.ts:20），须 D 完成后复验。A 的初轮 typecheck/140 项前端通过只对应 M06 加入前的快照。M06 期限分类及正式任务仍由 D 交付，本轮未授予其 Linux 或远程资格。

最终交付 `delivery.json` 中的 `public_files_released=true` 才是 A 的显式释放标记。释放后 B 可串行修改公共任务/契约；需先读取当前 diff 和 hash，并从最终模型重生成。C 继续节点客户端，D 继续 M06，P15 按整合后 receipt/政策和自身授权推进。A 不给其他任务发送实施指令，也不从其他任务时间戳推定其完成。

已生成可复查的契约并验证 M08/M14 类型仍在。后续修改任何模型或路由后必须重跑 generator/check、contracts:check 和相关回归。旧 P11 包/receipt、V2、用户语料、旧 EXE 及各任务未提交成果均保留。
