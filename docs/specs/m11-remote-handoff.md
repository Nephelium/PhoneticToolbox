# M11 → P06-REMOTE / P11 精确接线需求

2026-09-27，`blocked / integration pending`。本模块不创建第二队列、节点认证或私有上传协议。现有远程候选状态依据 [P06-REMOTE 报告](../testing/p06-remote-report.md)、[remote/1](remote-protocol-v1.md) 及 `packages/ptb_node/src/ptb_node/runtime.py` 的未接通能力。

## 已交付正式边界

- `POST /api/v1/jobs/m11/create`，请求 `M11Request`，operation=`mfa_alignment`，schema=`m11/1`。
- 原 JobStore 快照包含 `request`、输入 ID／hash／字节／到期时间、runtime fingerprint、模型／词典 hash、`config.max_output_bytes=64_000_000`。
- PostgreSQL 任务的 `execution_route=remote_pending`、`waiting_reason=m11_waiting_verified_node`。普通 worker claim 通过参数化过滤跳过；排队、取消、到期与 owner/project 继续用原平台。
- 节点离线或未合格不得清除此门，也不得修改 Beam／模型／语料。排队截止不超过输入到期时间前一秒，不延长输入 TTL。
- 模型由宿主允许清单登记。`model_id`／`runtime_id` 是 ID，不是用户路径或 URL。服务器 API 没有本地组件安装权限。

## 公共桥需要完成的工作

| 归属 | 精确要求与退出条件 |
| --- | --- |
| P06-REMOTE 的 `remote_models.py` | 在已有 operation 与 snapshot 联合类型加入 M11Request／M11Manifest，校验版本、模型、运行时指纹与 Beam。统一生成 remote/1，不能手写第二套契约。 |
| `remote_scheduler.py` + 原 JobStore／RemoteFilePort | 读取原 queued 任务、原子租约／generation fencing、优先实验室已验证 Linux 节点；输出进入既有 P07 原子发布。不能只返回独立 SyntheticFiles 测试结果。 |
| 节点 `P11Runtime.validate/execute` | 验证独立 **Linux** 3.3.8 运行时及模型探针凭据，使用 P11 受限进程组。Windows 文件包不可充作 Linux 环境。节点不持账号库权限。 |
| P11 资源门 | MFA 原生临时目录、SQLite、JIT cache 与中间文件不能绕过配额 writer。需要目录写入预留／计量和硬预算后再开放服务器；当前本机磁盘轮询仅是软上限。 |
| 输出／错误 | TextGrid 真实解析、完整数量、hash、有效时间区间；`m11-provenance.json`；任务及资源版本；取消／超时／OOV／模型错误仍通过现有 fail/complete。 |
| 平台验收 | 断连、重启、取消与发布竞争、过期、旧节点恢复、并发账户隔离、上传中配额失败、无合格节点持续等待分别实测；实际服务器、个人 WSL、实验室节点分别记证据。 |

当前全词典 Windows 小任务峰值约 1.76 GB，已高于初始服务器约 1 GiB 预算。不得以小词典约 0.54 GB 或 num_jobs=1 推断全词典可在服务器执行。Windows 配置上限 2 GiB 也不是 Linux 或服务器准入证据。

未执行现存库 DDL、生产部署、节点登记或凭据写入。以上是串行接线合同，不是已接通远程计算的声明。
