# P06-REMOTE 独立协议与调度验收

2026-09-27。井井明确 A 尚未释放公共文件，先完成独立部分。**独立协议/调度候选已实现；完整 P06-REMOTE 为 in_progress。**

入口：[remote/1 与 C 交接](../specs/remote-protocol-v1.md)、[独立 ADR](../decisions/ADR-P06-REMOTE-001.md)、[迁移与串行接线](p06-remote-migration-review.md)。

## 状态与边界

| 范围 | 状态与证据 |
| --- | --- |
| 协议 | independent synthetic verified：Pydantic 单一源、独立 OpenAPI/示例生成、HTTPS scheme/鉴权/块上限/幂等路由测试；尚未挂统一宿主 |
| 调度 | independent synthetic verified：真实 SQLite 多连接事务、节点竞争、节点服务器竞争、账号/节点槽/节点合计内存/服务器一槽分离、有界等待/重试/deadline、恢复冷却 |
| 服务器回退 | synthetic verified：租约后接管、超能力/预算保留队列、恢复后新任务节点优先，服务器旧任务继续；实际云服务器回退 planned |
| 个人电脑 WSL Linux | 协议/传输/事务测试通过；科学节点 blocked 于 systemd/cgroup 委托及公共接线；未把合成 adapter 当真实节点 |
| 个人电脑 Windows | Python 独立协议测试通过；Windows 原生科学节点未实施/未测，由 C 按具体阻断再评估 |
| 实验室 Linux 实机 | 待验，本轮未连接 |
| 生产 | 未部署，现存库未迁移，无凭据配置，无 capability 开放 |

## changed_files

- `backend/src/ptb_api/remote_models.py`：remote/1 请求/响应、输入和快照；科学参数复用已有 M01/M03/M04 模型。
- `backend/src/ptb_api/remote.py`：独立路由工厂，HTTPS/短效 bearer、16 KiB 控制体、1 MiB 块、10 秒接收上限，成功响应 no-store。
- `backend/src/ptb_worker/remote_scheduler.py`：既有 JobStore 事务上的节点登记/轮换/撤销、调度/恢复、attempt fencing、分块与原子完成编排。不执行科学算法。
- `backend/src/ptb_worker/remote_files.py`：P07 同事务文件 port，明确 pending 生产 bridge，未制造第二套正式存储实现。
- `backend/migrations/007_remote.sql`：六表及活动 attempt 唯一索引提案，只在测试新库移除 namespace 后执行。
- `backend/tests/test_remote_protocol.py`：41 项远程合成测试，使用真实临时 SQLite 和明确标注 SyntheticFiles。
- `scripts/generate_remote_contracts.py`、`contracts/remote-v1.json`：独立可复现快照及由模型生成的示例，不触碰公共生成器或总 OpenAPI/TypeScript。
- 本报告、独立 ADR、remote/1、迁移审阅；旧 remote-compute 顶部追加现行入口，历史初案保留。

本轮未修改 A 的 storage/files/store/main/job_models、P11 resource_profiles/capabilities、D 的 M06、C 的节点包、前端、V2 或研究语料。工作区其他修改保留，不能算成本任务成果。没有 Git commit/push。

## 验证命令与结果

Windows cwd 为 v3 根目录，源码导入，不安装依赖：

```powershell
$env:PYTHONPATH='backend/src;packages/phonetic_core/src'
& .venv/v3-dev/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini backend/tests/test_remote_protocol.py backend/tests/test_job_policy.py backend/tests/test_job_boundary.py -q -p no:cacheprovider --junitxml=output/validation/p06-remote/windows-final.xml
& .venv/v3-dev/Scripts/python.exe -X utf8 scripts/generate_remote_contracts.py
& .venv/v3-dev/Scripts/python.exe -X utf8 scripts/generate_remote_contracts.py --check
& .venv/v3-dev/Scripts/python.exe -X utf8 scripts/check_architecture.py
& .venv/v3-dev/Scripts/python.exe -X utf8 scripts/validate_docs.py
```

Windows **44 passed**，其中远程 41、旧任务定向回归 3；8.26 秒。2 条已有 FastAPI/Starlette 弃用 warning。独立生成 check 成功，架构 errors=[]，六份 Python AST 解析成功。最终全局文档检查有 3 处缺链：README 的历史 M10-R5 EXE，以及并行 P15 runbook 指向尚未出现的两份验收文档；本任务自身文档无报错，未修改其他归属。

WSL 原生 Linux：

```powershell
wsl -d NInfer -- env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/d/PhoneticToolbox/PhoneticToolbox_v3/backend/src:/mnt/d/PhoneticToolbox/PhoneticToolbox_v3/packages/phonetic_core/src /home/ninfer/ptb-p11-20260926/venv/bin/python -m pytest -c /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/tests/pytest.ini /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/backend/tests/test_remote_protocol.py /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/backend/tests/test_job_policy.py /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/backend/tests/test_job_boundary.py --basetemp=/home/ninfer/ptb-p06-remote-20260927-b/pytest-restart -q -p no:cacheprovider --junitxml=/mnt/d/PhoneticToolbox/PhoneticToolbox_v3/output/validation/p06-remote/wsl-final.xml
```

WSL **44 passed / 1 warning，15.05 秒**，`wsl-final.xml` 已回读。测试运行于 Linux 原生项目 Python 3.11.14，独立 `/home/ninfer/ptb-p06-remote-20260927-b` 目录；从挂载工作区只读源码。无系统安装、网络/代理/防火墙变更，没有停止整个 WSL 或无关服务。

## 最低场景证据映射

| 场景 | 本轮实际证据 |
| --- | --- |
| 两节点竞争、节点服务器竞争 | 两线程/独立 SQLite 连接，只有一个 generation=1；非真实两台硬件 |
| 领取响应丢失 | 相同 request_id 返回同 attempt；12 秒后剩余租期 48 秒，不重置 |
| 下载中断 | 相同区间重读可恢复，重复字节不更新进展；停滞后撤销并接管 |
| 计算断网 | compute 阶段租约前不可接管，租约后 generation+1；旧输入读拒绝 |
| 心跳正常但上传停滞 | 心跳不改 progress_at；30 秒停滞阻断续租，recover 显式撤销 |
| 上传/complete 响应丢失 | 重复块/预留不重复计量；完成回执在新协调器及全新 Python 进程返回一致，结算次数=1 |
| 旧节点迟到 | 新 generation 后旧 read/write/complete 拒绝，不影响新 attempt |
| 取消完成竞争 | 两线程事务序列化，结果只能是 succeeded/1 次结算或 cancelled/0；另有取消先成立的确定性测试 |
| 撤销/作用域/过期凭据 | 六类 attempt API 均拒绝；轮换旧 token 失效，新 token 可用 |
| 输入到期/总 deadline | 当前输入无效拒绝；发布中推进时钟导致事务回滚，无 manifest/额度结算；租约不得超过输入/凭据 |
| 超额/清理失败占用 | SyntheticFiles 持久额度保留旧 attempt reservation，下一次预留被拒绝；未声称真实磁盘 unlink 故障已验 |
| 服务重启 | 新连接、全新 Python 解释器读取同一持久完成回执；未测正式 ASGI/systemd 服务重启 |
| 网络抖动/恢复 | 三次间隔 31 秒的健康报告不恢复；连续 10 秒健康与冷却后新任务走节点，旧服务器任务持续本地心跳不迁回 |
| 正常退出 | fail(node_shutdown) 明确撤销后有界重排；保留独立原因，不谎报网络错误或用户取消 |
| 服务器超预算/无能力/忙 | 对应 reason 持久保留队列，不放宽 max_running，不启动假计算 |

所有时钟推进、网络中断和输入有效性故障均为合成注入。HTTP TestClient 的 https origin 只验证 scheme 与路由，不证明 TLS 握手/公网 RTT。没有调用科学核心生成成果，没有任何数值/跨机端到端结论。

## WSL 实查及最小环境方案

NInfer 为 Ubuntu 24.04.4 LTS/x86_64。PATH 无 python3，但上述项目解释器可用，pytest/psycopg/pydantic/fastapi 已有。PID 1 为 init(NInfer)，无 systemctl、无 `/run/systemd/system`，cgroup v2 controllers 可读但根目录只读、无 writable subtree delegation。不能只加环境变量或把 trusted-worker 改名 server-small 绕过此门。

可先继续该环境的纯协议验证。真实科学节点需要获授权的独立 Linux 环境，具备 systemd 用户管理器、可委托 memory/cpu/pids、受限进程组以及 P11 运行时/科学 receipt；具体环境安装/配置另行列出并获授权后执行。也可由 C 评估现有 Windows Job Object 的补充 adapter，但本任务不同时建设第二套客户端。没有修改 .wslconfig、发行版、系统包或全局科学运行时。

## 剩余工作与来源

A 释放后的准确顺序在迁移审阅中。正式 RemoteFiles/P07 发布、旧调度器兼容、服务器执行器收口、节点运行时能力门、管理员 HTTP、统一 UI 和总契约尚未接通。真实 PostgreSQL、磁盘崩溃/清理、真实 HTTPS、云端接管和科学对照尚待。由于公共接线未开放，本轮不自行尝试云端凭据或生产部署。

复用既有 Pydantic/FastAPI/psycopg 与 Python/SQLite，不新增外部算法、第三方代码或依赖，来源注册表不变。第一次 Windows 测试 25 passed/1 failed：测试先启动服务器任务再推进 40 秒健康时钟，导致合法的 10 秒租约过期；修正测试顺序，未放宽产品租约。后续协议校准、C 时间字段反馈、资源合计及实际 runtime hash 记录均通过新增/定向测试后收口。
