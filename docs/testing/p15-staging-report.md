# P15-STAGING 准备与独立验证报告

2026-09-27。**状态：可审阅待部署，准备件已形成；真实主机/域名/节点联合验收未执行。** 不标已上线，不把本机double测试标为真实TLS/PG或完整节点。

交付：[运行/回滚手册](../deployment/p15-staging-runbook.md)、[能力清单](../deployment/p15-capabilities.md)、[实际验收矩阵](p15-acceptance-cases.md)、[独立计划](../plans/2026-09-27-p15-staging.md)、[宿主决策](../decisions/p15-staging-host.md)。脚本为 `scripts/p15_staging/`，模板为 `deployment/p15-staging/`。

## 实际完成

1. 聚焦读取根/相关规则和全部指定前置，保留工作树已有修改。未改 A/B/C、科学算法、公共生成协议、根 README/AGENTS/台账。
2. 新增部署薄宿主，绑定回环API、同UID、policy2实际库门、release源码hash、Linux runtime receipt；维护/排空从独立控制文件读取，未审科学/remote/ZIP/retry保持关闭。启动恢复副作用写入手册，不称只读。
3. 形成同源HTTPS代理、user unit、公共配置、日志轮转审阅模板；无实际密码/私钥/.env。模板不自行安装或发布。
4. 新增只读Linux/source盘点、迁移后DB/存储绑定预检、P11原始证据复核、HTTPS双账号合成检查器，可选LPC真实任务检查。现场故障/旧数据/精确政策/负载/节点场景均有明确验收步骤和边界。

## 前置核对

| 前置 | 当前核实状态 |
| --- | --- |
| P07接线 | A最新[收口](p07-policy-closeout-report.md)/[交接](2026-09-27-prerequisites-handoff.md)已到，源码确认FileConfig=QUOTA_BYTES、完整snapshot期限分类、最终发布policy2；旧manifest按1读取 |
| P07生成契约 | 初次在A修改中读到openapi/envelope drift；A完成后本任务重新执行Python `--check` 与npm `contracts:check`均通过。未来B改路由后须再生成 |
| 目标数据库 | **未验证在线版本/未迁移**。A目标本地库尚待具体授权，云目标库身份也未提供；本任务未启动库、未读秘密配置、未执行DDL，不因SQL存在声称生效 |
| 1GB/3天实际行为 | A报告Windows169项主套件、独立PG迁移/发布/超额/删除故障证据。P15 driver在ASGI doubles验证字段/序列；真实Linux PG/域名边界未验 |
| P11最终负载 | 本任务独立校验657份下载证据，0 hash mismatch；runtime2与mixed报告、三个模块receipt绑定一致，1800.520071秒、success/30min gate均真 |
| P11新包适用性 | 对runtime profile中161份backend/core Python源码比较，检查时24份与当前源码不同。历史合成负载有效，**当前整合包性能未验证** |
| A交接 | 读到新文件与明确释放标记规则；尚不能将A本地库审批沿用到云端。具体版本/hash见其delivery |
| B协议 | [remote/1](../specs/remote-protocol-v1.md)候选和007已出现；生产桥/公共接线仍需其交付，不能同旧claimant混跑 |
| C客户端 | [C交接](../specs/p06-node-handoff.md)列出相对租约时间/生成响应模型/受限adapter待接，CLI仍UnavailableBinding/P11Runtime；未声称已领真实任务 |

P11证据：`output/validation/p15-staging/p11-audit.json`。runtime SHA-256 `c17de6420e64f951a8f2a2e49c1841dba49f9bdd74231661f01c4754a6d6ccad`，mixed报告 SHA-256 `3230e4198660742a6d69d5c4e4ff335aaac70c6e76d3da79b23aa789e152c8a7`。P11正常负载峰值约215.7MiB、主机最低可用约1.99GiB，仅限定其短合成输入。A交接提到的226,136,064字节是API collector范围，P11最终226,177,024字节还含独立诊断最大值，两者不合并成同一测量。

## WSL真实盘点

NInfer正在运行，UID1001/ninfer、Linux6.6.87.2-microsoft-standard-WSL2/x86_64。PID1为init(NInfer)，PATH无python3，但现有项目 `/home/ninfer/ptb-p11-20260926/venv/bin/python` 为Linux CPython3.11.14。cgroup2存在，CPU/memory/pids控制器可见，但无systemctl/systemd-run，用户manager不可用，cgroup根不可写。**不能完成P11硬限制执行或完整节点验收。**

独立源码测试在 `/home/ninfer/ptb-p15-20260927/<UTC时间>/` 下进行，backend/core/tests/P15源码只复制到新目录；不改原venv或WSL配置。既有localhost代理告警保留，测试离线，不修改代理。准确路径见 `output/validation/p15-staging/wsl-snapshot.json`，原始盘点见同目录`wsl-inventory-*.json`。

| 轴 | 结果 |
| --- | --- |
| 个人电脑WSL Linux | 独立Python/ASGI组件测试通过；受限科学节点blocked |
| 个人电脑Windows原生节点 | 未测试，未建设第二套客户端 |
| 云服务器回退 | 未测试；旧P11服务器单独计算证据不代替节点故障接管 |
| 实验室Linux实机 | 待验，必须独立登记身份和运行时 |
| 真实主机/域名HTTPS | 未执行本轮连接/部署/测试，等待具体目标与授权 |

## 独立测试与命令

本轮先写回归，首次因host尚未实现而收集失败，随后实现通过。**最终 Windows 27 passed / 0 skipped，WSL原生Linux 27 passed / 0 skipped**，以`windows-final.xml`/`wsl-final.xml`为准，不累加重复执行。Windows既有2条框架弃用warning、WSL既有1条warning保留。最后补验不支持的旧queued操作会在claim前暂停领取，保持排队；这只是服务器独立版的保守门，不代替B的能力路由。

模板渲染/check-config通过；架构检查`errors=[]`。全局文档检查扫描860份文件，退出1，仅根README原有M10-R5 EXE缺链，P15新增文档无缺链。历史快照未解析链接另外列出，未为修复它们删除/重建包。

```powershell
$env:PYTHONPATH='D:/PhoneticToolbox/PhoneticToolbox_v3/backend/src;D:/PhoneticToolbox/PhoneticToolbox_v3/packages/phonetic_core/src'
& '.venv/v3-dev/Scripts/python.exe' -m pytest -c tests/pytest.ini tests/staging/test_p15_staging.py -q -p no:cacheprovider --junitxml=output/validation/p15-staging/windows-final.xml
& '.venv/v3-dev/Scripts/python.exe' scripts/p15_staging/audit_p11.py --evidence output/validation/p11-perf-20260927/server --repo . --output output/validation/p15-staging/p11-audit.json
& '.venv/v3-dev/Scripts/python.exe' scripts/generate_contracts.py --check
npm --prefix frontend run contracts:check
```

WSL用同一测试文件快照、原生解释器与绝对PYTHONPATH，`-p no:cacheprovider`，junit写回P15证据目录。测试覆盖配置拒绝、维护/排空、hash漂移、真实代理中间件信任边界、Host/CSRF、双账号/资产/Range/三天响应及LPC驱动完整性。账号/存储/科学结果为明确double，测试通过仅证明脚本与HTTP适配逻辑。

模板渲染为 `review-example`，示例域名staging.example.org，check-config通过；没有在系统目录安装。Nginx/systemd实际目标版本语法验证未做，WSL未安装这些工具。完整DNS/证书/PG/账号/浏览器/科学/重启/故障/负载需要真实环境。

## 准确缺陷交接与部署阻断

### P15-F01：服务器ZIP能力宣告与执行拒绝不一致（所属公共API/P11/B整合）

`backend/src/ptb_api/main.py:110` 在file_jobs可用时无条件把archive_zip/extract_zip放入task_operations，尽管server-small执行层明确拒绝。本任务ASGI中仅配置ready storage/files、Linux资格空集，即得到`pipeline_check/storage_check/archive_zip/extract_zip`。证据`capability-audit.json`，无DB或真实主机。

期望：正式有效capabilities排除未准入操作，并与部署/调度实际允许范围一致；原Windows桌面ZIP保持不变。由原公共负责人串行修复及生成协议后，P15只读/HTTPS检查器复验。P15不改main.py，不通过隐藏按钮宣称已解决。宿主已拒绝这些写路由，真实公开验收仍会因宣告差异失败。

### P15-G02：B/C/P11节点绑定未闭合

C交接指出相对租约/响应模型与trusted-worker adapter门；当前P15远程始终关闭。不得拿C本机TLS组件模拟声称真实云端领取/故障回退。待其冻结交接后新增精确binding、独立007审批、云端与个人节点任务，实验室实机保持待验。

### P15-G03：目标恢复/身份/安装版本待确认

云主机标识、域名、库/存储instance、当前DNS/代理/服务/PG版本/备份恢复能力尚未从目标核实。已提供明确候选与审批表，不能为补全表格编造。没有安装服务、创建用户、改DNS/防火墙/凭据/系统配置、迁移、停写、重启或公开发布。

## 后续

先接收目标标识与共享文档释放情况，核对A/B/C最终交付与F01修复。按手册D1–D5明确目标审批，重建新runtime/receipt，完成真实Linux PG/HTTPS/站点与节点故障矩阵。全部稳定且其他任务释放后，才统一更新README、根AGENTS、总台账和能力矩阵，历史证据保留。当前只有P15独立清单，不提前覆盖共享文件。
