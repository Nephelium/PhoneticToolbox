# P15-STAGING 部署、排空与回滚手册

2026-09-27。**可审阅准备件，未上线。** [专项计划](../plans/2026-09-27-p15-staging.md) · [验收/阻断](../testing/p15-staging-report.md) · [宿主决策](../decisions/p15-staging-host.md) · [开放候选](p15-capabilities.md)。

## 1. 明确目标与配置

下表是待审核目标，不是已经创建的设施。用户未提供的标识保留 unknown，禁止猜测或套用其他项目。

| 项目 | 候选与当前事实 |
| --- | --- |
| 云主机 | 用户既有阿里云 Ubuntu 24.04.2/x86_64、2 vCPU、系统内存 3.42 GiB、无 swap。主机实例 ID/SSH 别名/IP 待井井指定，当前未重新连接 |
| 域名 | 待指定 staging 域名，`REPLACE_DOMAIN` 不可直接启动；DNS 当前值与所有权待只读核对 |
| 专用 Unix 用户 | 建议 `ptb-staging`，home `/var/lib/ptb-staging/home`，三服务同 UID，同一 user bus/cgroup/挂载命名空间；创建账号、linger 和资源委托需要目标授权 |
| 应用目录 | `/srv/ptb-staging/releases/<release-id>`，固定绝对路径，旧发行目录保留；应用源码/venv/assets 版本同清单，不复制 Windows venv |
| 配置 | `/etc/ptb-staging/config.json` 公共配置；`private.json` 仅 dsn/signing_key/storage_instance_id，专用用户所有、0600，父目录限制遍历；不进 Git/argv/日志 |
| 控制文件 | `/var/lib/ptb-staging/control.json`，初始 maintenance，原子替换 open/drain/maintenance。必须 root 或任务专用用户可写，路径不在资产根 |
| 内部监听 | API `127.0.0.1:18765`，先查端口占用；PostgreSQL 使用已有回环或 Unix socket，不开放公网 5432，不把 node 接到 DB |
| 目标数据库 | 建议新专用 `ptb_staging_20260927`、专用最小运行角色，实例/PGDATA/端口及现有库清单待核实；建库/角色/DDL 均未执行 |
| 存储 | `/var/lib/ptb-staging/assets`，空库时显式生成 instance UUID 和 marker/lock，与 DB state 完全匹配；已有根只验证、不初始化、不清空 |
| 代理/TLS | Nginx 同源 443 → 回环 API。域名绑定证书及链、TLS 1.2/1.3；80 仅重定向。证书签发/续期方案按目标已有工具审阅，不在准备时安装 |
| 运行时 | CPython 3.11.14；既有已验版本 NumPy 2.2.6、SciPy 1.16.3、Parselmouth 0.4.7、Matplotlib 3.10.8、pandas 2.3.3，backend/core 3.0.0a1。实际整合 wheel/源码/原生库/hash 必须重新锁定 |
| 字体/原生 | 中文 Noto Sans SC、拉丁 DejaVu Sans、IPA Doulos SIL 的文件 hash；M01 Linux REAPER 固定构建与 hash。字体预检针对执行环境。未配置 MFA/视觉/声道模型时能力关闭 |

现有恢复能力只确认：本机 Git 与历史发行/测试证据保留，P07 事务在 COMMIT 前可回滚，运行库能恢复有效预留和任务租约。**没有确认云端自动快照、PITR、WAL 归档或可恢复的配套存储副本，源码/Git/元数据 hash 不是数据备份。** 先只读列出目标既有策略、保留期和最近恢复演练。确需新增时提出一次性 DB＋对应资产的范围/空间/一致性窗口/加密/3 天政策处理，待单独确认；本任务不创建全量备份。

## 2. 准备件与本地命令

模板：[公共配置](../../deployment/p15-staging/config.json.in)、[用户服务](../../deployment/p15-staging/ptb-staging@.service.in)、[代理](../../deployment/p15-staging/nginx.conf.in)、[代理头](../../deployment/p15-staging/proxy.conf.in)、[轮转](../../deployment/p15-staging/logrotate.conf.in)。不包括可执行 apply 或秘密示例值。

以下 PowerShell 只生成审阅副本。示例域名没有被视为实际目标：

```powershell
$env:PYTHONPATH='D:/PhoneticToolbox/PhoneticToolbox_v3/backend/src;D:/PhoneticToolbox/PhoneticToolbox_v3/packages/phonetic_core/src'
& '.venv/v3-dev/Scripts/python.exe' scripts/p15_staging/render.py --templates deployment/p15-staging --output output/validation/p15-staging/review-example --domain staging.example.org --release p15-review-1
& '.venv/v3-dev/Scripts/python.exe' scripts/p15_staging/host.py check-config --config output/validation/p15-staging/review-example/config.json
& '.venv/v3-dev/Scripts/python.exe' scripts/p15_staging/probe.py --root . --output output/validation/p15-staging/source-inventory.json
& '.venv/v3-dev/Scripts/python.exe' scripts/generate_contracts.py --check
npm --prefix frontend run contracts:check
```

重跑生成器必须用新输出目录，旧证据保留。`probe.py --seal-release` 仅生成内容清单，不能生成科学成功 receipt 或部署授权。

## 3. 授权前只读清单

对指定主机检查 hostname/OS/UID/架构、`ss -ltnp`、磁盘、`systemctl --user`、user bus、cgroup 委托、已有服务与目录、PG 版本/监听和凭据提供途径。只读脚本 `probe.py` 不启动/停止服务。systemd 控制器存在不等于真实硬限额通过；后续必须验证 CPU/memory/pids/swap 限制及真实清理。

数据库使用既有安全 stdin 提供方式分别运行 `scripts/p07_policy_preflight.py`（迁移前）和 `scripts/p15_staging/probe_db.py`（政策 2 后）。后者校验 marker/DB instance、实际政策/配额、账本和活动任务，输出聚合，不写数据。DB 未运行时不能为了预检擅自启动。本机 A 专用库的授权不扩大到云端库。

冻结最终整合版本：先确认 A/B/C 发布标记、生成 contracts、前端构建、依赖/来源与最终 runtime 的 hash，再运行真实模块任务生成 receipt。P11 旧 profile `c17de642…6ccad` 不可沿用到新路径、新 wheel 或新代码。不能修改报告 success/hash 来补门。

## 4. 需要逐项批准的部署动作

| 批次 | 可审阅动作 | 前置与影响 |
| --- | --- | --- |
| D1 | 指定云主机只读连通/配置检查 | 确认目标标识，使用既有安全认证，不在对话粘贴秘密 |
| D2 | 专用用户、目录、权限、user manager/linger、环境安装 | 明确新增文件/包/资源，不升级系统或现存科学环境；同 UID 一槽必须实测 |
| D3 | 指定 DB 建库或 006 迁移、存储初始化/恢复 | 明确 instance/库/根/hash、停写窗口和恢复能力。006 与 007 分开审批；不运行固定 Windows 测试库脚本来迁移云端 |
| D4 | 指定 Nginx 站点/证书、DNS、80/443 防火墙规则、三个服务启动 | 私有 API/DB 不出公网，改动已有代理需列出现存依赖及 reload 影响；先配置检查再 reload |
| D5 | 专用 p15_ 双账号/项目与公开合成任务、定向故障 | 仅本任务服务/节点/连接，明确可停止 PID/unit/凭据身份；禁止整机网络和全 WSL 操作 |

所有目标未填/前置未过时只保留模板。不能把修改模板或命令行开关当成用户实际批准。

## 5. 获准后的实施次序

1. 记录目标与快照，确认已有恢复能力及可接受停机窗口。创建新发行目录并放入审核后的 Linux 运行环境、当前三包/前端/字体/原生资源，固定依赖来源和 hash，不拉取浮动最新版。
2. 生成实际 `runtime.json`（包含解释器、所有用到的 backend/core 源码与原生库/字体/REAPER hash）并在这个固定路径运行 P11 的 LPC→EGG→M01 短合成/故障门，取得新的 receipt。当前 `load_profile` 核对绝对路径，不能搬目录后不复测。
3. 数据库新建时顺序为 `001_accounts.sql → 002_jobs.sql → 003_storage.sql → 显式初始化唯一 state/marker/lock → 004_job_assets.sql → 005_acoustic_batches.sql → 006_storage_policy.sql`。每个 SQL 用 `psql -X -v ON_ERROR_STOP=1 -f <审核文件>`，连接由私有 libpq 配置提供。SQL 自带事务，不粗暴去掉 BEGIN/COMMIT。初始化具体 UUID/目录与事务方案由 A 针对目标审阅，不复制其固定本地路径初始化器。
4. 已有库只应用经版本检查缺失的已批准迁移，**绝不重放 001–005**。006 执行前停止所有写者，锁内复核账本。保存原资产期限/manifest/snapshot/used/reserved 的指纹，迁移后逐项对照。失败保持停写，禁止清零或自动删数据。
5. 远程首轮保持关闭，007 不执行。B 的桥接/契约、C binding 与 P11 trusted-worker 全部通过且新 DDL/节点身份另获授权后，才提供远程服务变体；不混跑旧 claimant 和 remote scheduler。
6. 以专用 UID 配置文件、user unit `ptb-staging@api.service`、`@worker.service`、`@cleaner.service`。三者固定同一源码与环境；systemd user manager 身份验证和 cgroup 实测先完成。`systemd-analyze --user verify <审核后的 unit>` 后才允许 enable/start。不要把 `User=` 改成三个不同账号。
7. 初始 maintenance，worker/cleaner 不工作，API 可提供登录和历史兼容读取。配置白名单为空时科学入口关闭；打开科学前只填当次 receipt 中完整通过的操作集合。`host.py` 启动会检查新政策 DB/存储绑定、源码 manifest、运行 UID 和 receipt，失败不自动降级。
8. 写门切为 open 前先停止三个任务服务，在已批准的恢复窗口按 API → cleaner → worker 顺序启动。open/drain 启动执行 `Storage.recover()`，会按现行规则删除已到期/删除失败文件、恢复预留、冻结未知文件，必须已获范围授权。不得仅编辑 maintenance 为 open 就当完整恢复完成。
9. 代理模板安装到新增站点/snippet，保持已有站点；运行 `nginx -t`，核对证书 chain/SAN/私钥权限与续期，再按 D4 授权 reload、DNS/firewall。检查外部 443、内部 18765/5432，不暴露资源路径。只信回环 Nginx，覆盖客户端 forwarded 头。
10. 运行下一节 HTTPS/浏览器/故障/负载，全部证据绑定最终 release/hash。失败退回维护门，不删除新结果，是否开放由证据决定。

## 6. 真实验收入口与分项

`verify_site.py --origin https://<真实域名> --output <新报告>` 默认仅未登录只读检查。它会拒绝 capabilities 中仍宣告 ZIP/M08/M09/M14 的服务器，不用前端隐藏冒充能力关闭。

`--execute-synthetic` 接收 stdin `accounts` 两项，字段 username/password，用户名必须以 `p15_` 开头。验证安全 Cookie、Origin/CSRF、双项目/资产隔离、合成 WAV、下载/Range、1 GB/三天响应及下载不续期，只删除自己刚创建的 WAV；合成项目保留。失败报告记录已创建 ID，保留现场，不泛化清理。

增加 `--lpc --expected-operation lpc_analysis` 执行固定 0.1 秒合成材料的 0.05 秒 LPC、字体快照、提交幂等、三件下载 hash 和历史。结果保留正常三天清理。客户端耗时与 `Server-Timing: app` 分开；后者是响应头准备时间，**不含完整流式下载，也不是纯科学计算时间**。公网 RTT 未测则 null，不用相减冒充 RTT。

其他实际场景见 [验收矩阵](../testing/p15-acceptance-cases.md)。HTTP 检查器不能替代浏览器字体、真实 Linux/PG 时钟与故障验收。精确政策边界、超额与删除失败在获准隔离 DB/服务注入，禁止调整整机时间或改真实用户资产。请求中的两账号只能是指定测试账号。

## 7. 日常启停、排空和故障恢复

这些 Linux 命令仅在目标 D4/D5 已批准后，以专用用户运行：

```sh
systemctl --user start ptb-staging@api.service
systemctl --user start ptb-staging@cleaner.service
systemctl --user start ptb-staging@worker.service
systemctl --user status ptb-staging@api.service ptb-staging@worker.service ptb-staging@cleaner.service
```

计划停机先原子写 control=drain。新 claim/提交关闭，运行任务继续。用只读聚合确认 running/cancel_requested=0、P11 active=null/cgroup populated=0，再逐个 stop worker、cleaner、API。排队任务可以保留，记录 counts 和版本约束。需要取消而不等待时，明确列出本次任务 ID，走现有取消接口。

进程异常：不要手工清空任务表或准入 journal。先确认精确 PID/start identity/unit/cgroup，等本地 10 秒租约失效，由现有 recovery 标 interrupted。当前服务器独立版不开放通用 retry，人工审阅后由 A/B 支持的安全入口重试，不修改 snapshot 来强行运行。B 模式的租约/重排以最终协议为准，当前 proposed 60 秒不可写进本地全局默认。

未知文件、账本不符、清理失败：保留文件与 used/reserved，关闭写入，定位对应 manifest/预留和失败机器码。不要删除文件令预检变绿。手动修复必须另审具体对象。

磁盘候选门：资产写入按现有原子预留及物理余量检查，保留至少 10 GiB；<15 GiB 告警，<10 GiB 或 MemAvailable<700 MiB 连续 10 秒、memory full PSI avg10>1 连续 10 秒、OOM/未知归属触发人工排空/故障处理。**没有宣称已实现独立 20 GiB 全站池或全机资源硬上限**。科学组 1 GiB 不包含 API/PG/代理/缓存，必须另测主机资源。不要降低阈值继续跑。

日志仅固定机器码和无 URI/用户/IP/凭据的耗时统计；Nginx 示例轮转每天/10 MiB、保留七份。先核实现有 nginx logrotate 通配，避免双重轮转。systemd journal 使用目标已有容量限制，未授权不改全局 journald；若现有无限保留，提出只针对本服务的替代收集方案后再开服务。测试原始结果/副本纳入 3 天政策，长期报告只保留聚合和 hash。

## 8. 回滚与恢复能力

| 时点 | 可以做 | 不能做 |
| --- | --- | --- |
| 未提交迁移 | SQL 事务失败 ROLLBACK，核对版本和旧指纹，保持旧服务停写直到一致 | 忽略迁移失败继续启新 worker |
| 006 已提交、未放新写 | 留政策 2 兼容版本、maintenance，优先向前修复。应用前一版本必须确认理解 policy_version=2 及三天 | 直接指向只懂七天的历史二进制 |
| 已有新结果 | 同上，保留所有新结果/期限/预留/manifest；需要 schema 补偿另审增量 SQL | 删除新结果、用旧快照覆盖新写、延长 TTL 假装旧行为 |
| 007 已提交（未来） | 停止新旧所有 claimant，兼容 remote attempt/generation/receipts 后才换版本 | 回滚到会无视 remote_jobs 的旧 worker，双重领取/发布 |
| 数据恢复 | 只能用已核验一致的 DB＋资产恢复点与授权窗口；回读所有 hash/账本，再开放 | 仅恢复 DB 或仅恢复文件、自动建全量备份、未经确认回放过期语料 |

应用切换以审核后的 unit/config 中**固定 release 路径**为准，停止/排空后切换；不替换运行中 interpreter/wheel。旧目录保留，schema 版本不随应用回滚变化。若没有合格回退版本，服务保留维护模式，不能声称具备可用自动回滚。

## 9. 个人 WSL 与实验室节点

首轮已授权个人 WSL 测试，NInfer 原生 Python 可用，缺 systemd/run/control 与可写 cgroup 委托。已完成独立测试后停止在硬限制门，不修改 `.wslconfig`、代理、防火墙、现有模型或整个 WSL 生命周期。

最小后续方案：由 C/P11 先完成 B 的 binding 和受限 adapter；在独立具有 systemd user manager/cgroup 委托的已授权 Linux 环境验证，或另审对 NInfer 的精确系统改动及仅该发行版的维护窗口。本任务没有执行此变更。Windows 原生只在有额外验证价值时单列，当前未建第二套客户端。

实验室机器将使用独立注册、自己的凭据、资源/版本/字体证据；不复制个人节点身份。仅出站 HTTPS，服务器不反向连实验室、不暴露 DB。断网注入用本任务客户端的 transport hook/专用 TLS relay，不能改主机网络；断进程只用所持 Popen/精确 PID。四轴报告分别为个人 WSL、个人 Windows、云端回退、实验室 Linux，最后一项当前待验。
