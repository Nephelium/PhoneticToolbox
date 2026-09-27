# P15 真实站点与故障验收矩阵

2026-09-27。所有真实站点/节点场景当前 **not_run**。本文件定义执行步骤与证据，不是已通过记录。独立 Python/ASGI doubles 的结果见[P15报告](p15-staging-report.md)。

## 前置与固定数据

记录 deployment release manifest、runtime/receipt、操作系统/UID/namespace、PG/存储 instance、节点独立身份指纹、协议版本、UTC 与单调时钟、所有本任务 PID/unit。公开合成输入由 `verify_site.synthetic_wav()` 生成0.1秒/16kHz/单声道/PCM16，3244字节；科学扩展可使用 P11 已登记的0.8秒 EGG 合成 fixture。仅测试账号/project，禁止真实语料。

每个场景报告至少包含 `case_id/status/environment/release_sha256/runtime_sha256/protocol/evidence/limitations`；失败也保存。环境枚举为 `asgi_double`、`wsl_component`、`local_tls_fixture`、`real_cloud`、`personal_wsl_node`、`personal_windows_node`、`laboratory_linux_node`。前四项不能冒充后面的实机节点结果。

## 站点与政策

| ID | 步骤/工具 | 通过条件/证据 |
| --- | --- | --- |
| S01 | `verify_site.py` 默认只读＋TLS工具，浏览器访问同源 /server/ | 证书链/SAN/到期正确，HTTP重定向，health server，静态与API同源，记录真实域名 |
| S02 | `--execute-synthetic`，两个新建 p15_ 账号登录/退出 | Host cookie为Secure/HttpOnly/Strict/Path=/且无Domain；退出旧cookie失效；CSRF/错误Origin/跨站禁止；响应no-store |
| S03 | 外网发送伪造 Forwarded、X-Forwarded-*、Host，同时查看脱敏代理/限流观测 | Nginx覆盖来源头，仅回环受信；用户来源不能被头伪造。P15单元用实际ProxyHeadersMiddleware验证信任边界，但目标Nginx仍须实测 |
| S04 | 双账号＋项目＋upload/block/finalize/content/Range，切换账号重读 | 另一账号项目/资产/任务/结果拒绝，Range 206正确、越界416；字节/hash一致；失败不得留公开文件 |
| S05 | 真实浏览器浅/深、宽/窄，中文与组合IPA、图表/PNG/历史 | Doulos SIL实际加载与文件hash正确、无fallback假通过，控制台无未处理错误，保存截图与下载件 |
| P01 | 只读 probe_db，核对A前置交付/SQL hash/实例及新旧期限指纹 | 目标实际policy=2，每账户十进制1,000,000,000，旧manifest/expiry/used/reserved保留，不能仅SQL文件存在 |
| P02 | 获准隔离PG中预留至最后1字节，竞争4请求，随后回读 | 恰好1成功，其余quota_exceeded，不实际写1GB填盘；这一方式明确标账本边界，HTTP大上传另测 |
| P03 | 在测试服务的受控DB时钟/独占资产到期注入上验证截止前/等于截止/后 | 精确259200秒，截止时禁止读取/发布。下载/Range不续期；不能调整整机时钟或声称自然三天 |
| P04 | 测试旧版本manifest/旧大结果；超额账号读取/下载/删除后再尝试增长 | 旧数据可回读、available=0，新增长禁止，未自动删旧资产，复制保留更早期限，新科学输出成功后三天 |
| P05 | 仅自有资产的 unlink/存储写失败测试hook | 实物未删时账本不释放，重试真实删除后才释放；保留预留/残留定位，不注入到其他账户/生产进程 |
| P06 | 迁移前新建在途输出→006→完成、晚代数worker回传 | 结果显式policy 2，旧snapshot不改，fencing先于发布，manifest整套原子可见 |

P02–P06 优先复用 A 的 `test_p07_policy_postgres.py`、`test_p07_publication_postgres.py` 和实际授权目标脚本；它们的随机合成库证据与真实目标证据分开。当前 Linux PG完整行为仍缺证据。

## 服务器单独执行与负载

| ID | 步骤/工具 | 通过条件/证据 |
| --- | --- | --- |
| C01 | `verify_site.py --execute-synthetic --lpc --expected-operation lpc_analysis` | Linux真实字体预检、提交幂等、三件下载hash、owner和历史，新包receipt匹配；只算M04短输入范围 |
| C02 | 运行时资格后对M01/M03各执行既有P11同参数fixture，接目标账号/PG上传入口 | 数值/角色/完整hash保留，不能用SQLite local token替代账号链路；记录每模块排队/计算/导出/传输 |
| C03 | 用两个本任务API/worker或1API+worker同时请求科学/预览/导出 | 同UID/namespace实际组区间不重叠；科学组memory/CPU/pids/swap约束先安装，子进程清理后才释放槽 |
| C04 | 取消queued与确认MainPID启动后的running任务；本任务子进程SIGKILL/30秒停滞hook | 不发布部分结果、预留真实回收、下一任务恢复。observed running状态不等于子进程启动证据 |
| C05 | control=drain→在途完成→精确stop/restart三个服务 | queued保留、lease/fencing恢复、历史可读；私有根无未知残留；重启权限只限列明服务 |
| C06 | 最终整合包短门→至少1800秒混合负载 | 绑定新hash；10合成客户端、PG/账号、工作台/预览/下载与科学任务并行，全部失败/busy记录；同时保存API PSS、PG/主机/cgroup/磁盘/清理证据 |

负载计时至少拆出公网客户端请求总时长、TLS/连接观测、响应准备Server-Timing、科学函数、导出、上传/下载字节与耗时、队列等待。没有包级测量时 RTT=null。分位数采用固定nearest-rank并附n；n少时p95/p99落在最大值不能推断尾分布。

停止条件沿P11：MemAvailable<700MiB或memory full PSI avg10>1持续10秒、OOM、磁盘<10GiB、API/worker异常或归属未知。必须由只读监控触发停止本次负载发送者，必要时按D5授权停止本任务任务，不能停别人的服务。轻API服务端p95≤500ms为限定目标，公共预览另报排队/busy；P11已见后段PSS增长，小时/天级稳定性另测。

`scripts/benchmark_modules.py` 现有入口仅证明其合成SQLite/local-token模型。先对新包复跑此门，再在真实PG站点用上述账号客户端场景组成相同测量范围；生产PG负载驱动的最终接线须在B稳定后冻结，当前不冒充已经具备该证据。

## 节点故障回退（B/C可运行交接后）

当前生产RemoteFiles bridge、P11 trusted-worker及C binding未完全交付，不向草案URL发送伪造成功请求。不自行实现第二个协议或在P15补调度。最终参数以B生成契约为准。

| ID | 有界注入 | 不变量与需要保存的证据 |
| --- | --- | --- |
| R01 | 个人WSL节点以独立身份登记，上线→领任务→下载→计算→上传→complete | 实际HTTPS、runtime/字体匹配、输出与服务器科学基准一致、单次发布/结算；无DB/共享盘连接 |
| R02 | 专用client尚未claim时暂停其领任务或关闭自有transport | 未领取任务按能力进入服务器或明确waiting；不抢占其他节点、不消失 |
| R03 | 捕获已领取且运行的attempt，断开本客户端TLS relay连接并停止其heartbeat | 旧租约失效前服务器不抢；失效后新generation，服务器有能力且预算内才重试；无能力/超预算保持队列 |
| R04 | 恢复同一节点连接，服务器任务已开始 | 服务器任务继续原处运行，后续新任务按B健康滞回重新优先节点；保留attempt时间线 |
| R05 | 旧attempt保留的下载/上传/complete依次重放 | 全部拒绝旧generation，不能复活结果；完整manifest数量=1、额度结算=1 |
| R06 | 专用relay交替延迟/断开少量连接 | 租约相对服务器时间、重试/退避有界，健康滞回不因一次成功立即反复切换；无重复发布 |
| R07 | 只暂停结果上传，heartbeat照常 | 实际新增字节不推进即传输停滞，B超时回收；不能用心跳伪装传输进度 |
| R08 | 服务器complete已提交后，relay丢失该响应 | 客户端重试相同attempt/generation返回同一manifest，不能再次预留/结算/延长TTL |
| R09 | 撤销本任务测试节点凭据 | 下一heartbeat/read/write/complete拒绝；客户端停止自己的子进程/传输，旧发布失效；不轮换其他节点凭据 |
| R10 | 两个本任务独立节点同时poll同一任务，另与服务器claim竞争 | 原子唯一有效attempt/代数；允许有记录的重复计算，最多一套可见结果和一次逻辑扣额 |
| R11 | 客户端自有进程退出/重启及休眠时钟组件测试 | 不复活过期lease，缓存按输入/任务更早期限清理；真实个人电脑睡眠/重启不自动执行，模拟时钟单列 |
| R12 | 实验室Linux独立注册后重复R01–R11适用项 | 自己的身份/资源/发行版/网络/磁盘/版本，不能复制个人私钥。四轴独立，未到实验室即待验 |

故障relay只能转发指定staging origin与本任务节点连接，由持有它的父进程停止。当前只有步骤契约，没有已启动故障代理，不存在关闭实验室/个人电脑整机网络、改系统代理/防火墙、wsl --shutdown或pkill的动作。

## 当前执行分工

- P15可立即执行：模板/宿主/HTTPS驱动单元、源码/证据哈希、WSL只读检查与独立源码快照测试。
- A：政策/生成协议/目标库迁移与PG事务复现修复。
- B：唯一节点协议、RemoteFiles生产桥、调度/排队/重试和generation。
- C/P11：客户端binding、相对租约时钟、受限执行adapter与能力receipt。
- P15在上述交接后：实际站点与客户端集成；准确缺陷回报原负责人串行修复后复验，最后才更新已释放公共文档。
