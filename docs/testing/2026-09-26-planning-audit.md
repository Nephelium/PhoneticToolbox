# 统筹前只读代码审查与环境记录

2026-09-26。任务 P11-COORD。范围：共享任务/进程边界、服务器入口、配额生命周期、当前模块页首与明显性能热点。**静态审查完成，未进行全库审计、科学重跑或 Linux 功能验证；下述问题未在本轮修复。**

基线：`codex/v3-rebuild`，HEAD `9d0283c`，另含开始时的 M12 等未提交变更。新需求与任务见[统筹计划](../plans/2026-09-26-server-coordination.md)。文件行号是本轮读取时的位置。

## 1. 发现与修复归属

| ID / 优先级 | 事实、影响与边界 | 后续任务与验收 |
| --- | --- | --- |
| CR01 / P1 Linux 上线阻断 | `acoustic_executor.py:44,65,86,105` 直接使用 `.native.windows.InputPipe`；`native/windows.py:10` 非 Windows 即拒绝。`spectrogram_preview.py:33`、`parameter_preview.py:18`、`font_preflight.py:18` 也明确拒绝 Linux。这是实际平台缺口，不是已确认的 Windows 功能故障 | P11-LINUX：统一进程接口+Linux 适配；Windows 回归、Linux 任务/预览/字体及取消进程树验证 |
| CR02 / P1 Linux 科学运行时阻断 | `egg_runtime.py:10–35` 要求 python.exe、固定 Windows Conda SciPy/MKL build；LPC 复用它。直接复制 Windows 环境或仅修改可执行名都不满足科学等价 | P11-LINUX + M03/M04：Linux 独立锁、fingerprint、同输入数值和边界对照，不移除校验骗过能力检查 |
| CR03 / P1 小服务器资源风险 | `store.py:58` 默认 max_running=2；`acoustic_executor.py:55,77` 分别给 LPC 2,000,000,000 与 EGG 3,000,000,000 字节进程预算；预览另有 1 GB 上限且 `spectrogram_preview.py:17` 信号量仅进程内。2 worker 与预览同时运行时，现有上限组合可能超过 4 GiB。未实测 OOM，不能称已崩溃 | P11-PERF：服务总预算、单计算槽与 capability 准入；实测父子进程/预览共同峰值、10 人混合负载 |
| CR04 / P1 公网入口缺口 | `ptb_api/server.py:30,48–49` 是 loopback 预览入口：固定 127.0.0.1 Origin、LoopbackGuard、proxy_headers=False。转发到域名后写请求 Origin/Host 可能被拒绝。`AuthSettings` 支持 HTTPS，但该入口没有配置成正式域名服务 | P15-STAGING：独立部署配置、同源 HTTPS、可信代理头/客户端 IP、Cookie/CSRF 与域名全流程；不移除检查作为修复 |
| CR05 / P1 新政策尚未生效 | `quota.py:4–5` 仍为 5 GB/7 天；`acoustic_models.py:21` 独立 604800；`003_storage.sql:12` 固定 CHECK 5 GB；前端 `ProjectStorage.vue:124` 和生成协议还有 5 GB 上限。只改一处会使 DB/校验/UI 不一致，旧七天 manifest 直接按三天验证还可能无法回读 | P07-POLICY：单一政策源、增量迁移、旧数据兼容、配额/期限边界与生成契约。先审阅旧数据策略，不自动清理旧文件 |
| CR06 / P2 能力报告不足 | `ptb_api/main.py:88–94` 主要看存储/批次对象，algorithms 仅列 M01，task_operations 未覆盖实际 M03/M04/M09 路由；未验证科学 runtime 或节点能力。新 Linux/外派环境中不能用此判断实际可执行模块 | P11-LINUX/P06-REMOTE：实际运行能力、版本与预算/离线原因的明确 schema，前后端消费同一来源 |
| CR07 / P2 明确优化候选 | `lpc/_legacy.py:20–26` 对全部 N 样本做 full 自相关，后续求解只取前 order+1 个滞后。直接相关的工作量随 N 二次增长，是小服务器不利路径；现有 ADR-047 已记录 48,000 样本约 8 秒的历史本机探测，该耗时非本轮复测 | M04-PERF：先 profile，比较有限滞后/FFT 候选及数值稳定性；未经独立科研验收不更换默认算法 |
| CR08 / P2 复制与内存热点 | `acoustic_executor.py:31–41,50,71,118–130` 将完整输入读入 bytearray、复制为 bytes、拼接请求，结果 bundle 与 payloads 又常驻；`egg_preview.py:37–46` 计算完整 PSD 后复制可见频带再缩图。这些是实在的全量分配，实际峰值/收益待测。`egg/export_series.py:48–63` 已分块 FFT，不能误报为从未优化 | P11-PERF/M03-PERF：流式校验/少复制/显示专用缓存，与科学导出严格区分；峰值报告覆盖父进程和子进程 |
| CR09 / P2 UI 规范分散 | M01 `ParameterEstimationPage.vue:131`、M02 `ParameterDisplayPage.vue:40`、M03 `EggAnalysisPage.vue:181`、M04 `LpcSpectrumPage.vue:71`、M09 `Spec2WavPage.vue:41`、M12 `AnnotationPage.vue:270` 都维护页首；字号/间距和操作位置不同。M04/M09/M12 等有重复关闭按钮，AppShell 已有标签关闭入口 | P04-UNIFY：公共框架、控件迁移表和实际视觉/关闭状态回归，不能仅全局隐藏标题 |
| CR10 / P2 历史交付链接失效 | 修改前 `python scripts/validate_docs.py` 报两条失效链接：README 与 `docs/plans/2026-09-11-m10-onset.md` 指向当前不存在的 M10-R5 EXE。只证明当前路径缺失，不能推断旧交付从未存在 | 后续文档/发行记录修订：核实产物去向或明确历史路径不可用，不为消除告警伪造 EXE 或重建旧包 |

CR01–06 面向新 Linux/服务器目标，均须在相应入口开放前解决。P1 表示该目标下的阻断/风险优先级，不代表已发现生产事故。CR07–08 为优化线索，收益未测。

## 2. 已存在且应复用的正确机制

- `ptb_worker/policy.py` 的 fenced 同时检查状态、worker、generation、lease 与 deadline。
- `JobStore._claim` 保持每 owner 一项运行任务、SKIP LOCKED 与事务；Postgres 写事务还有 advisory lock。这不是没有并发保护，也不能擅自删锁优化。
- 文件任务已有逐块验证、输出预留、seal/complete 与失败回收。远程节点须通过 API 复用这些约束。
- `assets.py:109–151` 的参数表/语谱预览有 owner、session、hash 与处理后到期复查；目前主要缺 Linux 受限执行与统一资源准入，不报告无依据的跨用户泄露。
- M12 等标签关闭有现成未保存保护。公共页首精简要走同一关闭路径，不能重写另一套绕过保护。

## 3. 本轮实际检查与限制

- 读取当前 Git 分支/status/diff、相关源码与层级规则；记录 590 个 frontend/backend/desktop/packages/contracts/scripts/third_party 文件的内容 hash 用于本轮前后比对。排除用户语料和运行环境，清单仅放忽略的 `output/validation/p11-coordination-2026-09-26/`。
- 执行 `wsl --list --verbose` 只读枚举，没有启动/安装发行版或改全局配置。
- 使用本机 OpenSSH，对用户提供的主机执行一次 BatchMode 连接探测，仅准备运行 `uname -m`。连接到认证阶段后被 `publickey` 拒绝；未执行远程命令。SSH 主机指纹首次登记只写入本任务忽略目录，不改全局 known_hosts；尚未通过独立渠道核对指纹。
- 后续用户终端截图补充：系统可见内存 total 约 3.4 GiB、available 约 2.9 GiB、Swap 0，CPU 2 个逻辑处理器，ext4 根分区显示 49G/可用 44G。据此将资源方案的科学进程组初始总预算收紧到约 1 GiB，最终值仍待实测；不依据截图中的虚拟拓扑推断独占物理核心。
- 工具可控浏览器清单只有无标签的内置浏览器，未接管用户截图中的网页终端；未安装 Workbench CLI、未读取私钥内容、未更改 authorized_keys。
- 文档校验基线为 666 文件、333 来源、32 任务，2 条当前失效链接；历史归档中的旧链接由工具另列且不算新增问题。后续本轮结果见同目录 validation 文件。
- 未做功能测试、性能测试、真实云端登录/部署/DDL、Qt 或 EXE 验证，没有修复以上业务代码。服务器规格来自用户截图，SSH 仅验证可达认证端点。

## 4. 交付判定

本轮“完成”只指统筹计划、当前规则同步和只读问题清单可供 agent 使用。运行时仍为旧配额/保留期与原模块行为；新政策实际生效须 P07-POLICY 完成，Linux 与远程计算须分别验收。

本轮最终验证：`python scripts/validate_docs.py` 检查 669 文件、333 来源、41 任务，仍报原有两条 M10-R5 失效链接，新增错误 0；`git diff --check -- <本轮文档清单>` 通过。590 文件 hash 清单中只允许 `contracts/ARCHITECTURE.md` 的本轮规范更新，其余内容 hash 不变。定向一致性检查确认 8 个实施任务均 planned、15 模块均追加平台/资源/UI 状态轴、新政策目标值一致。证据在忽略目录 `output/validation/p11-coordination-2026-09-26/final-validation.json`，未运行功能/性能测试。
