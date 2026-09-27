# P15 首期能力候选清单

2026-09-27。**当前实际开放：无，未部署。** 配置模板科学 allowlist=[]、remote_enabled=false。以下候选必须经过最终整合包/目标 PG/HTTPS 验收后才可进入允许表，不要求首期开放全部模块。

| 模块/入口 | 已有证据 | 目标站点状态/依赖 |
| --- | --- | --- |
| 账号、项目、上传/下载/Range/历史 | Windows 历史 PG；A 新政策独立合成 PG | planned，目标 policy 2、Linux PG/文件锁、Cookie/CSRF/代理验收 |
| M01 acoustic_analysis | P11 固定包真实 Linux/REAPER、短合成及30分钟负载 | 候选；新 runtime/receipt、配额账户发布、字体/原生 hash 必须全通过 |
| M02 参数显示 | Windows 原图窗/PNG；P11 公共受限预览 | 候选交互，独立 scientific op 不适用；真实网页账号 XLSX/波形/字体/下载待验 |
| M03 EGG | P11 Linux 短合成、CSV/PNG、practical/1 数值门 | 候选；新包/字体/账号/期限待验，旧 Windows MKL 不复制到 Linux |
| M04 LPC | P11 Linux 小 ROI及故障/三结果；Windows Chrome/Qt | 首轮最小科学候选；当前托管账号门待 P07 迁移。脚本固定 ROI 0.05s，不能外推48,000样本上限输入负载 |
| M05 | 无 Linux 正式交付 | 关闭；摄像头/视觉模型/设备不同于后台批任务 |
| M06 | 并行迁移中 | 关闭；等待模块报告与 B/P11 明确登记，不能从文件出现认定通过 |
| M07 | 无正式 Linux 交付 | 关闭；模型/算法/来源与资源门未过 |
| M08 | Windows 正式链路；Linux精确门25通过/5失败 | 关闭，禁止放宽断言或靠0.00000001量级差异宣称等价 |
| M09 | Windows 限定链路 | 关闭，Linux capability 未开放 |
| M10 | Windows R5最新定向报告 | 服务器关闭；VTL ABI、设备和录制专门验收 |
| M11 | MFA/模型未验 | 关闭；不自动下载全量模型占系统盘 |
| M12 | Windows R6及历史账号文件链路 | 候选交互；Linux站点长音频/保存/字体/账号待验，无已验远程科学入口 |
| M13 | Windows Chrome/Qt、WSL静态资源hash | 静态候选；无需科学worker，真实站点浏览器中文/IPA和PNG仍待验 |
| M14 | 核心/Windows正式任务，Linux独立child峰值约69.5MB | 关闭，Linux fixed entry/collector/capability/receipt和账号链路未关闭 |
| ZIP/解压 | server-small入站拒绝门 | 关闭，通用导出未迁入受限执行；当前 capabilities 宣告差异见缺陷单 |
| textgrid_segment/通用retry | Linux分段未单独验；旧retry缺少部署白名单过滤 | 本P15宿主关闭，保留历史文件读取 |
| trusted-worker | B协议/C组件并行候选 | 关闭，P11 profile仍拒绝，未接真实节点；个人电脑测试不能代替实验室 |

公共预算：同 UID/namespace 一槽、科学进程组≤1,073,741,824 bytes、1 CPU、64 tasks、swap=0；低于此值的模块预算保留。P11 测得峰值约215.7 MiB仅为0.8秒合成输入组合，当前整合包、大输入与长时常驻未测。

依据：[A交接](../testing/2026-09-27-prerequisites-handoff.md)、[P11最终报告](../testing/p11-perf-report.md)、[M08](../testing/m08-wiring-report.md)、[M14](../testing/m14-report.md)、[M04-E](../testing/m04-e-report.md)、[M13](../testing/m13-report.md)、[M10-R5](../testing/m10-r5-report.md)、[M12-R6](../testing/m12-r6-report.md)。完整来源/许可仍按 third_party 的待决状态，不由本候选表授予发布权。
