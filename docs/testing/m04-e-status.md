# M04-E 统筹状态摘要

2026-09-27。详细证据与 A01–A20 逐项表见[专项报告](m04-e-report.md)。

| 范围 | 状态 |
| --- | --- |
| M04-A/B/C/D | 保持各原报告限定的 Windows `verified` |
| M04-E Windows 本机页面 | `verified`，Chrome 27 组；波形 Shift 手动选区、可选 Praat 语谱图直接拖选同一 ROI，草稿/关闭、任务/历史、三文件实测 |
| M04-E 正式 Qt 宿主/自然语料 | `verified`，仅两份 P03 已授权本地样例的短 ROI、TextGrid、1024 点结果与 PNG/JSON/WAV 保存 |
| Linux 服务 | 继承 P11 真实 Ubuntu、受限合成任务及 HTTP 产物下载证据；自然语料、PG 账号网页和全机资源压力未验 |
| 托管账号 Chrome / A01、A18、A20 网页部分 | `in_progress`，专用旧政策库被 `storage_policy_migration_required` 拦在上传前；未执行 DDL |
| 整体 M04 / 跨平台 / 生产 | `in_progress`，不得并成整模块 `verified` |

下一次 M04 网页验收只在 P07 归属任务完成专用库新政策版本迁移并独立验收后运行 `scripts/verify_m04_web.py`，检查双账号、上传/刷新、LPC 任务与历史、三文件下载、owner/额度/期限。M04 本轮不改共享政策、执行器、契约、总台账、根 README/AGENTS，也未 push、部署或打包 EXE。
