# M14 统筹状态摘要

2026-09-27。建议总台账：**M14 in_progress**。

- `windows_functional=verified`：原五功能组全部迁移；正式统一 AppShell/desktop adapter/API/持久 worker/受限 child/文件交付，Chrome 与实际 Qt 通过。
- `core_export_windows=verified`、`core_export_wsl=verified`、`core_export_linux_server=verified`：原 V2 独立基准，38/37/37 项分别计数（Windows 多一项原生保存回滚）；实际三文件回读与 Word/Excel 视觉检查。
- `linux_durable_task=in_progress`：P11 尚未登记 M14 fixed entry/能力凭据，保持关闭；已交固定协议及受限 child 资格证据。
- `windows_browser_to_linux=verified_limited`：统一页面/health/capability/明确不可用响应；没有成功 M14 Linux task。
- `linux_native_browser=blocked`：项目内 Chromium 已获授权并下载，缺 9 个系统库，本轮不装系统依赖。
- `server_storage_joint=in_progress`：复用正式存储 writer/owner/project/政策，现存 PG 政策迁移和账号/配额/到期联合证据未完成，无 DDL。
- `ui_unified=verified_limited`：公共 Frame/Toolbar/Section/Status/ModalDialog、主题/字体、关闭保护，Windows 指定尺寸/合成输入。
- `resource_budget=verified_limited`：Windows 512 MB/60 s 与服务器同额 cgroup 短测；服务器单 child 实测峰值约 69.5 MB，非整机/长时吞吐结论。
- 修正单列：XLSX 调类顺序、空韵归并、完整/重名保存、长格行高、富文本空白 Office 兼容、worker 心跳前重库导入。原字音数据规则未重设计。

正式入口 `scripts/Start-M14-Workbench.ps1`，操作说明 `docs/manual/phonology-induction.md`，功能映射 `docs/modules/evidence/M14-source-map.md`，完整证据和命令 `docs/testing/m14-report.md`。全局台账/根文档由统筹更新，本任务未并行覆盖。未 push、生产部署、EXE、V2/语料修改。
