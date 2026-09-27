# M15 统筹同步摘要

2026-09-27。本轮已交付纯客户端正式模块，Windows 开发功能验收完成。完整跨平台条目保持 `in_progress`，不扩大 verified。

| 维度 | 当前状态 |
| --- | --- |
| 功能 | F01–F07 已接统一工作台，四范式、素材/序列/参数/问卷、资源助手、XLSX/项目往返、运行/恢复/导出 |
| Windows 浏览器 | verified，实际 Chrome：四范式断网 A、结果回读、刷新/双标签、安全边界、受控故障与主题缩放 |
| Qt | verified，实际 Workbench/ptbapp：X 试次、Web Audio/IDB/Web Locks、原生手势、XLSX/JSON 下载回读；非 EXE/全设备验证 |
| Linux | WSL 原生解释器完成当前静态资源检查；Linux 浏览器/音频未验。无 M15 服务端科学进程 |
| 小服务器 / 远程计算 | 不适用。只分发公开静态资源，无账号上传、配额、SQL、Python core/API 或 worker |
| 统一 UI | 已完成按需注册、公共组件/字体/主题、专注按键隔离、异步保存与未导出关闭保护 |
| 离线 | A、Qt C 定向通过；B 用户批准留具体方案，本轮没有 Service Worker |
| 科研边界 | 音频保留作答窗口开放后计时；文本/图片两次 rAF 修正已批准。无物理声学/键盘端到端时延测量 |

正式入口：统一工作台 → 标注与实验 → 感知实验。项目/会话/Blob 全在用户浏览器，结果须显式导出。JSON 保存完整角色/hash/最终序列/配置/问卷/attempt/时序溯源，CSV/XLSX 保留旧前置列。

本任务所有权：`frontend/src/modules/perception/`、`frontend/tests/m15*`、`tests/e2e/m15*`、`scripts/verify_m15_*` 与 M15 专属文档。共享最小改动为 AppShell 按需入口/关闭/专注接线、Qt 当前页面 blob 下载后缀白名单、source-registry 与生成数据中的 M15 依赖、台账/迁移表本模块行。其他任务已有差异全部保留，不提交混合工作树。

验收命令与实际证据见 [M15 报告](m15-report.md)。21 项 M15 逻辑测试和 172 项全前端测试通过，类型检查/构建通过。SheetJS 0.20.3 固定本地 Apache-2.0 原件，M15 JS 分包约 550 KB，仅打开模块时加载。

后续需独立安排：按 [ADR-M15-001](../decisions/ADR-M15-client.md) 串行实施 `/m15-offline/` 同一 AppShell 的限定静态缓存，验收浏览器断网冷启动；Linux/其他浏览器与真实设备/休眠/回环时序按实际证据补齐。无 P07 数据库迁移前置依赖。未 push、部署、修改 V2/语料/旧结果、安装全局依赖或自动打包 EXE。

[源映射](../modules/evidence/M15-source-map.md) · [实施计划](../plans/modules/M15-perception.md) · [用户操作与恢复](../manual/perception.md)
