# M03 开发态功能收口

2026-09-13。井井审阅下一步后授权继续。状态 verified（限定[收口报告](../testing/m03-dev-closeout-report.md)列明范围）；限定已有 Windows 开发环境和独立 Chrome，不启动 M04、引用核查、EXE 或部署工作。

## 本轮文件与行为

- `frontend/src/modules/egg-analysis/EggAnalysisPage.vue`：LP 阶数纳入本机草稿与未保存提示，继续与预览参数分离，避免仅修改 IF 阶数导致四图失效。兼容已有草稿。
- `frontend/src/app/AppShell.vue`：EGG 保存失败在当前关闭确认框内反馈，保留页面与草稿，重新打开确认框时清除旧反馈。
- `tests/e2e/m03-draft-closeout.cjs`：先复现只改 LP 阶数便丢失的行为，覆盖取消/Escape/放弃/保存/存储失败恢复、重开及真实分析/IF/导出快照。
- 使用说明、阶段记录及功能矩阵：交付开发入口、实际验证范围和剩余独立验收项。

## 验收与退出条件

`node tests/e2e/m03-draft-closeout.cjs`、既有试听与结果反馈回归通过。`npm --prefix frontend run test`、`run typecheck`、`run build`及文档/架构/生成数据检查通过。正常科学任务与文件读取使用真实服务；存储异常仅定向注入测试浏览器，不改用户存储。

确认六功能组已有映射和证据，本轮新增场景完成后将 EGG 开发态功能阶段标为 verified。完整 M03 的设备/生产/跨平台等未测项仍单列，不为这些暂停或范围外项目无期限延长当前功能收口。
