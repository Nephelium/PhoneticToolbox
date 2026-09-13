# M03-E3 批次错误反馈收口

2026-09-13，井井授权继续开发态 EGG；EXE 相关工作暂停。本轮 verified（限定 Windows 独立 Chrome，见[验收报告](../testing/m03-batch-feedback-report.md)），限定 A21–A24 批次提交的错误恢复，不改变科学算法、参数默认值或任务协议。

修改 frontend 公共 ModalDialog.vue 和 EGG state.ts、EggBatchPanel.vue、EggAnalysisPage.vue，增加独立 Chrome 定向用例。按既有 EggTaskConfig 契约检查参数，逐文件的采样率、声道与计算检查仍由后端负责。错误参数不得发起字体预检或任务；全部提交失败时保留弹窗、选择及上一批保存入口，显示失败原因；部分成功保留成功任务并清楚列出未提交项，不自动重交已接收任务。提交期间冻结参数，避免显示与快照不一致。

先运行浏览器回归复现错误，再最小实现，执行 npm --prefix frontend run test / typecheck / build、node tests/e2e/m03-batch-feedback.cjs 和文档/架构检查。正常批次走实际服务和兼容科学子进程；受控提交拒绝与延迟仅用于故障边界，不冒充自然网络故障。小窗口验证弹窗错误可见、正文滚动和底部操作可达。
