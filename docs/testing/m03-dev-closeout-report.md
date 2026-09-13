# M03 EGG 开发态功能收口

2026-09-13。**开发态功能阶段 verified，限定已有 Windows 环境与各报告列明的 Chrome/Qt/本机托管服务范围。** V2 六功能组已有映射，本轮完成剩余草稿及连续操作检查。完整 M03 的设备、生产和跨平台验收仍 in_progress；引用后续核查、EXE及相关探针按井井要求暂停。

## 本轮修复

1. 仅修改 LP 阶数时，原页面不会标记未保存，保存草稿也未写入该值，关闭重开会回到自动。现在使用同一草稿键原子保存 `inverse_lp_order`，读取时与科学参数拆开。仅改变该值不使已有预览失效，自动空值同样可保存，旧草稿缺少该字段时继续自动。只在 IF 提交时把它转换为任务的 `lp_order`，不向 API 发送草稿专用字段。
2. EGG 保存失败原本只写入被关闭确认框遮住的页面。现在确认框内显示失败原因，标签和编辑保留，可取消或恢复存储后重试。再次打开关闭确认框会清除旧反馈。

生产代码仅修改 `EggAnalysisPage.vue` 与 `AppShell.vue`。科学核心、数值默认、波形布局、任务协议及输出文件规则未改。继续使用 V3 公共弹窗和主题。

## 本轮验证

| 命令 | 实际结果与证据目录 |
| --- | --- |
| `node tests/e2e/m03-draft-closeout.cjs` | 9组通过，`chrome-258f9c7d15334735b98726da7e2bb617` |
| `node tests/e2e/m03-playback.cjs` | 4组通过，`chrome-98338c7d25c4458980efec2b00f9601a` |
| `node tests/e2e/m03-result-feedback.cjs` | 10组通过，`chrome-066c001bac164f5abd051ebd453c25a8` |
| `npm --prefix frontend run test` | 65 passed |
| `npm --prefix frontend run typecheck` / `run build` | 通过，当前前端产物已更新 |

`.venv/m09-ui/Scripts/python.exe scripts/validate_docs.py`：579文件、330来源、32任务，errors=[]，历史快照失效链接单列。`scripts/check_architecture.py` errors=[]；`npm --prefix frontend run ui-data:check`、`run contracts:check`无漂移；`git -c core.safecrlf=false diff --check`通过。未新增第三方来源或依赖。

Chrome证据均位于 `output/validation/m03-ui/`，23组检查均0页面错误。新增9组覆盖 LP 单独修改提示、取消/Escape与焦点恢复、保存/放弃后重开、受控存储失败与恢复、小窗按钮可达、自动值及显式保存、真实预览/IF/CSV三路径参数快照。既有回归覆盖归一化实际音频样本、交换声道、两个IF播放器切换、关窗停止、结果重读及迟到反馈，以及实际CSV下载哈希回读。

失败基线 `chrome-abfc2f48ed1945389e61fcd9ae70a240`：仅改LP后直接关闭，未出现应有的确认框。首轮修复 `chrome-e1e65e802a114ddba362fca3c6e907dd` 已通过9组；暗色截图落在按钮颜色过渡期间，最终截图禁用过渡后复跑并查看，未因此修改主题实现。

存储异常仅为测试浏览器对 EGG 草稿键注入 `QuotaExceededError`。结果回归中的到期/延迟是受控故障，正常任务、科学子进程及文件均真实。沿用既有测试schema隔离副本，未执行DDL；不把注入异常说成实际磁盘故障、自然到期或生产认证验收。

## 已覆盖的功能与限定证据

- [六功能组对照](m03-function-review.md)：文件/声道、四图与滤波、事件/CQ/SQ、双F0/逆滤波、选区/导出、完整批次及取消。
- [原核心基准](m03-core-report.md)、[任务与导出](m03-jobs-report.md)、[四图页面](m03-ui-report.md)与[120秒长文件](m03-long-report.md)：科学数值、双WAV、CSV/PNG及长文件范围按原报告限定。本轮未重新运行完整科学基准。
- [本机托管网页与自然录音](m03-e2-report.md)：账号隔离、配额/清理、指定自然ROI的已有证据保持原范围，本轮未复跑托管账号流程。
- [字体预检](m03-font-preflight-report.md)、[批次反馈](m03-batch-feedback-report.md)、[切文件反馈](m03-preview-switch-report.md)：沿用已验证的针对性修复。

## 开发入口与后续边界

当前入口：[Start-Research-Workbench.ps1](../../scripts/Start-Research-Workbench.ps1)。在项目根目录 PowerShell 运行 `& .\scripts\Start-Research-Workbench.ps1`，进入 EGG 信号分析。启动器使用已存在的项目环境/本地任务库并读取 `frontend/dist`，不会执行构建、更新或迁移。本轮确认入口指向已更新的构建，未额外启动Qt窗口。

使用说明见 [EGG分析](../manual/egg-analysis.md)。草稿保存参数，不保存音频；重开后重新选文件，已提交任务可从处理记录恢复。分析文件限120秒/576万帧，IF另限1秒/48000帧。

A20物理声卡/听觉多设备、A28原生多屏DPI、A30生产及跨平台仍未完成。这些独立范围不阻止当前开发态功能阶段收口，完整模块状态不扩大。A29引用后续核查与EXE工作暂停。下一模块建议M04 LPC谱图，本轮没有实施，后续按具体授权核对V2说明书/源码与V3组件。
