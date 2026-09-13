# M03-E3 文件切换反馈

2026-09-13，verified（限定 Windows 独立 Chrome 的下列场景）。井井本轮明确代码引用暂不处理、优先功能，A29 进一步核查暂停，已有说明保留。完整 M03 仍 in_progress，EXE及相关探针继续暂停。

## 问题与修复

旧文件的预览任务完成并读取结果时，用户可以切换到另一份文件。正常结果已有选择版本校验，但读取失败进入 `poll()` 的公共异常分支后，仍会把旧错误写到新文件上。

`frontend/src/modules/egg-analysis/EggAnalysisPage.vue` 现在将轮询异常绑定到开始时的文件选择版本。切换或恢复其他预览使版本失效后，迟到异常不再写入当前页面；当前文件的真实异常仍显示。任务记录继续更新，科学计算和持久任务未变。

## 回归与证据

新增 `node tests/e2e/m03-preview-switch.cjs`，使用既有真实本机服务、兼容科学子进程和合成双声道输入。只在浏览器结果读取处注入延迟及 `asset_expired`，不伪造正常计算或输出。使用既有测试schema的隔离副本，未运行DDL。

- 修复前 `chrome-b0dc6f8f689145579bd60cf9b312e473` 确认旧错误出现在新文件页面，首项断言失败。
- 修复后 `chrome-7fbfd03001224fba83d30cc2bf47af64` 的四项功能检查通过；最后帮助检查误点了工作台同名按钮，等待 EGG 帮助超时。测试改为限定 EGG 页面按钮，未修改产品行为来迎合测试。
- 最终 `chrome-d3b795b8de85435b82d6763a1975bc01` 五项通过：迟到失败、迟到成功、当前失败可见、重新计算恢复，以及保留说明的浅深主题显示。0页面错误。报告与截图位于 `output/validation/m03-ui/` 对应目录，深色截图已人工查看。

`npm --prefix frontend run test`：65 passed；`run typecheck`、`run build`通过。来源生成数据检查随本轮说明保留执行，未继续查询引用。`.venv/m09-ui/Scripts/python.exe scripts/validate_docs.py`：577文件、330来源、32任务，errors=[]，历史快照失效链接单列；`scripts/check_architecture.py`：errors=[]。`npm --prefix frontend run ui-data:check`、`run contracts:check`无漂移，`git -c core.safecrlf=false diff --check`通过。

## 范围

已核实的 V2 来源说明见 [已有复核记录](m03-provenance-report.md)，其后续核查按最新要求暂停，不作为功能推进的前置条件。V2 布局/功能与 V3 公共风格约束继续有效。

本轮没有改动科学核心、源音频、V2、环境或数据库，没有打包或 push。不将浏览器注入的到期错误称为自然到期/生产认证验收，不扩大为物理声卡、多屏DPI、Qt或完整模块重验。后续优先按实际使用反馈完善开发态功能，不自动推进M04。
