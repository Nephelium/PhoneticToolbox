# M01 共用基频设置与本章说明书审阅

2026-10-10。井井要求将最小／最大基频移到常用设置，REAPER 页仅保留 Hilbert 变换与禁用高通滤波两个开关，并同步、审阅参数估计这一章。

## 修改与审阅结论

共享 `SettingsDrawer.vue` 现在显示常用设置 12 项和 REAPER 设置 2 项。最小／最大基频相邻排列，页签数量由实际分组计算。原字段、默认值、校验、草稿和任务提交配置保持不变。源码 `acoustic.py` 的 Praat、REAPER 与 WM 调用继续传入同一组 `min_f0`／`max_f0`。

正式 `manual/chapters/m01.json` 与模块 README、历史 Markdown 操作入口已经同步。本章审阅覆盖 37 个标题、16 张表、80 个目录参数，以及输入关联、试听、分析范围、联合 EGG、保存、切分和错误恢复的说明；不扩展到其他章节。

| 发现 | 处理 |
| --- | --- |
| 正文、按钮速查和截图仍为常用 10／REAPER 4，基频范围归入 REAPER | 改为 12／2，解释共用基频范围与两个专用开关的默认含义 |
| 900–800 Hz 校验示例位于 REAPER 小节 | 移到常用设置小节，保持图表及段落稳定 ID |
| 五张完整／局部设置图片仍显示旧分组 | 用当前 Windows Qt 实际界面重新取图并登记摘要，保留源工程旧素材 |
| WM 微扰说明可被读成界面分析窗必须至少 160 ms | 明确实际计算窗为 `max(160 ms, 分析窗口)`，界面默认仍为 40 ms |
| TextGrid 选区步骤未说明现有空格播放／停止 | 补齐操作与输入框、普通按钮的焦点边界 |
| 历史 Markdown 入口仍写侧栏 220／250 | 按统一工作台源码改为两侧各 300 逻辑像素 |

本章的 1797 个稳定 ID 全部保留，80 个目录键均有说明。14 项设置默认值、范围和单位与当前契约核对；联合 EGG 默认值与 `JointEggSettings.vue`／`state.ts` 核对。CQ／SQ、gF0 取时、缺失值、预览与科学输入的边界仍与现有实现对应。搜索索引同步更新本章，其他章正文和搜索条目保留。

## 实际验证

环境：Windows，Node 24.13.0／npm 11.6.2，项目 `.venv/m14` 的 Python 与 Qt。截图使用自有隐藏 Qt 实例、独立任务库副本和缓存，不读取研究录音，不提交科学计算任务。正式完整图为 2560×1600，Qt DPR 1.5、页面 100%、最大化状态和 1707×1067 逻辑视口，原始 2561×1601 仅裁右下各 1 像素；这属于独立取图实例，未更改系统缩放或用户设置。

- `npm --prefix frontend run typecheck`、`contracts:check`、`ui-data:check`、`build` 通过。阅读资源更新后再次构建通过。
- `node --test frontend/tests/m01-state.test.ts frontend/tests/m01-r3.test.ts frontend/tests/m01-r4.test.ts frontend/tests/m01-r5.test.ts frontend/tests/m01-legacy.test.ts`：16 项通过。
- 实际 Qt 设置检查：常用页恰为 12 个控件，基频上下限相邻；REAPER 恰为两个开关且默认开／关正确；切页保留输入、取消放弃、应用后重开保留；900–800 Hz 阻止应用；浅深主题通过。
- `scripts/manual/validate.py --project manual --strict`：无错误、无警告，含引用目标与媒体 SHA-256 检查。
- `scripts/manual/build.py --project manual --output frontend/public/manual --distribution software`：成功生成当前阅读资源。校验器检查整份工程的结构依赖，本轮人工内容审阅仅限 M01。
- 实际 Qt 阅读器：M01 帮助打开更新后的章节，12／2 文字正确，五张更新图片均实际解码，常用／REAPER 稳定标题锚点定位及截图检查通过。

本机证据在 `output/validation/m01-settings/`：`qt-b2d41c67d4334cb5affab384849891f2/report.json` 为正式设置截图与交互结果，`qt-67488945c0704e2f82e9604dca1ce55b/report.json` 为阅读器与实际图片检查，`manual-validation.json` 为严格工程检查，`frontend-build.log` 为更新阅读资源后的构建。取图脚本、正文展平稿与中间诊断保留用于复核；全部十次独立 Qt 运行的临时库／缓存已在进程退出后按 `cleanup.json` 回执清理。

阅读器检查脚本的初版使用绝对 URL 查询相对 `src` 属性，又把四级标题当成二级目录按钮，产生了定位假失败。实际诊断中的五张图片均已解码。最终改用 `HTMLImageElement.src` 和正文稳定锚点，保留原图片读取条件，不修改产品代码或降低检查标准。失败报告与修正后的成功报告分别保留。

## 全量检查的独立问题与边界

`npm --prefix frontend test` 为 345 项中 344 项通过、1 项失败。失败来自既有 `source-coverage.test.ts`，M05 的 `third_party/acknowledgement-coverage.json` 未同步当前来源登记中的 `PKG-PYAV`；相关来源文件在本轮开始前已有修改，本轮未修改它们。这一失败单列，不能报告全量测试通过。

`scripts/validate_docs.py` 本轮检查 1813 个文件，报告 1 个当前错误：同期 M03 README 指向尚不存在的 `docs/testing/2026-10-10-m03-r8-report.md`。另保留 36 个未改变历史快照中的链接提示。本次修改的文档未出现在错误列表中，未改动 M03 文件。具体输出为本机 `docs-validation.json`。

本轮不重复声学数值基准、全部自然语料与 EGG／切分任务，未重读所有方法论文，未构建或验证最终 EXE、实体设备及其他平台。引用核对限于现有工程记录和跳转完整性。冻结 EXE 仍需下一次打包才包含当前源码和说明书。
