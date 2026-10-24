# 感知实验键盘流程、作答提示与被试分析验收

2026-10-20追加：本页共享实现已同步公开网页，相关操作在实际HTTPS Chrome155重新验证，Linux声道回退为R21。当前网页部署以[同步验收](2026-10-20-web-module-sync.md)为准。以下保留原本机/各平台验收范围，最终Windows/macOS成品与实体设备边界不扩大。

日期：2026-10-19。范围：当前 Windows 源码、共用前端、独立 Chrome/Edge 与实际 Qt。实施见[计划](../plans/2026-10-19-m15-keyboard-analysis.md)，原计时与恢复约定见[ADR](../decisions/ADR-M15-client.md)。

## 实际变化

- 问卷选项逐行编辑和添加/移除，直接键入中英文逗号；空白或去首尾空格后重复阻止预检。
- 被试问卷支持原生 Tab、方向键、空格与回车。指导语、阶段继续、进入试次、手动播放使用空格/回车，页面明示；Esc 暂停。推进事件和 held 状态跨阶段保留，长按和播放中事件不计为答案。
- 参数增加作答页提示语及显示开关，播放/等待作答页使用安全静态文本。默认关闭，旧项目缺字段也不显示。网页账号只保存这两个偏好字段，不保存试次/问卷答案/结果到偏好。
- 每试次标准反应键可在序列页或 XLSX 的 Correct_Key 列设置，Space 表示空格。项目/会话 JSON、序列 XLSX、生成、复制与洗牌保留。原结果表仅末尾追加 Correct_Key 与 Is_Correct。
- 分析页展开当前项目各会话，显示 RT 均值、中位数、Q1/Q3/IQR、样本SD、极值、准确率及计数。同名记录保留 UUID，不自动合并。
- 默认用会话答案快照。当前设计重评分需显式选择，只匹配项目、范式、试次 ID、角色 hash 与原反应键，不回写会话。XLSX 的 Participants 为汇总、Scoring 为逐次评分，CSV 为汇总。

## 统计口径

每试次取最后 completed 作答，RT 仅用有限且非负的数值，保留零，不自动剔除离群值。空样本统计为空，单样本SD为空。Q1/中位数/Q3 使用 [R quantile type 7](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html) 线性插值，IQR=Q3−Q1，样本SD分母 n−1。方法来源统一登记为 REF-M15-R-QUANTILE-TYPE7，独立实现公式，没有引入 R 运行库或复制上游代码。

准确率为正确数/有标准答案且完成作答的试次数；无可计分值显示未计分，未完成、未设答案/无法计分、缺失RT及异常计数分别列出。RT 起点仍为作答窗口开放，音频整段结束后开放，不将软件时钟当作物理耳机/键盘时延。

## 定向检查

执行于项目现有 Node 24、Windows、Chrome 155/Edge 155，Qt 使用 .venv/m14，source binder 指向当前工作树。

```powershell
node --test frontend/tests/m15.test.ts frontend/tests/m15-runner.test.ts frontend/tests/m15-lifecycle.test.ts frontend/tests/m15-analysis.test.ts
node --test frontend/tests/account-preferences.test.ts frontend/tests/web-settings.test.ts
npm --prefix frontend run typecheck
npm --prefix frontend run contracts:check
npm --prefix frontend run ui-data:check
npm --prefix frontend run build
python -B -X utf8 scripts/verify_m15_keyboard_analysis.py --browser chrome
python -B -X utf8 scripts/verify_m15_keyboard_analysis.py --browser edge
& '.venv/m14/Scripts/python.exe' -B -X utf8 scripts/verify_m15_keyboard_analysis_qt.py
python -B -X utf8 scripts/manual/validate.py --project manual --strict
python -B -X utf8 scripts/manual/build.py --project manual --output frontend/public/manual --distribution software
python -B -X utf8 scripts/check_architecture.py
python -B -X utf8 scripts/validate_docs.py
```

- 感知实验40项通过：解析值100/200/300/400的均值250、Q1=175、Q3=325、IQR=150及样本SD；零/空/负/非有限值、重复完成、中断、未计分、答案匹配/不可变、JSON/XLSX往返、逗号/空白/重复、追加列、逐次评分。另覆盖阶段键不泄漏、自动推进/播放、保存失败/恢复与迟到提交。
- 设置18项通过；后端 `test_perception_preferences.py` 经当前源码绑定、`-c tests/pytest.ini` 执行6项通过。仅偏好字段白名单，无测试库或迁移。
- Chrome/Edge 使用独立进程和 scratch 内专属配置目录。真实导入4份合成 PCM16 WAV（16kHz、0.2s、220+i×30Hz、峰值1500），从问卷到最终反应仅发键盘事件。验证播放中忽略按键、独立被试/提前结束、无答案与显式重评分、RT均值/Q1与准确率、实际JSON/XLSX/CSV回读、提示语开关、Esc暂停、浅深色1440/900宽布局和说明书帮助入口实际渲染。
- 实际 Windows Qt：off-record配置、两份UTF-8 TXT刺激，QTest原生键盘完成问卷/开始/进入/手动播放/J与F作答，实际JSON回读，分析显示50%，XLSX回读Participants/Scoring，浅深截图。听音确认由自动化勾选，未声称人耳或实体设备测量。
- 类型、契约、UI数据及前端构建通过。说明书21章/1095素材严格校验零错误，生成阅读资源；架构零错误，文档检查2349文件/448来源/169任务零错误，定向diff检查通过。历史快照中的旧失效链接单列，没有改动。

## 本机证据与收尾

本机证据根目录：`output/validation/m15-keyboard-analysis/`。单元/设置/类型/构建与文档日志保存在根部。每次浏览器/Qt运行保留报告、必要截图、生成配方、输入/导出摘要与清理回执；合成媒体、导出和配置目录放 scratch，由共享生命周期清理器复核归属、联接与自有进程后清理。

过程中修复：首轮 Vite 动态端口落入浏览器禁止端口，改用正常测试端口；分析页展开后的窄屏交叉轴未拉伸，修复后原布局断言通过；Qt测试的子进程继承安全导入路径导致测试辅助模块未找到，显式绑定当前scripts目录。Qt首轮在进程退出前清理失败，后改为父级监督并在子进程退出后收尾。保留失败报告，未降低断言或改产品科学语义。

最终本机证据：

- Chrome155：`output/validation/m15-keyboard-analysis/chrome-7f4ed0cfc6424399bfec95097c0e7c5a/report.json`，包含新说明书分析段实际定位截图。
- Edge155：`output/validation/m15-keyboard-analysis/edge-df27111c7c424fc6959b5ca4996fd3b7/report.json`。
- 原生Qt：`output/validation/m15-keyboard-analysis/qt-828f73cf068d4b18b5c575b6c190562a/report.json`。
- 本任务全部运行的scratch均已清理，包括首轮Qt保留项在自有进程退出后的补清；每轮cleanup.json为deleted。正式说明书素材、既有项目/会话、运行环境与其他任务产物保留。

## 验收边界

三端共用源码与说明书已同步。macOS/Safari实机、实体声音/屏幕/键盘端到端时延、长期与大规模被试运行未验。当前生产网页及最终 Windows/macOS 安装包未更新。本轮无公开部署、发行、push、现存库迁移、全局依赖或系统设置修改。
