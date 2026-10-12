# VoQS 符号解释修订

任务 `M17-VOQS-EXPLANATIONS`，2026-10-11。当前状态：`verified`，限 Windows 源码与独立 Chrome。

## 内容与范围

参考作者本次指定的《音质符号及其翻译问题》修订悬停与详情解释。原文只读，不复制进仓库，不新增该文的应用引用。通过既有 `symbol-content.json` 维护入口覆盖 65 项含义、用法与区别，包括 26 项发声类型。目录、原表中文名、英文名、Unicode、插入方式、来源定位和媒体配置不变。

发声类型重点区分耳语、耳语声、气声、嘎裂与嘎裂声，解释复合符号各组成部分，并补充假声带发声与双音调嗓音等名称。气流、喉位、喉上设置和范围工具也逐项完善。重复的通用临床提示改为与条目直接有关的区别，临床符号保留必要解释。糙声范围模板的说明按实际包裹选区行为修正。

未照搬文中固定基频、气流量等示例数值，也未采用嘎裂声仅等于升高基频的嘎裂、室襞发声必然只能提取一个基频等简化表述。喉位与音高、假声与假声带、咽门柱局部动作与咽腔整体扩展分别说明。参考核对了既有目录与生成源中的解释，以及公开的[喉构音模型书籍概述](https://www.cambridge.org/core/books/voice-quality/62327F676CC6B8823389A76449731F85/listing)、[发声类型与音高维度研究](https://www.isca-archive.org/icslp_1992/esling92_icslp.html)及[糙声研究](https://pubs.aip.org/asa/jasa/article/157/4/2582/3342867/Harsh-voice-and-its-interaction-with-vowel-quality)。后者仅用于交叉核对糙声描述，没有新增软件引用。

## 验证

- `node --test frontend/tests/m17-ipa-plus.test.ts frontend/tests/m17-r2-sources.test.ts frontend/tests/m17-r3.test.ts`：22 项通过，覆盖目录、原始译名、字符插入、撤销、来源及悬停定位。
- `npm --prefix frontend run typecheck`、`contracts:check`、`ui-data:check`：通过。
- `scripts/manual/validate.py --project manual --strict`：20 章、1044 素材，0 错误、0 警告。
- `scripts/manual/build.py --project manual --output frontend/public/manual --distribution software`：阅读资源已生成。
- `scripts/check_architecture.py`：0 错误。
- `npm --prefix frontend run build`：通过，保留现有大 chunk 提示。
- 内容检查：覆盖精确等于全部 65 个 VoQS 稳定 ID，每项仅含 descriptionZh、usageZh、contrastZh，维护工具格式校验通过，无文章题名或作者身份进入条目。
- `node output/validation/m17-voqs-explanations/check.cjs`：独立 Chrome 逐项打开 65 个详情，确认三段内容均实际显示。6 个代表条目在 1366×768 / 1024×720、浅深主题、100% / 150% 页面缩放下共 48 组悬停检查通过，无横向溢出，面板处于视口内，指针可移入并滚动。糙声 Unicode 输入、撤销及糙声范围包裹选区通过，页面错误为 0。
- 目视核对糙声浅色与复音深色截图，中文、音标、段落及已有来源均可读。
- 受影响 JSON / UTF-8 无 BOM、文档本地链接及 `git diff --check` 通过。

浏览器证据和截图保存在 `output/validation/m17-voqs-explanations/2026-10-11T12-30-00-189Z/`。运行配方保存在同主题目录的 `check.cjs`。自有浏览器、Vite 服务已关闭，独立 profile 经项目清理入口清理，`cleanup.json` 为 `deleted`。没有生成音频或复制文章。

## 交付边界

本次调整限于符号介绍及配套文档，未修改科研算法或原文章。源码工作台重启后可读取本次前端构建。未重新打包最终 EXE，未验证原生 Qt、实体设备与跨平台显示。同期改动保留。
