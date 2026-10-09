# 说明书工程 — 工作规则

继承[根规则](../AGENTS.md)。这里只维护说明书工程约束，内容编写见 [AUTHORING.md](AUTHORING.md)，章节映射与排版见 [V2_STYLE_GUIDE.md](V2_STYLE_GUIDE.md)。

- `project.json` 保存章节顺序、媒体和引用索引，`chapters/*.json` 使用 `ptb-manual/1` / `ptb-manual-chapter/1` 与 Tiptap 节点。章节、标题、图表和音频使用稳定 ID，不以编号或中文标题作为身份。
- 根据当前源码和适用验收材料说明真实控件、参数和输出，不虚构素材或把未完成能力写为可用。内容状态分别记录 draft、reviewed、verified、deferred。
- 自然录音和处理结果仅允许随软件分发，禁止公开 GitHub；标记 `distribution: software-only`、`git: false`。私有原始路径只存被忽略的 `local-data/manual-authoring/`，公开媒体使用相对路径。
- 正式工程保留当前版本，自动恢复稿默认只保留最后一份；正常保存完成后清除旧历史和已提交事务，异常未完成事务先保留恢复证据并列清理回执。正文及未引用正式素材不自动删除，作者缓存不进入 Git。
- 构建不得回写源工程。使用 scripts/manual 的校验和阅读资源生成流程，正文/媒体修改只核查受影响部分，组件通过不代表全部操作正文已核验。
