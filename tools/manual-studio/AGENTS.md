# 说明书作者工具 — 工作规则

继承[根规则](../../AGENTS.md)。本目录的依赖独立安装，产品前端不可反向导入作者工具。

- `src/` 为 Vue/Tiptap 编辑器，`server/` 仅绑定 127.0.0.1；正式预览复用 frontend/src/manual/ManualDocument.vue 及既有工程格式。
- 保存只作用于明确打开的工程，校验相对路径、同源、会话能力和内容哈希，媒体不写绝对来源路径。
- 删除章节仅撤下索引，未引用正式正文/素材不自动删除。自动恢复稿默认只保留最后一份，先原子写成新稿再清旧稿；正常保存完成后清历史/已提交事务，未完成事务保留恢复证据与回执。
- 已结束的独立测试工程按根规则清理，操作前核对 test-output/ 的真实路径、归属、活动进程与联接，不影响正式工程或联接目标。
- 检查：`npm run typecheck`、`npm test`、`npm run build`；需要浏览器往返时用 `node tests/browser.mjs`。
- 不处理科研音频，不修改产品依赖、科学核心、数据库或 CI/CD。`.runtime/`、`test-output/`、node_modules/ 和 dist/ 不进入 Git。
