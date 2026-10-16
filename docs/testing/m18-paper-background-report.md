# 论文 PDF 背景可读性

2026-10-13，任务 `M18-PDF-PAPER-COLOR`，Windows 当前源码和实际 Qt 范围 verified。

## 原因与修复

Qt PDF 栅格保留原始文字颜色，纸张背景可能透明。开启背景图后，共用样式将 `.pdf-page` 强制设为主题面板底色，深色面板与黑色文字组合导致阅读困难。

`frontend/src/design/background.css` 将 PDF 页面自身及其后代排除于透明布局规则，并取消页面的主题面板底色覆盖。沿用 PdfReader 的白色纸张，原文、译文和解读保留原始颜色。未修改 PDF 文件、渲染算法、科研内容或批注数据。

## 验证

- `frontend`：`npm run typecheck`、`node --test tests/papers.test.ts tests/paper-citations.test.ts tests/appearance.test.ts`（13 项）、`npm run contracts:check`、`npm run ui-data:check`、`npm run build` 通过；构建保留既有大分块提示。
- `.venv/m14/Scripts/python.exe -E -s -B -X utf8 scripts/verify_m18_paper_background.py` 通过。使用隔离 PaperService、离线许可资产与实际 QPdfDocument 渲染，在可见 Qt 默认 GPU 窗口中检查。
- 本地 `2408.05641v1/layout-20261012/publish` 中的原文、译文及解读均先核对目录 SHA-256。三个文档分别检查 dark/light 主题属性与背景 on/off，12 组页面计算底色均为白色，截图角落像素均为 `#ffffff`，图像滤镜为 `none`。
- 图像与透明可选文字层继续叠合，透明度仍为 0，实际选中文字后复制选文按钮可用。人工查看深色背景原文截图，标题、作者与正文黑字清晰。
- 最新本机证据：`output/validation/m18/paper-background-qt-2f7c592563a24d9f96390b4ca35b8886/report.json` 及 12 张截图；`cleanup.json` 确认 scratch 已删除。原始论文及其他任务文件保持。
- `scripts/validate_docs.py` 检查 2029 文件、427 来源与 110 任务，errors 为空；受影响文件的 `git diff --check` 通过。既有历史快照缺链继续单列。

首次验证为测试宿主传入尚未初始化的独立任务库，启动失败。PDF 检查不需要任务库，改为宿主既有无任务库模式后通过。首轮窗口延迟释放影响 scratch 清理，验证脚本补齐 Qt 对象销毁和事件处理，第二轮自动清理通过，首轮在进程退出后经同一清理入口成功收尾。失败诊断及首轮报告保留。

## 范围

主题通过根属性与实际主题变量切换，验证针对 PDF 样式组合，未扩大为设置持久化验收。本轮未重建最终 EXE，未验证其他系统或长期阅读。源码工作台须重新打开以加载新前端资源，已安装应用须后续构建包含该修复的版本。
