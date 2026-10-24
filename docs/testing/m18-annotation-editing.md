# 论文标注撤销、荧光笔删除与便签验收

2026-10-20追加：本页共享实现已同步公开网页，相关操作在实际HTTPS Chrome155重新验证，Linux声道回退为R21。当前网页部署以[同步验收](2026-10-20-web-module-sync.md)为准。以下保留原本机/各平台验收范围，最终Windows/macOS成品与实体设备边界不扩大。

日期：2026-10-19。任务：`M18-ANNOTATION-EDITING`。状态：`verified`，限定 Windows 当前源码、独立 Chrome 与真实 Qt 默认 GPU。三端共用同一阅读器与现有标注端口，Mac 本轮仅源码，无格式变更或数据库迁移。

## 行为

- Ctrl+Z 撤销、Ctrl+Shift+Z 重做，macOS 同时识别 Command。工具栏提供撤销/重做按钮。新增、编辑与删除都记录历史，同次跨页荧光笔为一步。当前打开文档的最近100步保留在会话内，切换文档或重新载入标注清历史；textarea/input 或弹窗中的文字撤销由原生编辑处理。颜色下拉仍可触发标注快捷键。
- 历史只记录已保存操作。撤销/重做沿用原修订号写入，失败不提前移动游标；多项操作部分成功时，历史按实际回执拆分。跨页标记部分失败后只保留未完成页的选文，重试不重复添加已经成功的页。
- 纸面覆盖层保持 `pointer-events:none`，通过点击位置命中原归一化矩形，选择最新重叠荧光笔，显示边框及删除按钮。Delete/Backspace可删除选中的标记，按钮和侧栏删除也可撤销。拖选标记上的文字不被覆盖层拦截。
- `AnnotationNoteDialog.vue` 复用公共 ModalDialog 的焦点与退出语义，批注弹窗和列表均只显示批注文字，采用黄/绿/粉便签配色及对应深色主题。内部引文、页码和矩形保留，继续用于定位及标准 PDF 注释导出。
- M18 正式说明书、README、当前状态、台账及修改记录同步维护，阅读资源已生成。

## 验证与证据

工程已有 Node / `.venv/m14`，未新增依赖或更改用户配置。前端目录：

```powershell
npm run typecheck
node --test tests/paper-annotation-history.test.ts tests/papers.test.ts tests/papers-server.test.ts tests/paper-citations.test.ts
npm run build:workbench
```

17项通过，其中5项历史测试覆盖新增/编辑/删除往返、跨页分组与快照、失败不推进、部分撤销与部分重做、重做分支失效及历史上限/隔离。类型检查与保留旧哈希资源构建通过，Vite 既有大 chunk 提示仍保留。

根目录：

```powershell
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/verify_m18_annotations.py
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/verify_m18_annotations_qt.py
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/manual/validate.py --project manual --strict
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/manual/build.py --project manual --output frontend/public/manual --distribution software
```

- 独立 Chrome155 / 1440×1000：10组交互通过，含实际鼠标拖选、注入Range清空后的操作、Ctrl+Z/Ctrl+Shift+Z与按钮、重叠荧光笔选中/删除/恢复、批注编辑与删除、输入框原生文字撤销、失败撤销重试、跨页一步与部分保存重试、文档隔离和普通/全屏便签。便签实际 computed 背景为浅绿 `rgb(226,243,217)`、深绿 `rgb(41,55,43)`，宽500 CSS px，无原文 blockquote。几何与存储使用合成夹具，此项不证明线上论文服务。
- Windows Qt6.11.2 / Chromium140 / 默认GPU：8组通过。原生快捷键真实写入撤销与重做；纸面点击/Delete删除/恢复，选文批注写盘，编辑撤销/重做，输入框文字撤销不改已保存标注。黄便签浅深截图已核对。文档重开回读，其他语言无记录，pypdf回读导出副本仍含 `/Highlight` 与中文批注；M18帮助入口加载新正文并检查浅深显示。输入使用既有获许可P2A本地分发文件，仅复制到自有scratch，无公网下载。
- 经 `workbench_source.configure()` 绑定本工作树后，`pytest -c tests/pytest.ini desktop/tests/test_paper_annotations.py -q`：10项通过。
- 说明书严格校验零warning/error，21章/1095素材software阅读资源生成通过。
- 文档检查2354文件/448来源/170任务零错误，架构检查零错误；定向diff、Node/Python脚本语法通过。M18源稿、public与dist阅读资源的JSON内容及LF归一化字节一致。最终保留资源构建后重新执行Qt8组通过。

当前证据：

- `output/validation/m18/annotations-browser-3eba1e953d0f43cd8da4c88e7554b9be/`。
- `output/validation/m18/annotations-qt-b8a34b78f0fc4fafa2e57bd676a95b8f/`。

两轮报告、必要截图及输入摘要保留，浏览器profile、隔离论文/数据库/导出副本由所有者停止并经 `verification_run` 清理，`cleanup.json` 均为 `deleted`。

## 修复过程中的失败

首轮 `annotations-browser-22a2e3525c034698aef4c0fa60bc2158` 在颜色选择后 Ctrl+Z 没有触发：焦点留在 select，旧快捷键保护把它与文本输入一起排除。现允许 select 的标注撤销/重做，同时保留 input/textarea/弹窗保护。

第二轮交互通过后，截图和computed样式检查发现便签根样式未生效，原因是 scoped CSS 在根组件使用无宿主的 `:deep()`；带 `:global()` 的主题组合还将规则编译到主题根节点。已改为匹配根dialog并将deep限定为子元素，深色规则限定到实际卡片/便签。编译结果、实际背景与宽度断言和最终浅深Qt截图通过；该轮样式未完成的截图不当作最终验收。

## 边界

未打最终 Windows EXE/Mac 包，未更新线上网页，Mac 实机/Safari未验。已打开的旧工作台仍需重开加载。此前原始无响应与 Qt 全屏重复拖选问题的证据和范围保留在[选文验收](m18-annotation-selection.md)，本任务没有将这些问题扩大标为已解决。
