# 论文精读选文与批注操作

2026-10-20追加：本页共享实现已同步公开网页，相关操作在实际HTTPS Chrome155重新验证，Linux声道回退为R21。当前网页部署以[同步验收](2026-10-20-web-module-sync.md)为准。以下保留原本机/各平台验收范围，最终Windows/macOS成品与实体设备边界不扩大。

日期：2026-10-19。任务：`M18-ANNOTATION-SELECTION`。状态：`verified`，范围为共享源码、Windows 独立 Chrome 与真实源码 Qt。Windows/macOS/网页版继续使用同一个 `PdfReader.vue`，无数据库或文档格式变更。

## 观察与修复边界

井井反馈在 Windows 桌面版普通阅读中，拖选文字后荧光笔和添加批注没有反应。最初回答选择全屏，随后明确更正为普通阅读。

修复前，独立 Chrome 和 Windows Qt 的普通拖选、荧光笔与文字批注可以通过，当前未原样复现截图中的原始失效。代码检查证实：`selectionchange` 在原生 Range 折叠或焦点移到 PDF 外时立即清空应用选文。为工具栏焦点切换注入 `getSelection().removeAllRanges()` 后，旧代码禁用荧光笔，测试失败。此注入用于复核确定的选文生命周期边界，不能据此认定井井原窗口一定产生了相同事件。

修复后保留最后一次有效 PDF 选文的引文、页码和归一化矩形，工具栏指针事件及动作执行前同步读取当前 Range。颜色选择、浏览器焦点变化、批注弹窗取消不再丢失本次选文；保存成功清除选文，失败保留供重试。重新拖选、正文空白点击、跳页、文档/纯文字视图切换、模块停用及卸载清除旧范围。拖动正文选区继续使用原生文本选择与原有矩形计算。

说明书补充普通/全屏操作、颜色选择、保存反馈、取消/失败重试，以及桌面本机批注和网页版临时工作区的保存区别。原文、译文、解读仍按各自 PDF 内容摘要隔离。

## 已执行验证

使用工程已有 Node、`.venv/m14` 和默认 GPU，不安装依赖、不改用户配置。前端命令在 `frontend` 执行：

```powershell
npm run typecheck
node --test tests/papers.test.ts tests/papers-server.test.ts tests/paper-citations.test.ts
npm run build:workbench
```

类型检查、12 项论文定向测试、保留现有工作台哈希资源的构建通过。Vite 保留已有的大 chunk 提示。

根目录执行：

```powershell
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/verify_m18_annotations.py
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/verify_m18_annotations_qt.py
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/manual/validate.py --project manual --strict
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/manual/build.py --project manual --output frontend/public/manual --distribution software
```

- 独立 Chrome 155 / 1440×1000：真实鼠标拖选和基本荧光笔、工具栏指针事件清空 Range 后换色/保存、全屏 Range 丢失后打开并保存批注、保存失败后保留选文重试、取消弹窗再次打开、点击空白/跳页/文档切换清理，以及原文/译文/解读记录隔离通过。浅深主题及全屏弹窗截图已核对。几何及存储为合成夹具，无真实论文接口请求。
- Windows Qt 6.11.2 / Chromium 140 / 默认 GPU：读取已获许可的本地 P2A 原文/译文/解读，使用隔离真实 `PaperService`，Qt 鼠标拖选；Range 丢失后普通阅读荧光笔与选文批注可保存，浅深主题标记/笔形按钮可见。另验全屏本页笔记弹窗及保存。文档切换后回读记录成功，pypdf 回读导出 PDF 含 `/Highlight` 及中文 `/Text` 批注，其他文档记录为空。此处来源为既有本机论文分发输入，测试只复制到自有 scratch，不访问公网。
- 说明书严格校验无 warning/error，21 章、1095 项素材的 software 资源生成通过。实际 Qt 帮助入口与浅深主题阅读核对单独记录在本轮 Qt 报告。
- `validate_docs.py` 与 `check_architecture.py` 均零错误，既有历史快照失效链接仍单列；Python/Node 脚本语法及定向 `git diff --check` 通过。M18 源稿、public 阅读资源和 dist 阅读资源的 JSON 内容及 LF 归一化字节一致，原始换行不同，未声称三份原始文件 SHA 相同。

证据目录：

- 修复前故障注入失败：`output/validation/m18/annotations-browser-511ad517eebb476bbcb96196ef7cb354/`。
- 当前 Chrome：`output/validation/m18/annotations-browser-00dcd32bea6f464f9d0ec676f3e3541e/`。
- 当前 Qt：`output/validation/m18/annotations-qt-89d187e5cdec4f86bafa8aacc8f12937/`。

报告、截图和输入摘要保留，隔离 profile、论文副本、导出 PDF、临时服务及缓存由各自持有者关闭，并经 `verification_run` 清理 scratch，回执见对应目录。首轮浏览器夹具误对 Vue 代理调用 `structuredClone` 导致模拟存储失败，已改成与实际端口一致的 JSON 快照，此首轮结果不作为产品缺陷证据。

额外全屏鼠标选文有不一致结果：`annotations-qt-6a611c31d5dc4c80a70482b57f782b6d` 曾通过选文批注/导出；随后 `5e8cf04aa74444289b4fba1a0ba7d96a`、`286eb42ab2f249f5978212b744b13848`、`9ae3f03e73794377a7a210a91ab4f246` 与 `76c59b573e564d709239fc502b774909` 在全屏后的再次拖选未形成 Range。后两轮保留了事件坐标与选区诊断，显式发送持有左键的 Qt move 仍未解决，原因未定。最终定向入口按井井更正的普通阅读验证鼠标荧光笔/选文批注，Qt 全屏只记录本页笔记，Chrome 全屏选文另有通过证据；不把 Qt 全屏原生选择写为稳定通过。

## 未验与交付

井井原窗口中的失效未原样复现。Qt 全屏重复拖选的稳定性待专项核对。Mac 实机、Safari、其他浏览器与线上论文服务未新增验证。未重打最终 Windows EXE/Mac 包，未部署网页或公开发布。共享源码和生成阅读资源已可用于后续三端构建与发布，当前已打开的旧窗口需重开才能加载新代码。
