# P19-R10 致谢引用复制与外链打开

2026-10-04。井井反馈所有开源与学术致谢页无法复制引用，PDF/DOI 链接单击也无法打开浏览器。

## 根因与修复

公共 `MethodReferences` 直接调用 `navigator.clipboard.writeText`，没有使用已有的桌面 `copyText` 适配器。原生宿主已有纯文本剪贴板写入槽，但这一页面绕过了它。现接入共享复制适配器，首页及各模块致谢页均采用该组件，引用内容保持作者与标题原文。

外链采用 `target=_blank`。宿主原来只处理当前框架的链接导航，没有处理 `newWindowRequested`，所以请求发出后无人接收。现由公共 `Page` 接收用户发起的新窗口请求，把有效 HTTP/HTTPS 链接交给系统默认浏览器。普通点击、Enter、Ctrl+点击及同源嵌入 M10 来源链接都共用该宿主处理。当前框架的外链处理同时支持 HTTP，工作台保持原页面。

保留自动脚本弹窗限制，文件和非网页协议不交给浏览器。没有开放网页读取系统剪贴板、允许远程页面进入本地工作台或改动科研计算。Qt [新窗口请求文档](https://doc.qt.io/qt-6.11/qwebenginenewwindowrequest.html)给出了请求 URL 和用户触发标记，处理按这两个字段限定范围。

## 定向验证

- 修改前，在 Windows Qt 6.11.2 隐藏原生窗口中复现原提示，复制未经过宿主；真实指针点击 PDF 发出用户新窗口请求但没有浏览器交接。证据 `output/validation/p19-r10/qt-baseline-78991ae79ae54506a59c88ba99dbbd41/report.json`。
- 修复后，实际前端、QWebChannel 和 Qt 用户鼠标/键盘事件验证通过：学术/软件引用原文复制、模块限定致谢复制、PDF 点击、DOI Enter/Ctrl+点击各交接一次、嵌入 M10 来源链接、自动弹窗继续阻止。证据 `output/validation/p19-r10/qt-04cf29d688df4a13a485b7e36f55a2f0/report.json`。
- 原生定向 13 项测试通过，包括剪贴板不可用、写入失败、文字超长和非网页/非用户请求边界。
- 前端类型检查、277 项测试和生产构建通过。首次构建遇到同期语音合成任务改动中的 `synthesisMethods`/`render` 不一致，在其源码更新后重跑通过，本轮没有改动或覆盖该模块实现。

原生验证使用独立数据库/缓存，系统剪贴板和浏览器启动终点替换为记录适配器，因此没有覆盖井井的剪贴板或弹出浏览器。验证证明 UI 到原生接口的完整调用，未声称实际 OS 默认浏览器配置及新冻结 EXE 已运行验收。

## 本机成品

本机新包已构建：PhoneticToolbox-v3-Latest-20261004-R7.exe（历史本地产物，当前工作区不存在；原路径 `../../dist/PhoneticToolbox-v3-Latest-20261004-R7/PhoneticToolbox-v3-Latest-20261004-R7.exe`）。大小 311,120,559 字节，约 296.71 MiB，SHA-256 为 `5e4f84b606b955dc99c262b46f95e4a3db9289d393ebb85178ea36fbcb57b99c`。沿用既有本机科学环境，`portable:false`，新 EXE 只构建，没有启动验收。无新 GitHub 推送、公开发布、数据库迁移、全局依赖安装或旧包清理，保留同期改动。
