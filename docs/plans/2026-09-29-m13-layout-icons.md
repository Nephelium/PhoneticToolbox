# M13-R1 / P04-ICON

状态：verified（限定 Windows Chrome、实际 Qt 宿主与 HWND 图标、WSL 静态资源回读）。井井于 2026-09-29 根据截图授权四项界面修复。见[验收报告](../testing/m13-layout-icons-report.md)。旧 EXE 未重打，Linux 原生界面未验。

- 两种排布共用右侧操作与参数栏，左侧仅切换输入/结果横排或上下排。窄视口保持右侧控件，内容可横向滚动。
- 多音字使用字旁约 240×240 逻辑像素浮窗，长列表内部滚动，贴近视口边缘时避让。支持选择、关闭、Escape、点击外部关闭；滚动和缩放后重新定位，文本/排布改变时关闭。
- 收起侧栏和响应式隐藏文字时保留 IPA 图标。
- Qt 窗口与应用图标复用当前前端构建中的 K2 图像，Windows 设置应用身份，避免归入 Python 默认任务栏组。

修改文件：M13 页面、公共 tokens.css、desktop 宿主/图标适配、定向 Chrome/Qt 验证及说明文档。不改转换字表、导出计算、数据库或其他模块行为。

验收命令：前端 test/typecheck/build；`node tests/e2e/m13.cjs`；新增 M13 布局定向 Chrome 检查；`scripts/verify_m13_qt.py`；desktop tests；WSL 静态构建回读。EXE 构建与公开发布不属于本次请求。
