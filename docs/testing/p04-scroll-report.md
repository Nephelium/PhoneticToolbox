# P04-SCROLL 公共内容滚动验收

2026-09-12。状态：verified，限定 Windows 开发态公共布局、EGG/M02 二维滚轮、Qt 窗口约束。完整 M03/E3 仍 in_progress，旧 EXE 未重新打包。

## 问题与改动

宽窗口 main.pane-workspace 原先 overflow:hidden，M02/M03/M09 的自然高度内容可被裁切。公共 main 改为 overflow:auto，保留 M01 的独立列滚动。ScientificPlot 和 ParameterFigure 普通滚轮不再 preventDefault，也不发出缩放；Ctrl＋滚轮保留原缩放系数与计算路径，说明同步。ModalDialog 使用 flex 首尾与可滚动正文，底部操作不随正文离开视窗。

Qt 初始 1440×900、最小 800×500 逻辑像素，限制在屏幕 availableGeometry 减去 32×64 边框余量内；切换屏幕重新计算。声道 iframe 外层最小高度 640px，矮窗口通过公共 main 滚动查看；本次不改声道模型缩放手势、算法或录制。

## 实际执行

- `npm --prefix frontend run typecheck`：通过。
- `npm --prefix frontend run test`：50 项通过。
- `npm --prefix frontend run build`：通过。
- `node tests/e2e/workspace-scroll.cjs`：10 组通过，pageerror 为零。真实核心子进程计算公开合成 EGG，1280×800、1024×600、800×500、1440×450 浅深主题均能用普通滚轮到达底部；时长和任务数不变，Ctrl 缩放重新计算。批量弹窗正文滚动，footer 坐标不变。M01/M02/M09 公共溢出样式通过；M02 另用明确合成参数表的实际组件验证普通滚轮不改曲线，Ctrl 滚轮改变视窗。
- `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_workspace_scroll_qt.py`，进程内 PYTHONPATH 指向 backend/src 与 desktop/src：9 组通过。使用真实 QWheelEvent 进入 QWebEngineView，1280×800 和 800×500 可滚到底部；Ctrl 缩放、弹窗首尾和 1.25/1.5/2.0 页面缩放通过。宿主当前截图为 1.5 倍设备像素。小屏 720×460 的 availableGeometry 替身验证最小值和实际尺寸均收紧到 688×396，再恢复正常最小值。声道仅验证 iframe 页头滚轮传递给外层及最小高度，截图仍处模型准备阶段，不作为声道渲染/录制验收。
- `python scripts/validate_docs.py`、`python scripts/check_architecture.py`、`git diff --check`：通过；历史快照原有失效链接仍单列。

## 本机证据

浏览器：output/validation/m03-ui/chrome-4397ba4098bb499f91f4c041cda4a5a9/report.json，四尺寸截图及 dialog-scroll.png。

Qt：output/validation/m03-ui/qt-scroll-af6220458c554c508da2aeef012fc0b9/report.json，bottom-1280-800.png、bottom-800-500.png、dialog.png、vocal-scroll.png。实际查看过浏览器 800×500、弹窗，以及 Qt 小窗底部和声道外层截图。

首次 Qt 检查 qt-scroll-666592703bc14f48bddc243b485e47fd 因测试代码中选择器引号转义错误失败，保留报告；改用 JSON 编码选择器并检查初始值确为 0.5 后，qt-scroll-3a31610431ca42d7a89fe03950295b52 及追加缩放/声道检查后的最终运行通过。没有放宽产品检查。libpng 历史资源色彩信息警告保留，未更改资源。

## 边界与下一步

未新增外部代码、依赖或方法来源。未改科研值、数据库 schema、v2、现有语料或旧 EXE，未 push。页面缩放不等同于三个真实系统 DPI 设备，未实测多显示器移动或触控板硬件。未实施模块不宣称逐页通过，M10 模型手势仍沿用既有行为。后续模块遵守 UI_SPEC 滚动规则；M03 回到 E3-B 长文件全段语义与预算、字体预检等剩余项，另行完成冻结 EXE 验收。
