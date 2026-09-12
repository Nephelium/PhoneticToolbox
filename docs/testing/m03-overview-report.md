# M03 音频总览紧凑布局

2026-09-12。井井追加明确反馈：图窗置顶，双声道开关、缩放和适合窗口放在图下，不保留顶部空白或适合窗口独占一行。

状态 verified，限定 Windows 开发态布局。WaveformViewport 新增可选 compactOverview，仅 EGG 总览启用；单声道波形直接置顶，高度110px，时间轴后是总览标题/双声道/缩放/适合窗口/简短提示。长文件平移滑条在下一行。去掉旧隐藏标题所保留的空间，并纠正原先未匹配到SVG的高度选择器。其他模块保持默认布局，数据、选区、双声道与60秒总览规则不变。

验证命令：`npm --prefix frontend run typecheck`、`run build`、`test`（50项通过）。`node tests/e2e/m03-overview.cjs` 在真实共享页面读取120秒合成录音，1440深色/800浅色波形到容器顶部间距小于15px、图高110px，适合窗口与双声道开关同排；双声道/2倍到4倍缩放/适合窗口恢复60秒及起点0均通过。证据 output/validation/m03-ui/chrome-84b6f304dc774f5082bc36a3c44e292a/report.json，3组通过、无页面错误。

`.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m03_overview_qt.py`，进程PYTHONPATH指向backend/src与desktop/src：原生Qt相同布局与按钮2组通过，证据 output/validation/m03-ui/qt-overview-26ca8554986b4c389f5c0d87eb0ac2cb/report.json。浏览器深浅总览截图已目视检查。

本布局改动没有新增来源、算法或依赖，未更改其他模块默认布局。此前长录音成果保存在本地提交9cbbf3c，见[长文件验收](m03-long-report.md)。旧EXE未打包；完整M03仍in_progress，下一项字体预检/剩余收口。
