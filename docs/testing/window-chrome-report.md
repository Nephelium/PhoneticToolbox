# 窗口标题栏与滚动条

本报告记录原生标题栏配色与共用滚动条的已验收检查点，范围为 Windows 11 build 26300 的源码工作台及实际 Qt WebEngine。Windows 当前窗口已改用[顶边标签栏](compact-shell-report.md)，单一品牌与标签同一行，以下 DWM 标题像素记录属于改动前证据。共用滚动条与默认展开的配色说明继续沿用。DWM 兼容路径依据 [Microsoft DWM 文档](https://learn.microsoft.com/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute)。其他平台保留系统装饰。

## 实现

- `desktop/src/ptb_desktop/window_chrome.py` 对本窗口设置 DWM 深浅/标题背景/文字属性，桥只接受两个 `#RRGGBB` 颜色；不修改系统外观。前端同步公共 sidebar/text tokens。
- 原生标题栏为桌面唯一图标/名称区。侧栏品牌区仅在浏览器预览保留，桌面工具搜索和菜单上移。
- `frontend/public/scrollbars.css` 为主文档及 M10 嵌入文档共用。轨道透明、9 CSS px 外宽、滑块内部 5 px、圆角、无箭头，悬停/拖动加深。兼容强制颜色模式的样式保留，但未实测系统高对比度。
- 配色下拉菜单后直接显示默认展开的“关于配色”；仍可手动收起。正式说明书文字及四设置页配图已同步。

## 实际验收

入口 `.venv/m14/Scripts/python.exe scripts/verify_window_chrome_qt.py` 使用既有 schema 测试库副本与独立 profile。最终证据在 `output/validation/window-chrome/qt-07a00e4742584259afad5495a7f1a98e/`，`report.json success=true, closed=true`。

- Everforest/Codex/Rose Pine 浅色与 Everforest 深色共四组，PrintWindow 原生窗口标题背景像素与公共 tokens 一致，标题文字属性设置成功。
- 非法桥颜色不修改标题栏。最小化、最大化、还原和移动后标题颜色保留，原生关闭停止所属服务。
- 主文档与 M10 嵌入文档均有 9 px 滚动条、圆角和无箭头。实际 Qt PageDown、竖向滑块拖动及横向滚动通过。
- 桌面侧栏无重复品牌，配色说明 open=true 且紧接下拉菜单所在区块。
- 当前四设置页、LPC 和 M02 配图证据在 `output/validation/settings-navigation/manual-qt-1b79428decf24b85930602f31b2284b1/`，正式说明书使用对应四设置页。

前端 appearance 五项、typecheck、contracts:check、ui-data:check 和 build 通过。资源摘要更新了 M10 v3.css 与共用 scrollbar CSS，最终架构/文档结果见本机 `output/validation/settings-icons/window-architecture.json` 与 `window-docs.json`。

初期检查误用 DWM 只支持写入的颜色属性进行读取，已改为原生窗口像素核验。Pillow 的按 HWND 截图未正确取得原生边框，最终使用仅捕获自有窗口的 PrintWindow，未使用桌面截图作为验收证据。M10 导航由真实模块菜单打开。失败诊断保留，最后通过范围以最终报告为准。

Qt 仍持有测试文件时首次 scratch 清理返回 retained。进程退出后逐个复核路径并调用项目清理入口，所有本任务 scratch 回执为 deleted，报告及截图保留。

最终 EXE、Windows 10、其他平台、系统高对比度、完整物理 DPI/多屏、贴边布局交互未实测。科研任务/文件和数据格式未改动。
