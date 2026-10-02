# M10 背景与分析视图切换修复

2026-10-01，verified，限定 Windows 开发态实际 Qt 页面。井井要求本轮只统一背景、修复声学图与实时波形/语谱图切换时页面变矮，并追加移除两个试听按钮。

## 修改与原因

- `frontend/public/vocal-tract/v3.css`、`theme.js`、`scene.js`、`monitor.js`：页面沿用 U2 蓝白/蓝灰令牌，模型、声学画布、实时画布、唇部预览和 F0 底色统一。头壳参考填充改为蓝灰，组织/气腔的语义配色保留。二维与三维模型背景均随主题更新。
- `desktop.css`、`v3.css`：移除依赖 monitor 状态改变网格行数/高度的旧规则。旧 `:has(.analysis[data-view=monitor])` 规则会在自动三列的部分宽度下创建额外 255 px 网格行，使三个面板一起缩短。现在三列只占一行，两列的下方分析区始终为 255 px，切换仅替换分析区内部内容。
- 分析内容提供 160 ms 淡入，尊重减少动画偏好。两列分析区同样使用纵向 flex，并为 canvas 设置尺寸包含，避免画布像素尺寸回写参与布局、反复触发 ResizeObserver 通知。
- `index.html`、`app.js`：移除“试听当前构形”和“试听 /a → i → u/”，同步清理事件绑定及禁用控件列表，保留发声 1 秒、持续发声和测试音。

未改科学参数、原生算法、数据契约或第三方依赖。无新增第三方来源。`scripts/verify_m10_appearance.py` 是本轮独立 Qt 回归入口。

## 实际验证

| 命令 | 结果 |
| --- | --- |
| `npm --prefix frontend run typecheck` | 通过 |
| `npm --prefix frontend test` | 191 passed，0 failed |
| `npm --prefix frontend run build` | 通过，保留已有大 chunk 提醒；最终样式修复后重新构建通过 |
| `$env:PROCESSOR_ARCHITECTURE = 'AMD64'; & '.\.venv\m10-ui\Scripts\python.exe' -X utf8 'scripts\verify_m10_appearance.py'` | 30 组实际 Qt/真实 VTL 页面检查通过；展开/返回、按钮移除、三维显示通过，页面 error 事件为空 |
| `git diff --check -- frontend/public/vocal-tract scripts/verify_m10_appearance.py` | 通过 |

Qt 窗口尺寸覆盖 1280×800、1650×1000、1770×1000、1920×1080、2560×1440，分别检查浅深主题及自动/两列/三列。每组对比切换当下、完成后和返回后的模型区、分析区、参数区、模型视口边界以及 SVG viewBox，实测完全一致。1770×1000 自动布局的 iframe 为 1536×921，三个面板高度均为 850.667 px，切换后不变，覆盖旧规则冲突的宽度区间。

实际截图已检查浅/深背景、声学/实时两种内容、声音栏和三维。证据位于本机忽略目录 `output/validation/m10/appearance-20261001/`，包括 `result.json` 与六张最终 PNG。最终日志为 `output/validation/m10/appearance-20261001-verified.log`。

首次离屏 Qt 图形上下文初始化失败，随后改用移出可见桌面且不激活的独立 Windows Qt Tool 窗口。测试执行器未提供 `PROCESSOR_ARCHITECTURE`，导致原生加载器报 `native_platform_unavailable`；经 .NET 确认系统为 X64、Python 为 64 位后，仅在测试命令进程中补齐 AMD64。未修改系统环境或应用加载检查。首次完整矩阵发现 ResizeObserver 通知循环，修正 canvas 尺寸反馈后复跑全部 30 组通过，未忽略错误或放宽检查。

## 边界

本轮未重新打包 EXE，旧 EXE 不包含上述修改。未播放或录制声音，不扩大为音频硬件、视频导出、跨平台或完整 M10 验收。原生 worker 使用随机独立测试配置，退出由测试宿主清理。原数据、V2、现存数据库、系统依赖保留，无 push 或公开发布。同期其他模块差异保留。
