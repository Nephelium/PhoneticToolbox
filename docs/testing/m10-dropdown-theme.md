# 生理参数合成下拉菜单主题验收

2026-10-13，`M10-DROPDOWN-THEME`，Windows 当前源码与实际 Qt 范围为 `verified`。

## 修改

声源模式展开后出现白底浅字。原样式只指定闭合 select 的背景与文字，原生弹出列表中的 option 缺少显式不透明底色。`frontend/public/vocal-tract/v3.css` 为全部 select option / optgroup 指定公共 panel / ink，选中底色使用 selected，禁用文字使用 muted。菜单数据、合成参数、事件与原生操作保持原样。两份资源清单及迁移证据只更新本次 CSS 与 bundle 相关摘要。

## 验证

- `npm run typecheck`、`node --test tests/appearance.test.ts`（5项）、`npm run build` 通过，build保留既有大chunk提示。
- `.venv/m14/Scripts/python.exe -B -X utf8 scripts/verify_m10_dropdown_theme.py` 通过。脚本绑定当前源码，使用隔离 Qt workbench、临时声道配置和实际原生引擎；不使用用户研究文件，不播放音频。
- 默认 GPU 可见窗口，1440×900，深→浅→深三次切换。每次核对全部32个选项不透明背景与文字，所有可见项对比度≥4.5，实际最低为深色8.49、浅色10.60。展开声源模式，真实弹出列表未选中行像素为深色 `#282828`、浅色 `#faf9f6`，截图人工核对文字清晰；方向键与回车正确选择清声。
- 最终证据：`output/validation/m10-dropdown/qt-7583b8ee68fb43bda8883cd47d20f224/report.json`，三张 popup PNG与 `cleanup.json`。测试 scratch 回执为 deleted。
- 初次测试夹具误给未初始化数据库路径，随后旧验收脚本的 engineStatus 标记已被当前 engineState 取代；Qt popup 实际为 QQuickWidget，按可见顶层组件捕获。夹具据当前源码修正后通过，初次报告保留。窗口释放补充 DeferredDelete，确保 Qt 子进程退出后清理；此前四处scratch已按归属清理。
- `scripts/check_architecture.py` 仍报五处既有资源摘要差异：app.js、index.html、presets.js、scene.js、floor-contact.mjs。本次CSS及bundle未报错，保留其他文件登记范围。

## 边界

最终 EXE 未重建，已安装版本仍使用其冻结资源。其他配色的真实原生弹出像素、其他电脑/显卡及跨平台未验。本次没有公开发布或部署。
