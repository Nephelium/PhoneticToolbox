# P19 外观设置与图标验证

后续字体下拉与 Everforest 默认调整见[外观 R1 报告](2026-10-03-p19-appearance-r1-report.md)。以下 30 套等计数记录初轮验证，不代表 R1 当前选项数。

状态：**verified，限定 Windows 开发态、独立 Chrome 和实际 Qt 离屏**。WSL 仅资源读取，实体任务栏、系统 DPI、Linux/macOS GUI 和冻结 EXE 未验。本轮未打新 EXE。

## 完成内容

- 原 PhoneticToolbox 加 29 套 Codex 同名适配配色，每套深浅两版；显示模式独立选浅色、深色、跟随系统，切换方案及系统变化同步背景/文字/控件/选中态。模式偏好兼容旧版本。
- 宽窗两栏、最大内容宽 1220 CSS px，左外观与缩放，右字体与预览；窄容器转单栏，保留字体草稿与关闭保护。
- 首次与恢复默认采用宋体、Times New Roman、内置 JetBrains Mono 2.304，IPA 固定 Doulos SIL；旧显式选择保留。缺系统字体时初始化兼容回退且解释原因，不覆盖保存偏好；显式应用缺失字体仍拒绝。
- K2 原图不改。原可见宽 990 / 1254 ≈ 79%，运行时 32 px 图标可见宽 30 px ≈ 94%，相对显示宽度增加约 19%。16/24/32/48/64/128/256 七尺寸与 future-build ICO 同源。不能据此保证未重打的旧 EXE 或实体任务栏已经更新。
- M10 已打开页面同步公共颜色；其算法、数据、录制与图表曲线色义未改。

## 实际命令与结果

| 命令/环境 | 结果 |
| --- | --- |
| `npm --prefix frontend run typecheck` | 通过 |
| `npm --prefix frontend test` | 242 项通过；含非法偏好、配对方案正文/弱文字/强调/状态/选中面板对比度和默认字体兼容 |
| `npm --prefix frontend run build` | 通过；固定 WOFF2 资源进入 dist；保留原大 chunk 警告 |
| `node tests/e2e/p19-appearance.cjs` | 8 组通过，含 60 主题/模式、系统切换、实际字体加载/等宽测量、旧偏好、草稿关闭、缺默认字体受控回退及 15 窗口/缩放组合 |
| `node tests/e2e/fonts.cjs` | 12 组既有字体回归通过，含真实 M02 PNG/SVG、IPA 独立像素参考、账号切换、24 px 图文与 TextGrid |
| 源码 PYTHONPATH 下 `.venv/m09-ui/Scripts/python.exe -m pytest -c tests/pytest.ini desktop/tests/test_p19_icon.py -q` | 2 项通过，七尺寸不裁切、占用、原画比例、ICO 逐条 PNG 解码、源图哈希保持 |
| 同环境 `scripts/verify_p19_qt.py` | 3 类检查通过，60 主题/模式、6 布局、离线 WOFF2/IPA、实际 QIcon、M10 已开 iframe 三次配色/恢复同步 |
| WSL NInfer 既有 `/home/ninfer/ptb-m06-20260927/bin/python` | 4 项静态回读通过：字体哈希、dist 字体一致、许可一致、构建入口可读。无 Linux GUI/font/device 验收 |
| `npm --prefix frontend run ui-data:check`、`git diff --check` | 通过，355 条来源登记与生成文件一致 |
| `python scripts/validate_docs.py` | 检查1279文件；仍有8条现有旧EXE缺失链接，命令未全绿。无本轮新文档链接/编码/语法错误 |

证据位于忽略目录：

- `output/validation/p19/chrome-1791032798420/report.json` 与浅深/窄窗截图。
- `output/validation/fonts/chrome-1791032805842/report.json` 及真实导出。
- `output/validation/p19/qt-11188a7395714080a022e22a23d895b8/report.json` 与实际 Qt 浅深截图。
- `output/validation/p19/icon/` 七张运行时 PNG 与验证 ICO。

## 检查中发现与修正

- 选中底色的强调文字需单独校验，已纳入颜色生成与测试，最低 4.5:1。只保证列明语义对，不扩大为全应用无障碍认证。
- 宋体/Times New Roman 在其他设备可能缺失，补启动兼容回退。用受控缺字体验证该路径，不冒充 Linux 实测。
- 首次浏览器等待使用 `document.fonts.check` 观察到尚未请求的 CSS face，修正验证为实际 `load` 后检查 FontFace/等宽尺寸。
- 首次图标断言把原可见主体当作正方形，独立原图检查得到 990×973，验收改为保留该比例并允许一输出像素的栅格量化，未拉伸原画。
- 早期截图截在 120 ms 按钮过渡中间态，最终截图等待过渡完成并逐张检查。
- 最初 pytest 误用了遗留根目录 coverage 参数，最终使用项目 `tests/pytest.ini`，未安装插件或更改检查配置。
- Qt 离屏日志保留 GLES 上下文告警，DOM、字体、截图和主题桥结果实际通过；没有更改产品渲染参数或系统配置，不声称实体 GPU/3D/DPI 已验。
- 全库文档检查现为8条旧 EXE 缺失链接（README四条、此前成品报告四条），独立归档快照的旧链接另列。不修改历史快照或制造文件绕过。

来源和许可见[来源记录](../references/p19-appearance-sources.md)。无数据库迁移、全局安装、push、部署或公开发布，保留同期 M05/P18/M17 未提交差异。使用[源码工作台](../../scripts/Start-M16-M17-Workbench.ps1)体验本轮修改。
