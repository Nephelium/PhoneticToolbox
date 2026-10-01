# P04-RESIZE / M02-DEFAULT 验收记录

2026-09-29。状态：**verified，限定 Windows Chrome 与实际 Qt 开发态公共布局和下列回归**。WSL 仅静态文件读取与哈希核对，Linux GUI、触屏、macOS、服务器部署和冻结 EXE 不在已验范围。完整科研模块的原有验收状态不因此扩大。

## 实际行为

- 参数估计文件栏/输出参数栏默认从 166/174 拓宽到 220/250 逻辑像素。
- 统一控制器接入导航栏、M01、M02、M04、M05、M06、M07、M09、M10、M11、M12、M13 的现有侧栏。M03/M08/M14/M15 无独立侧栏，保留原布局。
- 拖动内容区边界调整宽度，放开自动保存。按模块和账号隔离，同一账号跨项目共用布局，使用现有本机 ProjectStore，不新增数据库或跨设备同步。窄窗临时收缩不覆盖偏好，恢复宽窗后恢复保存值。M13 按同期独立授权保留右栏和横向滚动，已确认控件可达。
- 分隔线支持 Tab、左右键、Shift＋左右键、Home/End；Escape 取消本次拖动。边界限制侧栏与中央区最小宽度。存储失败保留当前布局并提示。界面缩放使用逻辑像素；导航折叠不误存窄宽度。
- M02 移除两个宽度滑块，首次读表不勾选、不自动绘制曲线；显式分配才绘图。显式保存的曲线配置继续恢复，保存的空配置保持为空。仅加载空表不会产生虚假的未保存状态。
- M02、M10 和公共占位页面移除剩余重复页首名称、说明及关闭按钮。保存/来源等原操作保留，名称与关闭操作在统一标签栏。科学图题、图例、坐标与单位保留。
- 设置、使用说明成为唯一工作台标签。切换保留字体编辑，关闭时支持应用/放弃/取消，应用失败保留编辑和标签。主题与缩放仍即时保存。

新增来源：无第三方代码、库、图片或字体。复用 Vue、现有 ProjectStore、公共主题及原字体资源。科研数组、算法、任务协议和文件语义未由本任务修改。

## 实际命令与结果

| 命令 | 实际结果 |
| --- | --- |
| `npm --prefix frontend run typecheck` | 通过 |
| `npm --prefix frontend run test -- --run` | 最终共享工作区 183 项通过，0 失败/跳过；新增 3 项栏宽边界、临时收缩与账号/模块存储键测试 |
| `npm --prefix frontend run build` | 通过，保留现有大分包警告，未更改阈值 |
| `node tests/e2e/p04-resize.cjs` | 9 组公共验收通过，另含 10 个模块 × 4 组视口/缩放/主题，共 40 组几何检查；最终报告 errors 为空 |
| `node tests/e2e/m02-png.cjs` | 3 组通过，5 份真实 PNG 独立解码、CRC、300 dpi、背景/图面与对齐检查通过 |
| `node tests/e2e/fonts.cjs` | 12 组通过，包括真实设置应用/失败/恢复、24 px 图形、PNG/SVG、IPA 与独立 Canvas 对照及账号字体隔离 |
| `.venv/m09-ui/Scripts/python.exe scripts/verify_p04_resize_qt.py` | 实际 Qt 4 组通过，合成 WAV/XLSX 通过真实本地 FileProvider 读取；QTest 原生指针双边界、重载恢复、M02 空图/显式绘制、辅助标签、110% 缩放 |
| `wsl.exe -d NInfer --exec perl /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/scripts/verify_p04_resize_linux.pl` | WSL Linux 读取生产构建 48 个 JS/CSS/HTML/字体文件，确认共享控制器和 M10 页首移除；Windows 再次逐文件 SHA-256 核对一致 |
| `node --check tests/e2e/m02-m09.cjs` / Python `py_compile` | 已同步旧联合/PNG/字体探针的显式参数选择与设置标签入口，语法通过；未复跑其历史全链路 |
| `git diff --check` | 通过 |

Qt 使用 `PYTHONPATH=desktop/src;backend/src;packages/phonetic_core/src`，自身临时 profile 和合成输入，`WA_DontShowOnScreen` 隐藏原生窗口。测试未显示弹窗窗口、未依赖现存任务库、未执行 DDL，结束时关闭自身宿主和服务。M10 Chrome 检查使用真实 wrapper/iframe 及公共控制器，原生引擎明确不可用，只证明布局接线，不能替代录制或设备验收。

## 证据与范围

- `output/validation/p04-resize-unit-final.log`：最终 183 项结果；初轮 181 项，另外两个为共享工作区同期其他任务新增测试。
- `output/validation/p04-resize-build-final.log`：最终生产构建与分包警告。
- `output/validation/p04-resize/1790691442350`：最终 Chrome 报告和实际截图。覆盖 M01/M02 拖动、保存重载、键盘极限/取消、70/100/150% 比例、窄窗恢复、其他布局真实指针、设置关闭保护、导航折叠、存储失败及 M10 三列。
- `output/validation/p04-resize/qt-bd97cefc697540e8a224c6835d702e27`：最终实际 Qt 报告与 M01/M02 截图；110% 时鼠标移动 33 像素，栏宽从 220 到 250 逻辑像素。
- `output/validation/p04-resize/linux-static-1790691464`：Linux 文件清单及 `windows-match.json`。
- `output/validation/m02-png/chrome-1790690553743`、`output/validation/fonts/chrome-1790690647217`：PNG/字体回归。使用合成输入，不读取用户研究数据。

已查看浅色 M01、深色 M02 空图、设置标签、窄窗/缩放与实际 Qt 截图。宽度拖动仅改变显示，M02 绘制仍使用原时间/数值和缺失值规则。

## 调试与未测项

- 初轮发现导航 CSS 变量被模块继承，导致单侧栏页面初始宽度和可拖动识别错误。控制器现在在每个根节点显式设置自己的变量，随后各模块边界和矩阵通过。
- Qt 探针初轮在 reload 后过早继续，已改为等待实际 loadFinished。缩放探针的错误引号曾使放大按钮未被点击，实测页面仍为 100%；修正测试并等待 110% 后通过，未扩大容差或修改产品来迁就测试。失败证据仍保留。
- 最终 Chrome 报告保存前，各页面及 iframe 的异常捕获为空；关闭浏览器后 Vite 日志仍记录一次 `ResizeObserver loop completed with undelivered notifications`。该退出阶段警告未由本轮定位至具体观察器，不能将本轮结果表述为所有时序完全无警告。Qt 保留原图片的 libpng 元数据警告，构建保留大分包警告。
- 未执行 Linux GUI、实际服务器/远程计算、触屏/多屏 DPI 设备矩阵或生产负载/资源预算测量。此任务没有新增计算任务。静态 WSL 文件检查不代表 Linux 交互验收。
- 同期 EGG/LPC/M13/桌面图标/本地预览任务的改动完整保留，未并入本任务功能声明。本轮未提交混合改动、未 push、未修改 V2/旧包、未生成 EXE、未公开发布。

使用入口为当前源码工作台/`frontend/dist`，已有冻结 EXE 不会自动获得本轮更新。操作见[设置与侧栏](../manual/settings.md)、[参数显示](../manual/parameter-display.md)、[参数估计](../manual/parameter-estimation.md)。
