# P04 统一工作台试用报告

2026-09-09。状态 **in_progress**：下列工程验证完成，首页与公共组件视觉等待井井审阅。依据 [P04 计划](../plans/2026-09-09-p04-workbench.md) 和总计划 G3；本报告不代替用户的视觉验收。

## 可试用内容

U2 浅/深/跟随系统主题、K2 品牌图、三个分组的 15 个入口、侧栏搜索/折叠、首页与独立标签。首次只有首页；最近使用来自真实打开记录。各模块明确显示“待接入”。

共同预览支持单个不超过 64 MB 的 PCM/FLOAT WAV、原采样率时间轴、独立声道波形、选区、缩放/平移与单声道试听。EGG 要求双声道；试听声道由用户选择，不根据文件名猜测。1–4.wav 是左 EGG/右音频；牧歌.wav 是左音频/右 EGG。此顺序沿用 P03 的用户确认，不改写音频。

80 个原参数键提供完整抽屉与搜索。参数草稿按标签独立；关闭时保存/放弃/取消，保存的参数可重开恢复。文件与选区仅在标签存续期间保留，关闭后释放；本机持久化不含私人音频或文件路径。方法与引用由统一来源登记生成，不把旧实现参考宣称为当前已经运行的方法。

浏览器与 Qt 共用同一前端构建。Qt 试用直接加载自定义 scheme 的本地静态文件，无额外 HTTP 服务；浏览器开发使用统一 Vite 入口。启动步骤见 [开发说明](../development.md)。

## 实际验证

平台为 Windows，沿用 P02 项目隔离环境 `.venv/v3-dev` 和既有 Node/npm；未安装新依赖。

| 命令 | 结果 |
| --- | --- |
| `npm --prefix frontend run typecheck` | 通过 |
| `npm --prefix frontend run test` | 8 项通过：原契约 2 项、WAV/模块/选区 5 项、主题对比度 1 项 |
| `npm --prefix frontend run ui-data:check` | 80 参数、283 来源，无生成漂移 |
| `npm --prefix frontend run build` | 通过，41 个模块 |
| `.venv/v3-dev/Scripts/python.exe scripts/check_architecture.py` | 无错误，包含新增前端资产登记检查 |
| `.venv/v3-dev/Scripts/python.exe -m pytest -c tests/pytest.ini tests/architecture -q` | 4 项通过 |
| `.venv/v3-dev/Scripts/python.exe scripts/validate_docs.py` | 145 个文件、283 来源、32 任务，无现行文档错误；原样保留的历史快照失效链接另报 |
| `git diff --check` | 通过 |
| `.venv/v3-dev/Scripts/python.exe desktop/experiments/p04_host.py --self-test output/validation/p04/qt-final` | 通过，15 入口、首页、真实桌面标识与无横向溢出 |

真实浏览器通过界面完成以下定向检查，未通过脚本注入应用状态：

- 公开双声道 fixture 为 44,100 Hz、35,280 帧、0.8 s，两轨来自实际样本；0.1–0.5 s 选区、主题和标签切换保留。
- 参数改为 1/80，取消关闭保留；保存关闭后重开恢复参数，音频已释放。测试结束恢复 80/80。
- 实际点击播放/暂停/停止；同源两个浏览器标签中后播放者暂停前者。长测试音频由公开 fixture 重复生成。此项不等于硬件回环验证。
- 文件选择器打开 WAV；损坏 RIFF 显示错误并保留此前有效文件；EGG 搜索、侧栏折叠、4 倍缩放和平移可操作。
- 80 个参数复选框在窄窗口可访问；来源筛选、IPA 字体加载及 Doulos OFL 全文可见。
- 标签 Home 键切回首页并移动焦点；字体许可弹窗 Esc 关闭后焦点回到“关于”。初次检查发现关闭前焦点恢复被模态状态阻止，已改为关闭原生 dialog 并在卸载后恢复焦点，复测通过。

布局证据保存在本机忽略目录 `output/validation/p04`：首页两主题各检查 1280×800、1440×900、1920×1080；工作台检查 1280×800、1440×900、390×844，均无横向溢出。最终三种工作台尺寸的播放器底边分别为 769.33、869.33、813.33 px，均在可见高度内。1280 窗口的播放栏初版需要向下滚动，已移到工作区固定底部。

最终截图：`home-light-final.png`、`workbench-dark-1280.png`、`workbench-dark-1440.png`、`workbench-dark-390.png`。布局数值见 `browser-layouts.json` 和 `browser-final-layouts.json`。截图含真实测试产生的最近使用，不冒充首次空状态。

Qt 自测在当前 Windows 屏幕 DPR 1.5 下，页面缩放 100%、125%、150%、200% 均无横向溢出，IPA 抽检无缺字。证据 `qt-final/qt-summary.json` 与对应 PNG。这是 WebEngine 页面缩放检查，未修改系统缩放，也不是多显示器/跨平台 DPI 验收。Qt 对沿用 K2 PNG 发出 iCCP/tRNS 元数据警告，图片可显示；保留原资产字节，未忽略失败检查。

## 来源与保留检查

K2 原图和已核验的 Doulos SIL 7.000/OFL 原文件复用，未下载新字体或图标库。前端资产清单登记四项及校验和；来源表新增项目自行生成的公开测试信号记录，现有字体/K2 项补充 P04 使用位置。历史许可 unknown/发行阻断仍保留，不据此宣称所有资产可再分发。

通过只读 `desktop.experiments.capture_context.capture()` 与 P03 收尾证据比较，v2 HEAD、index、工作状态、原环境位置和包元数据五项一致；427 个继承基线文件无变化。证据为 `context-after.json`、`preservation.json`，仅写入 v3 忽略目录。

## 剩余边界与下一步

- P04 待井井审阅首页、主题、密度、播放器与参数操作；反馈通过前保持 in_progress。
- 15 模块的科研算法、实际后台任务和进度、摄像头、批处理、结果导出、账号/存储未接入。任务面板已有状态/错误/取消/重试视图接口，当前显示真实空态；实际生命周期由 P06 验证。
- WAV 解码和包络用于前端预览，不能作为 P03 科学等价性证据；FLOAT64 预览以 Float32 保存，不用于科研结果输出。
- 多窗口停播仅验证同源浏览器标签。Qt 跨进程、浏览器与 Qt 之间的协调、Qt 原生选文件和设备音频回环尚未在 P04 验收。
- Windows Qt 静态页面启动不等于打包通过；本轮未构建发行 EXE，Mac/Linux、干净机器、原生设备与发行许可仍待对应任务。

完成视觉反馈后，按总计划进入后续基础能力任务；本轮未提前展开全面模块迁移。
