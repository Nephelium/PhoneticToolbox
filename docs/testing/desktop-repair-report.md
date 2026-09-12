# M01/M02/M09 Windows 单文件联合修复

2026-09-12。状态：**verified，限定以下 Windows 本机合成数据、Qt 页面及单文件流程**。这次验收不扩大为全模块、全平台或正式发行。井井已明确允许新建专用本地任务库，未修改现存数据库结构、v2、网页服务器或原 R5 EXE。

## 交付与使用

- 入口：`dist/research-repair/PhoneticToolbox-v3-Research-Fix1.exe`，326402762 字节。
- SHA256：`da3a5fe82840e6cb4705e1517cf6a5b9919a61da866cbe78695ca3724122341e`。
- 双击进入统一研究工作台。本机回环服务和任务执行器由应用自动持有，无需登录或手动启动服务器。
- 首次在 `%LOCALAPPDATA%/PhoneticToolbox/v3/research-v1` 排他新建 `jobs.sqlite3`、`files` 缓存和 ready 标记。使用既有 002/005 表结构。再次启动只校验，已有非本应用目录或不完整初始化拒绝，不覆盖其中的数据。
- 原 `dist/m10-recording/repacked/PhoneticToolbox-v3-M10-R5.exe` 保留。本次应使用 Research-Fix1 验证分析功能，旧录制专用 EXE 不会自动更新。

## 根因与修复

| 用户现象 | 已核实原因与本轮处理 |
| --- | --- |
| 分析、语谱图重建按钮不可用 | 原录制入口没有传入任务库及缓存。新统一入口提供独立持久本机工作区。 |
| 参数/语谱图读失败，时不时另弹白窗口 | 冻结 EXE 收到 Python `-m` 子进程命令时重新走主窗口入口。统一白名单 `--ptb-worker` 调度覆盖参数、语谱图、持久执行、计算、切分、旧格式及导出入口。未知参数非零退出，不回退 GUI。 |
| 截图含当前窗口 | 仅 hide/processEvents 可能捕获上一帧。新适配等待 Qt 事件循环和 Windows DwmFlush，再截取并恢复原窗口状态。 |
| TextGrid 看起来是等宽卡片 | 新时间轨按真实 xmin/xmax 定位，空标签占据实际区间，短段不扩宽，与波形同宽并同步缩放/平移。点击仍选择原始时间段，完整标签列表收进折叠区。 |
| 参数表失败原因不清楚 | 区分格式、读取子进程、超时和预算，不把执行环境失败统一描述成原表无效。Windows 安全文件读取锁未放宽。 |
| 深色主题下拉白底浅字 | 纳入此前公共 option/optgroup 背景及文本颜色修复，沿用公共令牌和原生键盘操作，见 [主题验收](p04-theme-popup-fix.md)。 |

算法、默认值、时间单位、帧网格、原始数据及资源上限没有因本次入口修复更改。M02 保留同一绘图区叠加曲线及多窗批量分配。

## 实际命令与结果

运行目录为 v3 根目录，Windows 11，项目独立 Python 3.11.14 / `.venv/m09-ui`，PyInstaller 6.22.2。Python 定向测试设置 `PYTHONPATH=backend/src;desktop/src`。

1. `.venv/m09-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini backend/tests desktop/tests tests/contracts tests/security packages/phonetic_core/tests tests/parity/test_spec2wav.py tests/parity/test_parameter_estimation.py -q`：**414 passed**。既有 Starlette/AnyIO 弃用警告两项。
2. 收尾再次运行 `backend/tests/test_local_workspace.py backend/tests/test_frozen_workers.py backend/tests/test_m02_display.py`：**23 passed**，这是上述测试的定向复跑，不能叠加计算成独立总数。验证九个白名单入口、未知模块拒绝、新库及再次打开不执行 DDL、参数读取与边界。
3. `npm --prefix frontend run typecheck`、`npm --prefix frontend test`、`npm --prefix frontend run build`：通过，**35 tests passed**。时间裁剪、短段比例保留原值，主题和其他模块已有回归均通过。
4. `scripts/check_architecture.py`：`errors: []`；`git diff --check`：通过，Git 仅提示现有 CRLF 转换设置。
5. `scripts/research_entry.py --local-root <新建测试目录>/state --verify-repair <新建测试目录>/results`：开发态 **14 个实际页面步骤通过**。参数估计、Praat 预览、SQLite 三曲线显示、M09 重建和四文件导出完成；WAV 样本数/采样率与 JSON 回读一致，所有原始输入哈希不变。
6. `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/build_research_repair.py`：单文件构建成功。只更新本任务独立候选，未覆盖 R5。
7. `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/run_research_repair_check.py`：真实冻结 EXE **14 步通过**，两项持久科学任务成功，完整 WAV/PNG/JSON 导出回读；截屏隐藏受控测试通过；可见工作台最大数量 **1**，退出后自有子进程 **0**；三种未知/缺失入口均返回 **2**。
8. 对同一已初始化测试 state 再运行 `scripts/run_research_repair_check.py --state <第一轮state>`：**14 步再次通过**，之前两项任务保留，加上本轮两项共 **4 项成功任务**；窗口数量、退出清理及未知入口拒绝再次通过。
9. `scripts/verify_textgrid_timeline.py <开发态合成inputs>`：**7 个实际 Qt 检查通过**。0.1/0.2/0.7 秒段按 10%/20%/70% 显示，轨道与波形边界差小于 1 个 CSS 像素；点击选区精确到 0.1–0.3 秒，2 倍缩放后中段占 40%，尾段裁剪保留原 xmax=1。截图人工检查了 IPA 标签、空区间和选区对齐。

## 原始证据目录

均位于忽略的 `output/validation/desktop-repair/`，不包含用户研究语料：

- `dev-3d1158c1a81d41dc9328e3e625203173`：开发态完整流程。
- `frozen-8bd29bf0e66e4b6aaabd25b8ca8a3c3e`：首次冻结 EXE 流程、进程观察及新库。
- `frozen-fcf70a0f4c7e40f59ab61fde672e12cc`：复用前轮 state 的重启验收。
- `timeline-d02f44e2157b461cb2c26e1859b163a7`：TextGrid 实际比例、点击和缩放截图。

开发中的失败记录保留：测试 fixture 的 SQLite 连接未关闭导致安全读锁拒绝；完整目录比较误把正常新增导出当成原文件改变；一处测试 JavaScript 选择器转义错误导致等候图片超时。均修正测试本身，未削弱产品的文件锁、结果校验或计算预算。截图测试另出现一次前景竞争，改为测试自有置顶纯色背景后复验；只保存 4×4 纯色检查图，不保存个人桌面。

## 来源与限制

复用现有 Qt、Praat/Parselmouth、REAPER、科学核心和已登记依赖，无新算法、外部素材或第三方运行依赖。所有权观察用 Python 标准库和现有 Windows API，没有安装额外包。打包打印部分可选模块未找到的警告，本报告只依据上述实际执行路径通过，不声称所有可选能力可用。

未验证：井井截图中每一份实际历史参数文件、多屏/混合 DPI 截屏、真实操作系统主题动态切换、长时间压力、安装版和 macOS/Linux。M10/R5 科学及录制行为未重新验收，原有 R5 未检验边界保留。其他模块不继续迁移；公网服务未部署，未 push 或发布。
