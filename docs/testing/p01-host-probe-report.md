# P01 桌面宿主与音频原型验证

2026-09-09 更新。任务状态：**verified（Windows 原型与来源风险评审范围）**。井井试用反馈“目前没问题，可以继续”，已按 ADR-013 冻结开发主线并授权 P02。下列实测证据保持原样；历史材料再分发、Mac/Linux 原生和设备仍未验收。

## 交付与范围

- 本地单文件：`output/p01-probe/PhoneticToolbox-P01.exe`。
- 最终文件为 **207,535,078 字节（约 207.5 MB / 197.9 MiB）**；SHA-256：`e297b46d2298e85f0a7249660c7642298b7bdf617510e492327e210168907eeb`。
- 源码与命令：[desktop/experiments/README.md](../../desktop/experiments/README.md)。运行时不需要启动 Node、Vite、Python 命令或云端服务器。
- 页面：真实 WAV 波形、多声道、整数采样选区、选区缩放、试听/暂停/继续/停止、中文/IPA、浅深主题与窄窗口适配。
- 本轮只读载入，不修改音频；合成测试音明确标为测试音，不是科研计算结果。P01 格式限制见试用说明，未实现模块没有伪装成可用入口。

## 已执行的验证

| 项目 | 结果与证据 |
| --- | --- |
| 原始 WAV/选区 | Node 内置测试 10/10：44.1 kHz、声道分离、PCM16/24/32、float32、单帧/空选区、越界、坏文件、非有限样本、奇数大小 RIFF 块 |
| 前端 | `npm.cmd ... run typecheck`、`run build` 均通过，Vue 3.5.42 / Vite 8.2.2 / TypeScript 5.9.3 / vue-tsc 3.3.11 |
| 资源读取边界 | pytest 9/9：正常资产、路径穿越、编码路径、Windows 盘符/反斜杠、UNC、NUL、目录及缺文件 |
| 实际 Qt WebEngine | 源码和单文件内均运行。44.1 kHz、2 声道、88,200 帧载入正确；空选区不播放、暂停位置保持、继续推进、停止复位 |
| 精确片段 | 原始 `[123,22173)` 对应 22,050 帧/0.5 秒；WebAudio OfflineAudioContext 输出长度与原采样率一致，所有样本最大绝对误差为 0。末尾 `[88199,88200)` 保留恰好 1 帧 |
| 字体 | 本地 Doulos SIL name 表为 Version 7.000；内嵌 OFL 与版权提取原文。所测 IPA 字符无缺字，中文与 IPA 实际截图已检查 |
| 窗口与主题 | 1280×800、960×720、640×700 的实际 Qt 页面截图，未出现横向溢出；主题/尺寸改变保持同一选区。较小窗口允许纵向滚动 |
| 实际数字音频输出 | 源码和单文件均通过 WASAPI 默认输出端点回环；在同一稳定窗口检测到声道 1 的 440 Hz 和声道 2 的 660 Hz。当前端点为 ROG DELTA II 蓝牙耳机、48 kHz；未保存混音原始音频 |
| 单文件生命周期 | 在非项目工作目录启动，独立进程运行；首次清理后版本测试追踪 9 个相关进程，退出后均结束、解包目录清理。双实例初测追踪 18 个相关进程，独立解包目录均清理 |
| 最终交付复验 | 中文且含空格的 EXE 路径通过；数字回环通过；实际像素比例约 1.25 / 1.50 / 2.00，三档均保持选区且无横向溢出。200% 双实例运行均通过，18 个记录到的进程全部退出，两个临时目录均清理 |
| 原 v2 保持 | 427 个基线文件无变化；v2 HEAD、index SHA-256、status、原 conda 包元数据摘要均与实施前相同 |

完整 JSON、日志和截图写入忽略的 `output/validation/p01/`。`onefile-clean` / `onefile-dual` 是构建路径修复后的中间验收；`final-*` 是最终交付二进制的复验，具体汇总见同目录 `final-summary.json`，不得混用不同构建的 SHA-256。

缩放复验通过本次子进程的 `QT_SCALE_FACTOR` 在当前 150% 桌面缩放基础上分别乘 5/6 和 4/3，实际 canvas 像素/逻辑宽度比为 1.2497 / 1.5005 / 2.0000；没有更改 Windows 全局显示设置。报告中的 `loaded_seconds` 起点在 Python 导入 Qt 后，不含完整单文件解包时间；整次测试约 15.7–18.8 秒，不能把它当作启动耗时。

## 两个已定位并解决的环境问题

1. **conda 派生 venv 的 ICU 冲突。** 初试 `.venv/p01-pyqt6` 使用原 conda 解释器派生，Qt 6.11.2 实际解析到原环境 `Library/bin/icuuc.dll`，缺少 `ucnv_open` 等 20 个导出符号。诊断证据为 `dll-diagnostics.json`。改为项目内独立 CPython 3.11.14 后，相同 PyQt/Qt 版本加载成功；未改系统、未替换原环境 DLL。
2. **PyInstaller 从工具 PATH 收入了另一份 ICU。** 初次 EXE 构建成功但启动失败，Analysis TOC 指向 Codex 附带 Poppler 的 ICU。`build_probe.py` 将此次构建进程 PATH 限定到探针 Python 和 Windows 系统目录，并审计每个 BINARY/EXTENSION 的来源；修复后意外二进制输入为 0，EXE 实际启动通过。未通过强行忽略导入、禁用沙箱或修改系统 PATH 解决。

附带修正：初次 pytest 误读 v2 的 coverage/发现配置，随后为探针建立独立 `pytest.ini` 并显式指定目录；初次 TypeScript 检查指出 ArrayBuffer/SharedArrayBuffer 类型边界，已用实际 ArrayBuffer 类型修正。回环测试最初按混音中最大幅值窗口取值，容易选中启停瞬态；现采用满足幅度条件的稳定窄带窗口，保留原检测阈值，并记录窗口时间及最大幅值窗口的对照结果。

原 K2 图标存在 libpng 的色彩配置/tRNS 告警，保留原图字节，没有为了隐藏告警改动原素材。当前图标正常显示；正式图标整理属于 P04/P12。

交付文档检查：46 个本轮文件的 UTF-8/无 BOM、JSON、Python 语法、Markdown 本地链接检查通过；150 个来源 ID 唯一，32 项任务保留。完整 `git diff --cached --check` 仍报告复制自 PyInstaller 原始 COPYING 的 3 处行尾空格（第 4、26、27 行）；原文保持字节一致，本轮编写的代码和文档没有空白错误。该告警不被写成“全量 diff 检查通过”。

## 原生依赖与来源风险

| 项目 | 本轮事实 | 仍未证明的部分 |
| --- | --- | --- |
| VTL 2.4 | 所继承 Windows API DLL 初始化/关闭返回 0；API 返回构建日期 Dec 3 2025，常量为 48 kHz、50 管段、19 声道参数、9 声门参数。保留的 API 源 ZIP 含 64 个 C/C++ 源/头文件及 Visual Studio 项目 | 本轮未重建 DLL；源 ZIP 与现有 DLL 的可复现对应、Mac/Linux 构建和设备尚未验收；今日开发分支不能替代原 2.4 来源 |
| REAPER | 继承 EXE 成功运行并输出 393 个有声估计；官方仓库提供 CMake 源码，Apache-2.0 | 440 Hz 纯正弦探针中位估计为约 44.012 Hz，出现子谐波选择。只能确认原生入口可运行；P03 需用合适语音/谐波结构样例独立调查，未修改算法或扩大科学容差。现有二进制对应 commit 仍 unknown |
| IRAPT / WM-PC | 已登记 GPL-3.0 / MPL-2.0 来源及本地证据 | 文件级移植、修改义务和最终发行组合仍待闭合 |
| 载瓦语材料 / VoiceSauce 历史材料 | 已有论文/仓库与移植线索 | 未取得覆盖对应代码/录音/数据的完整再分发证据，不能把“可公开读取”当成发行许可 |
| 字体与探针依赖 | Doulos SIL 7.000 已确认；主来源表现有 150 条记录，20 个探针 Python 包、70 条 npm 锁定项均有元数据 | npm 可选平台项不等于实际安装；Qt/Chromium 的完整原生第三方清单及发行策略还需审查 |

原始证据为 `native/native-report.json`、[来源注册表](../../third_party/source-registry.json) 与 [P01 依赖清单](../../third_party/p01-dependency-inventory.json)。

## 宿主比较与当前决定

| 候选 | 本轮证据 | 判断 |
| --- | --- | --- |
| PyQt6 + Qt WebEngine | 独立运行时、前端交互、单文件、数字音频输出及退出均实测；PyQt6 包目录约 575 MB（含 Qt，不等于 EXE 大小） | 保留为 Windows P01 的技术基准，当前无需重做宿主。PyQt6 与 WebEngine 绑定是 GPLv3/商业双许可，不能把 Qt 的 LGPL 自动套到绑定上 |
| PySide6 6.11.2 | 单独环境实际安装并运行到页面、IPA 字体与设备枚举；目录约 663 MB。当前共享自动探针的 Qt 鼠标点击未触发样例加载，两次未通过完整交互自测 | 本轮未冻结、未打包；失败定位在候选/测试适配边界，不能据此宣称 PySide6 本身不可用，也不能当作已经通过的替代方案 |
| Electron | 官方文档确认独立应用打包流程；未在本机建立原型 | 需要额外维护 Electron 与 Python 计算进程的生命周期，当前没有已测优势支持换栈 |
| Tauri | 官方文档列出 Rust 和各平台 WebView/构建依赖；未安装 Rust 或建立原型 | 增加当前计划之外的构建和 WebView 差异；当前保留比较记录，不无证据切换 |

依据：[PyQt6 发行元数据](https://pypi.org/project/PyQt6/6.11.0/)、[PyQt6-WebEngine](https://pypi.org/project/PyQt6-WebEngine/6.11.0/)、[Qt for Python 许可清单](https://doc.qt.io/qtforpython-6/licenses.html)、[Electron 打包](https://www.electronjs.org/docs/latest/tutorial/application-distribution)、[Tauri 前置条件](https://v2.tauri.app/start/prerequisites/)。包目录大小是当前安装实测，不是跨方案内存或最终发行体积的公平基准。

## 保留的验收边界与下一项依赖

- **不是已完成的 v3：** 未迁移科研算法、未建立 P03 数值黄金基线、未做账号/配额/多人服务或 P12 正式安装包。
- **音频证据：** 数字渲染端点回环已测；人耳听感、模拟输出响度、蓝牙延迟、麦克风/相机、实验刺激时序仍未测。页面普通文件选择框与长/多声道真实语料需要试用反馈和后续验收。
- **平台证据：** 本机 Windows 是开发机，不能冒充干净 Windows 账户或无运行库机器；Mac/Linux 原生构建与设备尚未做，WSL 不替代这些证据。
- **冻结门槛更新：** 用户试用已通过，P01 技术原型与风险评审完成，授权 P02。完整发行许可与历史材料处理仍未解决，按 ADR-013 继续阻断对应发行物；P02 不携带这些材料，后续全面模块移植仍须遵守科学基线与来源要求。
- **未执行的外部动作：** 没有推送、发布、部署、向作者发信、修改全局环境、删除旧目录或改动现有用户数据。
