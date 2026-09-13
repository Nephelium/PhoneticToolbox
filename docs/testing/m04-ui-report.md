# M04-D LPC 页面验收

2026-09-13，状态 **verified / 限定 Windows 开发态独立 Chrome、真实本机任务服务与科学子进程**。完整M04仍in_progress，下一项E的托管网页联合验证与A01–A20逐项收口。未启动Qt、生成EXE、执行DDL或push。

## 实现与行为

新增LPC页面并注册到既有AppShell，复用WaveformViewport、TextGridTimeline、ScientificPlot、AudioTransport、TaskPanel及全局字体。左侧选择音频/标注，主区切换波形/频谱，下方分行组织时间、参数、试听和保存。小窗使用工作台的普通滚轮滚动，频谱和波形用Ctrl＋滚轮缩放。

时间范围与频率视图独立。Shift框选、精确时间输入和清除选区均可用；清除后采用可见波形时间窗，显示切换不重新计算。参数使用文本草稿，空值、过短和超过48,000样本的ROI在提交前明确拒绝。多声道分析均值、半开样本区间、1024点原算法由既有后端执行，前端只显示结果。

TextGrid同名自动关联、首层/循环层与比例时间轨接入。波形试听原始所选声道，频谱试听任务保存的单声道片段。历史结果带原文件/参数/时间，编辑变化显示旧结果提示。字体预检失败不丢选区；异步输入/标注/结果/目录保存按归属检查，迟到创建只取消该任务。刷新后已移除或变化的源文件不能继续使用旧预览。

公共ScientificPlot按刻度字符长度预留纵轴边距，并将横轴单位放在刻度下方。实测24px动态小数刻度不再裁剪负号，Hz与末端数值分开。该显示修正不改变曲线数组、坐标范围或导出PNG。

## 验证证据

| 范围 | 实际结果 |
| --- | --- |
| 前端 | 69项node测试、vue-tsc及Vite构建通过；新增参数/半开区间/状态签名与服务器adapter账号、CSRF、SHA契约检查 |
| LPC实际Chrome | 24组通过：真实WAV/TextGrid、LPC子进程、样本边界、动态范围、试听、取消/迟到创建、失败重试、历史恢复、草稿、目录保存及浏览器三文件下载 |
| 连续操作 | 旧文件错误/结果回读/目录选择迟到不串入新选择；缺字体不提交；刷新失效文件清除预览 |
| 布局/字体 | 浅色1440、深色1440/1000/390宽度截图，无横向裁切；390小窗普通滚轮实际滚动工作台；24px刻度边界与Hz不重叠 |
| 共享模块回归 | EGG总览定位/末尾/拖动/取消，M01与M02默认选区手势专项复验；未把LPC的Shift门槛加到其他模块 |
| Python定向 | M04契约与已有本机保存保护13项通过 |
| 文档/生成 | UI来源数据与TypeScript契约一致性、文档编码/链接及架构检查通过 |

主要证据为忽略目录 `output/validation/m04-ui/d7b997f1587d46c1932a80f0831a39a2/`，包括 `report.json`、波形/频谱/窄窗截图、`dynamic-font-24.png`、实际下载的PNG/WAV/JSON。PNG回读2400×1350，JSON回读1024点。三文件命名含IPA，重复保存不增加文件，0.1–0.2秒对应4800–9600半开样本。

最终公共图表修改后的5组共享回归证据：`output/validation/m03-ui/chrome-a4f40a4759794418a8a5da5c19485638/report.json`。文档检查616文件、333来源、32任务无新增错误；冻结的历史快照缺失链接仍单列保留。

连续操作使用真实RPC结果加浏览器延迟，以及受控缺字体/旧文件失败/文件列表移除响应。计算和保存成功来自实际服务与MKL子进程；这些注入不代表生产故障或自然到期。浏览器下载通过本机桥接取真实资产，不能当作托管登录页面验收。

## 命令

```powershell
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run build
npm --prefix frontend run contracts:check
npm --prefix frontend run ui-data:check
node tests/e2e/m04.cjs
node tests/e2e/m03-function-review.cjs
$env:PYTHONPATH = ((Join-Path (Get-Location) 'backend/src'),(Join-Path (Get-Location) 'desktop/src') -join ';')
& .venv/m09-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini backend/tests/test_m04_contract.py desktop/tests/test_m03_save_names.py -q
& .venv/m09-ui/Scripts/python.exe -X utf8 scripts/validate_docs.py
& .venv/m09-ui/Scripts/python.exe -X utf8 scripts/check_architecture.py
```

首轮滚动检查误选页面section，改为实际`main.pane-workspace`后通过。延迟测试最初在路由响应完成前移除handler，引发测试脚本`Route is already handled`，改为等待响应完成后再解除拦截；任务拥有的进程已退出。没有通过跳过断言处理。截图发现大字号裁剪后修复公共轴布局并重新验证。

既有M03脚本的Vite扫描仍提示公共vendor中`three`未解析，随后实际页面与回归完成；本轮M04脚本将扫描入口限定到自己的测试页。Python保留两项Starlette/anyio弃用提示，未安装依赖或屏蔽警告。

## 剩余边界

使用说明已补至[操作手册](../manual/lpc-spectrum.md)，应用内帮助同步。现有Makhoul方法引用与V2/NumPy/SciPy/Matplotlib来源登记沿用，未引入外部代码或新依赖，未新增来源查验结论。

本轮未改原数值核心、V2、科学环境或五份暂停的EXE草稿。托管Chrome的上传/登录/任务/下载联合流程、自然录音页面及完整A01–A20证据汇总留E；真实声卡、多屏、其他平台和生产负载尚未验收。输入还受公共预览3200万采样值上限，不将接口800万帧上限扩大为所有声道组合都可页面预览。
