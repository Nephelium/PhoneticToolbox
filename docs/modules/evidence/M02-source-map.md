# M02 说明书、源码与新入口映射

2026-09-11。旧来源为本工程继承的 `phonetic_toolbox/gui/widgets/parameter_display_widget.py`、`acoustic_widget.py` 及 v2 说明书 2.2 节，相邻 v2 保持只读。说明书中的 CSV 描述与当前 XLSX/SQLite 实现差异保留在覆盖门，不承诺任意 CSV 已接入。

| 功能组 | 旧操作/行为 | v3 入口 | 正常与边界证据 |
| --- | --- | --- | --- |
| M02-F01 | WAV/参数目录、刷新、搜索、同名配对，SQLite 优先 | ParameterDisplayPage / FileProvider / assets parameters / parameter_preview | test_m02_display、m02.test、verify_m02_m09_local、Qt 与 Chrome 的实际 XLSX 读取；损坏、公式、错误 SQLite、重复关联测试 |
| M02-F02 | 参数搜索，多选，独立 reaper/correction | 右侧参数栏 / visibleParameters | 双开关不丢选择，批量分配测试与 Chrome 实际操作 |
| M02-F03 | 波形、参数、文字层、语谱图 | WaveformViewport / ParameterFigure | 原时间、缺失断线、极值抽稀、文字区间单测；Qt/Chrome 实际图面，新增图高与字体像素断言 |
| M02-F04 | 时窗、位置、缩放、选区 | 共享 wave 状态、波形导航、精确时间输入 | 共享波形回归与 m02 原时间单测；图窗间同一状态，无二次估计 |
| M02-F05 | 试听、保存图像、说明、帮助、关闭 | 公共 AudioTransport、图窗 SVG、主帮助与来源 | 原生保存对话框/真实 SVG 回读，网页下载，现有波形与播放回归 |
| M02-U01 | 井井新增：默认同一绘图区叠加，可分多窗，批量分配 | assignParameters / removeGroup / overlayPlot | 两窗、第三窗、合并再分配；空窗 ≥300 px，双曲线图433/410 px，字体实际尺寸检查；每窗恰好一个 shared-plot-area、多条曲线共享刻度断言 |

同图语义纠正：此前误做独立纵轴子图，井井指出后已撤回该设计。重新阅读全文及图2-9（img_kzzc61x82.jpg）、图2-11（img_z1dy8u7wp.jpg），并核对旧 `_plot`：数值参数同区叠加，图例区分；可见原帧各列平均绝对值最小值大于0且最大/最小超过50时，均值大于100者走右轴。v3 overlayPlot 按该规则分类，共用坐标区，不归一化，统计不使用抽稀点。文字标注同步波形和参数图。普通滚轮缩放时间轴、左键拖动平移。Qt/Chrome实际叠加截图及导出回读见联合报告的最新复验，旧堆叠图截图不作为同图语义通过证据。

交互/导出适配差异：旧 Ctrl 多选与连续选择由复选框及全选可见参数实现；勾选后显式批量分配，防止仅选择候选项就移动曲线。列宽通过滑块调整。旧整幅 PNG 保存现为当前参数图 SVG（含图例/文字，不含上方波形）；没有声称 PNG 或整个波形联合图导出已实现。图窗属于统一宿主，不创建独立 HTTP 服务。

读取移植复用已经受限的 M01 legacy_parameters；本模块无需新科学核心。来源新增使用位置由统一登记生成至模块致谢。总证据见 `docs/testing/m02-m09-report.md`。
