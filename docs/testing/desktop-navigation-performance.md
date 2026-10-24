# 说明书、音标表与桌面窗口响应排查

2026-10-24，任务 `DESKTOP-NAVIGATION-PERFORMANCE`，Windows 本机源码及独立生产前端验证。最终包须重新冻结后验收，本页当前证据不能替代新 EXE。

## 方法与输入身份

排查覆盖国际音标表 Plus 的 IPA、extIPA、VoQS，说明书首次打开、同章定位、跨章、滚动、隐藏恢复和关闭重开，以及壁纸、玻璃、字体和窗口尺寸。科学内容、音标字形、计算默认值、章节和媒体语义保持。

- 基线前端为只读冻结 `output/build-PhoneticToolbox-305-20261023-204825-9a75ba/snapshot/frontend/dist`。当前源码生产前端独立构建到 `output/validation/m17-manual-performance-20261024/source-after/dist`，未覆盖工作台 `frontend/dist`。
- Chrome 使用本任务独立配置和生产构建，记录 Performance 主线程/脚本/布局/样式指标、长任务、资源请求及 CPU profile。壁纸使用自生成棋盘格，记录共享磨砂位图是否稳定。
- Qt 使用当前桌面源码、独立测试 profile、可见窗口、默认 GPU，仅添加静音参数。实际为 Qt WebEngine 6.11.2 / Chromium 140、Intel Graphics / ANGLE Direct3D11、150% DPI。原生取图只使用自有 `QWidget.grab()`。
- `elapsed` 包含脚本等待和渲染调度，不能当作纯点击耗时。主线程 CPU 指标、长任务和资源请求分别报告。自然复现与只用于定位的替换探针分开。

## 热点一：说明书播放状态传播

`active` 改变时，原 `playbackAllowed` 通过 `ManualDocument` 和所有递归 `ManualNode` 传播，整章文字、表格节点也随之更新。M01 当前正文约 2595 个 DOM 元素。改为由文档提供只读播放许可，媒体组件订阅，保留独立媒体的显式限制、暂停和重新激活语义。

同一套有界操作的 headed Chrome 生产构建对照：

| 6 次首页/说明书往返 | 修复前主线程 CPU | 修复后主线程 CPU | 长任务 |
| --- | ---: | ---: | --- |
| 普通材质 | 1.650 s | 0.508 s | 修后 0 |
| 棋盘壁纸与玻璃 | 1.898 s | 0.487 s | 18 个 59–118 ms → 0 |

玻璃场景该项主线程时间约减少 74%。这组结果只代表本机当前操作。首次打开大章节仍整章渲染，跨章媒体按当前挂载范围加载，没有引入全文预加载或删减内容。

证据：`output/validation/m17-manual-performance-20261024/probe-baseline/`、`probe-after/`，均保留分场景报告、CPU profile 和截图。3 秒静置没有新请求/DOM 变更/布局，连续滚动也未出现与标签切换同等级的重复正文更新。共享玻璃位图在操作和静置中保持稳定，正常主题/尺寸改变仍按既有规则重新生成。

## 热点二：宿主递归扫描

旧前端与修后前端在相同当前桌面宿主中都出现 Qt 回调延迟。异步 CDP 排除嵌套事件循环这一初始怀疑后，连续 30、60、90 秒的 Python 线程快照均指向 `FirstMaximizePresentation.attach()` 的 `view.findChildren(QWidget)`。当时渲染器 CPU 时间远小于界面等待，资源后台线程处于等待任务状态。

诊断时仅把发现限制为直接子控件并复用存活结果。实际直接子控件为 `QLabel` 和 `QQuickWidget`，没有关闭 GPU 或取消首次最大化保护。相同 9 次音标体系切换从约 26.328 秒恢复到 0.993 秒，随后全部 94 个诊断场景完成。该替换探针用于确认热点，不能作为产品修复验收。

产品修复位于 `desktop/src/ptb_desktop/presentation.py`：

1. 缓存存活且仍属于当前视图的 Quick 控件。
2. 只扫描 WebEngine 的直接 QWidget 子控件。
3. 合并同一轮 ChildAdded/ChildRemoved 触发的发现，删除和重建后重新发现。

原最大化前截图、当前视口文档检查、连续两张有效帧和超时保护保留。没有全局关闭 GPU，也没有改变科学功能。

旧宿主诊断证据为 `output/validation/manual-performance/qt-async-before2-c19c3e8960994ea4b7be779db4c17980/` 及 `output/validation/m17-manual-performance-20261024/qt-async-before2.log`。诊断直接子控件探针为 `output/validation/manual-performance/qt-direct-diagnostic-97cfbca1fde348f2b721c0d7ad59ff38/`。旧诊断进程在 inspector 失败后仅终止了其已确认自有进程树，未操作用户实例。

## 验证与失败记录

- 35 项说明书/M17 定向测试、类型检查、契约/UI 数据检查及独立生产构建通过。
- 实际 Chrome `manual-outline` 通过 7 组主要行为、6 组主题/字号、19 个侧栏图标，并验证原生媒体在许可撤销时暂停、隐藏时拒绝播放、恢复时重新可用，正文/媒体元素保持。证据 `output/validation/manual-outline/navigation-1792822312478/`，scratch 清理回执为 deleted。
- 14 项桌面 presentation/startup 测试通过，包含 100 次存活控件复用、实际 Quick 控件删除后重建、视图销毁后的迟到发现、100 次子控件创建合并发现，以及原有黑帧/错误尺寸/旧回调/恢复/超时保护。
- 初始 Qt 测试回调超时及异步 inspector 失败保留，排查后定位到宿主扫描。首次扩展的单个大步滚轮事件未稳定滚动，已按实际 Qt 事件路由改为三档 Begin/Update/End 手势，保留失败并完整重跑，未删双向滚动断言。
- 额外的 offscreen 测试在跨用例再次创建 Quick 控件时触发原生 access violation。将不属于该用例目标的额外 Quick 创建移除，控件删除重建仍由独立用例实际覆盖，原有保护断言保持。
- 一次外部 Pillow 取图未捕获目标窗口，`qt-progress-native.png` 明确排除并清除，`excluded-image-cleanup.json` 回执为 deleted/remaining=false，不用于任何像素或界面结论。后续仅用自有进程内 Qt 取图。

## 当前产品 Qt 交互

`qt-native-after4-90c257b2cb9e4c7eafde353169910893` 使用产品实现，仅包裹原调用计时，未替换控件发现算法或 GPU 路径。179 个场景通过，包含原 11 场景，以及浅深主题 × 宋体/微软雅黑/楷体 × 10/12/14/15/18/20/24 px × 普通/最大化 × extIPA/说明书的 168 组原生 QTest 双向滚轮。当前 21 章全部打开，渲染进程退出为 0。

168 组矩阵只有 1 组出现 67 ms 长任务，最大渲染器定时间隔为 80.9 ms。此数值属于当前操作与设备，不代表首次加载大章节没有短暂停顿。实际 attach 调用 475 次，只执行 2 次直接子控件发现，耗时 0.594/0.041 ms，attach 最大 0.625 ms。浅深/普通/最大化截图已检查，正文和图片可见。该轮 scratch 回执为 deleted。

普通/最大化/还原及地图切换补验分别记录，失败保留。其中首次地图关闭检查未处理原有草稿协商，另一次重跑在原生键盘检查得到 `focus-visible=false`，该轮未执行地图步骤。原 179 场景轮的键盘检查通过，未用未进入地图的那轮判断地图结果。

产品补验 `qt-stable-final-5501f8d8fd024db48eaf7ce3dd09a2d1` 通过 11 个 CDP 场景。包括实色/玻璃两组音标切换、同章定位、静置和标签往返，及 M17/说明书/地图 18 次往返。所有重复导航和静置资源请求均为 0，同章正文/媒体及地图控件保持原实例。原生窗口按钮连续完成最大化/还原两轮，视口在 1707×1019 与 1440×900 间恢复，随后按真实草稿协商关闭本轮临时地图。原生 `native-final.png` 已核对 extIPA 字形与玻璃壁纸，初次收尾因子进程尚在而保留空 scratch，进程退出后既有清理入口记录 deleted。

35 项前端定向与 14 项桌面定向的最终日志分别为 `frontend-targeted-final.log`、`pytest-presentation-final.log`，位于 `output/validation/m17-manual-performance-20261024/`。文档校验检查 2502 文件、460 来源记录、183 任务，错误为 0，未修改的历史快照断链单列。相关差异格式检查通过。当前选定源码与独立前端入口摘要记于同目录 `source-identity.json`。

大 M01 章节在 Qt 中连续 12 次隐藏/恢复仍累计约 1.624/1.689 s 渲染器主线程时间，单次长任务约 79–138 ms。没有新资源请求，也没有宿主秒级阻塞。进一步只用于定位的 tab `scrollIntoView` 取消实验反而测得 2.093/2.166 s，未支持这一优化假说，因此没有修改工作台的自动揭示标签行为。诊断记录为 `qt-no-tab-scroll-diagnostic-1f7296fcc26940458c2aa5876038e6e6`，不混入产品通过结论。

主要复现入口：

```powershell
node --test frontend/tests/manual-reader.test.ts frontend/tests/manual-reader-render.test.ts frontend/tests/m17-r3.test.ts frontend/tests/m17-r2-sources.test.ts frontend/tests/m17-ipa-plus.test.ts
.\.venv\m14\Scripts\python.exe -B -X utf8 -m pytest -c tests/pytest.ini desktop/tests/test_presentation.py desktop/tests/test_startup_presentation.py -q
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/verify_manual_performance_qt.py --frontend <独立生产构建目录> --glass-matrix --label <本轮唯一前缀>
node scripts/verify_reader_responsiveness_cdp.mjs <自有Qt端口> <证据目录> source-qt --geo-map
```

Python 测试需按工程入口绑定当前 `desktop/src`、`backend/src` 和 `packages/phonetic_core/src`。最终 EXE 可使用同一 CDP 入口，第三个参数改为 `final-exe:<最终SHA256>`，同时由成品启动器核对该 SHA 和独立配置归属。CDP 报告不代替原生像素检查。

## 最终包待验

最终冻结必须包含本次 `presentation.py` 与说明书三个产品文件。最终 EXE 需重复正常可见、默认 GPU 的操作和像素检查，并绑定最终 SHA。第二台机器、其他显卡、实体设备、跨平台和长时间运行均未由本页验证。

## Mac 退出回归与缓存生命周期（2026-10-24）

79e678 同业务源码的 Mac 诊断应用完成页面检查后，两次在逐对象退出阶段崩溃，实际栈顶均为 `QtWidgets.meth_QWidget_parentWidget`。新热点与 M19 两套独立配方都复现，不能仅依据报告中的业务断言成功认定成品通过。原失败栈位于 M19 Mac 验收目录的 `final-79e678/ui-hotspots-exit.ips` 与 `m19-exit.ips`。

共享修复保留原始 QWidget 包装作为缓存生命周期依据，在 destroyed/ChildRemoved 时立即失效，避免解引用可能悬空的 cast Quick 父指针。成功关闭窗口时调用 `dispose()`，移除发现事件并停止排队回调；关闭后的 prepare/begin 不再访问界面。两帧就绪、视口、超时、默认 GPU 和恢复重试语义保持。涉及 `desktop/src/ptb_desktop/presentation.py`、`host.py` 及相应测试。

17 项 presentation/startup 定向通过，包含缓存命中不调用 cast.parentWidget、已销毁/重建缓存、旧 destroyed 回调不能清新缓存、待处理发现与旧文档回调在关闭后不再扫描。Mac 源码严格复验沿用原 page/profile 分别释放的流程，热点与完整 M19 两进程均实际返回 0，分别核实 cleanupComplete 和末条 cleanup-complete；热点 7 组/30 测量、M19 21 项含8图通过。证据 `macos-review/remote-20261024/source-close-fix-1/exits.json`。

这轮结果是修复确认，79e678 Windows/Mac 转为诊断候选。整个应用扩查还发现其他模块的保存/迟到试听/文档缓存问题，修复后须统一重新冻结、重验最终字节。本机修复后矩阵已完成179场景，其中168项为三字体七字号、浅深、普通/最大化与音标/帮助组合。实际进程返回0；最大渲染间隔169.3ms、长任务4次，保留短暂负载事实。证据 `output/validation/manual-performance/qt-dispose-lifetime-b68fec9d369b48b98d631ee4906cc520/report.json`。最终包另验。
