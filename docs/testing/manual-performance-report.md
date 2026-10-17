# 使用说明定位性能与目录焦点验收

2026-10-16，Windows，当前源码及本地构建前端。任务 `MANUAL-READER-PERFORMANCE`。

## 问题和改动

反馈包含使用说明打开、标签切回、左侧目录操作卡顿，以及国际音标表Plus切换后可能持续无响应。反馈者尚未回复软件/Windows版本。先前诊断只用于确定测量入口，本次结论来自当前源码与独立Qt实测。

`ManualReader.vue`原先对同章小节也设置loading并重建整章，同一张图片再次经过Qt资源处理。现在同章目录、搜索、引用及宿主target定位保留已挂载正文与媒体，只滚动到目标。工程修订清空仓库后仍重载，跨章取消及请求代次保持。连续滚动的location通知最多每250毫秒一次，相同位置不重复通知；切章、定位、active=false、KeepAlive停用、页面隐藏和卸载时立即补存。加载中的空正文不覆盖原位置。

用户补充的目录框线来自共用buttons.css对选中按钮的focus-visible覆盖。共用焦点样式排除manual-toc，目录键盘焦点使用文字下划线；选中底色、文字与aria-current保持，其他按钮的焦点提示不变。未修改科研算法、音标内容、草稿格式或说明书正文/素材。

## 实测结果

可见默认GPU Qt 6.11.0实际Workbench，1440×900逻辑窗口，独立内存配置，使用真实ptbapp资源入口及内置字体。

| 操作 | 修复前 | 修复后 |
| --- | --- | --- |
| 参数估计章同章定位 | 正文重建，170毫秒长任务，13次资源请求 | 正文和媒体保留，本轮未观察到长任务，0次资源请求 |
| EGG章同章定位 | 正文重建，82毫秒长任务，20次资源请求 | 正文和媒体保留，本轮未观察到长任务，0次资源请求 |
| 每章12次连续滚动 | 各12次同步位置写入 | 各2次写入，末次位置一致 |
| 浅深主题原生Enter定位 | 共用焦点样式可能绘框 | 选中项focus-visible=true，outline=none，下划线可见 |
| 隐藏、切回及关闭帮助标签 | 原行为对照 | 离开前立即保存，恢复到920px，关闭保留最新记录 |

音标表IPA/extIPA/VoQS浅深主题切换六组及帮助/音标表12次标签往返完成，没有渲染进程退出。本次仍未复现持续挂起，不能据此认定反馈者的全部问题已修复。

独立Chrome 155验证真实软件说明书，18模块目录图标、折叠/点击/滚动/搜索及阅读标记通过，浅深×14/18/24px六组宽窄布局通过。新增断言验证同章正文及img/audio/video元素身份保留，20次滚动合并为2次写入、pagehide立即保存。auto/all/plain × solid/glass六组键盘选中均无outline/阴影且显示下划线，保留原无框断言。

受控浏览器夹具另验五项：同章不重新请求；工程version变化重载；A→B→A晚到B不覆盖；同章连续目标取最后一次；卸载补存并取消后续通知。实际宿主从首页打开全局帮助仍显式定位入门章，未擅自改变这个既有入口语义。

## 命令与证据

```powershell
node --test tests/manual-reader*.test.ts tests/m17-ipa-plus.test.ts
npm test
npm run typecheck
npm run contracts:check
npm run ui-data:check
npm run build
# 上述命令在frontend目录执行。
node tests/e2e/manual-outline.cjs
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/verify_manual_performance_qt.py --baseline "$env:TEMP\ptb-readonly-hang-diagnosis-20261016.json"
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/validate_docs.py
```

- 全量前端401项（含定向28项）、类型、契约、UI数据与build通过。文档校验2105文件/126任务零错误，历史归档断链单列；git diff --check通过。Vite保留已有大chunk提示；Qt资源保留已有PNG色彩配置警告。
- Chrome报告/6组截图/清理回执：`output/validation/manual-outline/navigation-1792149156467/`，7组实际交互加5项夹具边界。
- Qt报告/浅深截图/清理回执：`output/validation/manual-performance/qt-98bbb9671b654a62b61573ca85f3f574/`，报告内保留本次修复前实测摘要。elapsed_ms包含脚本settle等待，只以long task、资源请求及元素身份比较，不用该值声称纯交互耗时。
- 首轮Chrome如实因焦点框线失败，随后按用户要求修正样式，保留失败记录。前两轮Qt用例错误假设全局帮助会恢复上章，按宿主真实入口语义修正测试，没有改变应用入口或删除位置保存断言。
- 最新Chrome/Qt scratch均deleted，测试自有进程已结束。首次失败Qt测试的空scratch因进程引用保护暂留，证据/回执保留，未删除用户数据或测试报告。

## 范围与未完成项

以上为2026-10-16源码阶段证据。2026-10-17已随[重复计算排查](redundant-work-report.md)统一生成并验收本地3.0.3免安装/安装版：实际最终EXE浅深主题各连续六次同章点击保持正文/媒体节点，子节点重建0，原生Enter焦点的outline/border/box-shadow均none且下划线可见。产品字节绑定及新成品默认GPU/安装/换版范围以该报告为准，不将此结果扩展为本节位置写入计数的EXE实测。当时仅完成本地验收，后续3.0.3正式发行状态以[裁剪与发行验收](package-trimming.md)为准。

未测试反馈者机器、实体DPI/设备、跨平台及长期负载。首次打开大章节仍整章渲染，Qt资源仍同步读取；后续若需处理这些路径，应单独测量并保持内容/媒体语义。2026-10-17截图已确认反馈者安装版为3.0.3，持续无响应根因仍待进一步证据。

## 2026-10-17 · 3.0.3用户持续挂起复查

任务 `DESKTOP-UI-HANG-DIAGNOSIS`。井井授权先自行排查，范围为本机诊断和记录，没有修改产品源码、成品、系统配置或服务器。用户截图为Windows程序未响应，发生在录音说明章，另报告音标表切换也会卡住。本次将自然操作与人为慢读取实验分别记录。

### 自然操作：未复现持续挂起

当前源码的可见Qt工作台11项检查通过，同章定位无新资源请求，音标表六组切换无长任务，12次标签切换最大脚本计时间隔134.9毫秒。源码报告为 `output/validation/manual-performance/qt-d32e03f2de474d26b7bd3f16b6ac8192/`。

随后直接运行正式3.0.3免安装EXE，494,149,024字节，SHA-256 `3a4158ff9f2126d1287b5b6eeeac689614c0d28da9a13d9c223869a3673ae0cf`。使用新建LOCALAPPDATA/TEMP、可见窗口和默认GPU，关闭测试实例的自动更新/论文接收。Windows 11 26300，实际DPI为144（150%），实际渲染器为Intel Graphics / ANGLE Direct3D11，QtWebEngine UA为6.11.2 / Chromium 140。本机另有NVIDIA显卡，本轮没有切换至该显卡。未使用disable-gpu、用户既有配置或研究文件。

| 成品场景 | 覆盖 | 最大脚本计时间隔 |
| --- | --- | --- |
| 冷打开使用说明并进入录音章 | 用户截图对应章节 | 187.1毫秒 |
| 录音章目录 | 24轮，每轮原生输入点击与轮转小节点击 | 300.5毫秒 |
| 音标表切换 | IPA/extIPA/VoQS共30次 | 97.3毫秒 |
| 说明书与音标表往返 | 60次标签切换 | 184.4毫秒 |
| 首次最大化与还原 | 实际窗口按钮 | 90.4毫秒 |
| 全书章节 | 21章各两轮，共42次 | 319.8毫秒 |
| 深色、大字与背景 | 楷体/Times New Roman、24px、自生成背景、玻璃按钮、最大化，40次标签切换 | 182.1毫秒 |

上述计时为50毫秒脚本定时器的最大间隔，包含调度和渲染影响，不能直接当作点击耗时。独立线程每100毫秒向本任务窗口发送WM_NULL，单次超时250毫秒，345次探测全部成功，最大返回耗时237.847毫秒。脚本异常、资源失败、渲染进程退出均为0，程序退出码0。标签往返仍可观察到约68–122毫秒长任务，这能解释短暂停顿，本轮没有秒级持续挂起证据。最终截图已目视检查，玻璃按钮的实时backdrop-filter元素为0。

成品证据为 `output/validation/ui-hang/exe-4ca5752569264ad8ad2a4335a2da3a40/`，保留EXE身份、环境、各场景报告、窗口响应原始记录和最终录音说明截图。

### 受控慢读取：确认主线程阻塞机制

当前源码Qt宿主的77次资源处理均实测发生在GUI主线程。自然读取的说明书资源处理最大9.403毫秒，全部资源最大89.452毫秒（index.html），未自然出现六秒读取。

在独立测试进程中，仅对一次 `/manual/chapters/m16.json` 资源处理人为加入6秒等待，原文件和产品实现保持原样。处理耗时6001.782毫秒，GUI定时器最大间隔6021.242毫秒，独立窗口消息探测记录26次超时，等待结束后正文恢复。两秒时的Python线程栈保留在 `controlled-wait-stacks.log`，GUI栈停在测试注入的等待点。证据为 `output/validation/ui-hang/resource-9c93c90449e04494a39b0342ccb41fe5/`。

这项实验确认了主线程资源处理对慢I/O的脆弱性。它没有测得反馈者机器的文件读取、杀毒拦截或显卡等待，不能用人为等待代替用户根因。随后井井明确授权先修复后台读取，下节记录该修复；自动记录本机挂起线程的功能仍未添加。

### 复现配方、失败和收尾

本轮诊断脚本保留在忽略目录 `output/validation/ui-hang-20261017/`：`run_exe.py`、`probe.mjs`、`probe_resource_wait.py`。成品启动器使用既有Miniconda基础Python的psutil，源码Qt使用项目m14解释器，没有安装依赖。入口如下：

```powershell
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/verify_manual_performance_qt.py
& 'C:\Users\13680\Miniconda3\python.exe' -E -s -B -X utf8 output/validation/ui-hang-20261017/run_exe.py dist/PhoneticToolbox-303-20261017-015527-89c0e7/PhoneticToolbox-windows-x64.exe
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 output/validation/ui-hang-20261017/probe_resource_wait.py
```

首轮成品诊断启动器因m14缺少psutil而在启动产品前退出，随后使用已有基础解释器。前两轮受控实验分别因测试库未初始化、reload后点击旧DOM而失败，保留失败报告并修正诊断夹具，未改产品或放宽断言。源码11项检查、成品7场景和最终受控实验完成，所有本轮scratch均由既有入口清理并记录deleted，失败证据和脚本保留。Python AST、Node语法、UTF-8/JSON和任务ID唯一性、git diff --check通过；文档验证2203文件/142任务零错误，历史快照断链另列。未安装或切换驱动、未读取反馈者设备、未测试其他显卡/第二台机器/跨平台/长期运行，持续挂起根因保持未定位。

## 同日后续 · 桌面资源后台读取修复

井井明确要求修复已确认的资源阻塞风险。本次仅修改 `desktop/src/ptb_desktop/web_assets.py` 及其测试/验证脚本，维护桌面架构、状态、台账和修改记录，没有修改前端、服务器或既有EXE。

### 实现与生命周期

现有 `ptbapp://app/` 入口保持。请求开始时只做主机、方法和路径字符检查，路径resolve、根目录归属及文件检查、读取、HTML原生通道注入和MIME判断由本模块共享的四线程池完成。GUI线程接收完成信号，再创建以请求为父对象的QBuffer并调用reply/fail。文件字节、JS/CSS/SVG映射、系统MIME、两个HTML入口的WebChannel注入和原有错误状态保持。

请求或handler销毁时设置取消标记并移除待回复记录。已取消的排队任务不读取文件，正在读取的任务完成后丢弃结果。线程池不以profile为父对象，关闭profile不会因销毁线程池而等待磁盘；后台不直接操作Qt请求和设备。Qt允许异步处理scheme请求，并要求跟踪由WebEngine拥有的job生命周期及保留reply设备，依据见[资源处理器](https://doc.qt.io/qt-6/qwebengineurlschemehandler.html)与[请求对象](https://doc.qt.io/qt-6/qwebengineurlrequestjob.html)。

网页版说明书使用fetch和HTTP静态服务，不经过此桌面适配器。这次修改可消除桌面慢资源读取阻塞GUI的路径，不能据此推断所有前端渲染或用户持续挂起均已解决。

### 修复验证

`desktop/tests/test_web_assets.py` 新增22项，另验既有网络白名单和反馈25项，共48项定向测试通过。覆盖后台读取与GUI回复、慢读时其他资源/GUI可用、排队和读取中取消、job/handler销毁、双HTML通道、二进制字节/MIME、缺失/非法路径、Windows junction及根目录被替换、异常错误回传。

可见默认GPU的真实Workbench/FeedbackManager通过 `scripts/verify_asset_reads_qt.py` 检查。最终受控场景结果：

| 指标 | 修复前受控实验 | 修复后当前源码 |
| --- | --- | --- |
| 章节读取人为等待 | 6001.782毫秒，在GUI线程 | 6001.407毫秒，在后台线程 |
| GUI计时最大间隔 | 6021.242毫秒 | 56.704毫秒 |
| 等待读取期间窗口消息探测 | 26次超时 | 64次探测，0超时 |
| 等待读取期间切换音标体系 | 窗口消息阻塞 | IPA/extIPA/VoQS完成，169.09毫秒，读取仍未结束 |
| 另一条读取仍在等待时关闭Workbench/profile | 本轮旧版未测 | 366.545毫秒完成，迟到结果安全丢弃，独立管理器继续响应 |

本轮87次实际资源读取均在后台线程。可见录音章图片complete=true，原尺寸2560×1600，解码及两次动画帧后再截图；实际录音说明和独立管理器图片已检查。第二次人为慢读6002.055毫秒用于真实页面/请求销毁检查，没有渲染进程退出。最终证据为 `output/validation/desktop-assets/qt-eb5345cfdcb841768d353011e8de40ad/`，前一轮关闭检查为 `qt-c8517c76e9af434bba06d7099041dfc4/`。48项测试及清理回执为 `output/validation/desktop-assets/unit-78909b46caac48fc81dac0fe9aefcc46/`。

原有说明书/音标表可见Qt11项复验通过，正文/媒体身份、同章0新请求、位置合并保存、浅深主题三体系及键盘焦点保持；12次标签往返最大脚本计时间隔122.8毫秒。证据为 `output/validation/manual-performance/qt-ad2d68473d9d4a1fbf9eda53a0bbd1c9/`。

```powershell
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/verify_asset_reads_qt.py
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/verify_manual_performance_qt.py
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 -c "import sys; sys.path.insert(0,'scripts'); from workbench_source import configure; configure(); import pytest; raise SystemExit(pytest.main(['-c','tests/pytest.ini','desktop/tests/test_web_assets.py','desktop/tests/test_feedback.py','desktop/tests/test_m01_files.py::test_host_network_allowlist_is_bound_to_its_owned_service','-q']))"
```

首轮新增测试有3项夹具失败，分别是硬编码WAV MIME与本机注册表不同、PowerShell参数传递错误、创建QApplication后才导入WebEngine。随后按既有系统MIME和Qt初始化要求修正夹具，未改产品映射或放宽断言。首轮Qt截图未等图片解码和绘制完成，最终脚本明确等待可见图片解码和两帧后捕获。失败/中间证据保留，本轮临时数据已清理并记录回执。

脚本语法、差异检查和UTF-8/JSON检查通过，文档验证2205文件/142任务零错误，历史快照断链另列；架构检查返回errors=[]。成功、失败及中间验证的scratch回执均为deleted。

这次为Windows当前源码修复，已发布3.0.3及前节冻结EXE基线保持原字节，尚未重打包或公开发布。用户机器的实际挂起、其他显卡、长期等待和跨平台仍未验证。
