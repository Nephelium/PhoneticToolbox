# P17 M08 逐项验收报告

已定位加载瓶颈并修复：原页面等待持久F0任务完成才显示音频，管道4KiB片段各自触发本机write/fsync/状态保存，使34.499秒音频首显106.385秒。现在先读原音保留实际采样率/声道/时长，再执行同一F0任务。本机Receiver聚合至1MiB后持久写入，预算/哈希/取消/fencing不变，托管仍原64KiB。科学结果未改。详见 ADR-P17-M08。

0779bee实测短音冷/热波形约2.30s、长音106.385s。完整扩展轮新值：短音冷波形106ms、F0及历史1904ms，热波形91/90ms、F01888/1892ms，长音波形219ms、F04318ms。不同运行长音200–219ms / F03834–4318ms。`transport-timings.json`保留每阶段真实read/task耗时。波形首显与F0完成分开报告，未把219ms当全分析完成。20次完整Shift拖动含双rAF等待p95=51ms（数组29–53ms），为浏览器整次手势观察值。

`stream-parity.json`：3份自然输入的metadata/FLOAT64 WAV共6产物旧新逐字节一致。`science-20261001-r3/report.json`：原V2真实输入0.8/1/1.2速度，在两端固定Praat随机seed42以对齐无声随机重合成，26267数组值与3个PCM16导出逐字节相同。实际任务取消约305ms、恢复约1085ms。`python -m pytest -o addopts= backend/tests/test_p17_m08_stream.py -q` 10通过：997/4096/65536/1048576片段、聚合次数/边界、错误哈希/截断/尾随流拒绝。结构测试使用任意字节，不冒充自然音频。

另修正反向手绘插值端点颠倒，同源V2也有该缺陷；该编辑行为修正单列，不改科学算法。17项前端状态含新增反向回归通过。真实快速换源最终F0归属、保存编号/下载/PNG、F0导入、拐点添删/constant/full、批量倍率1.1+10Hz真实提交通过。

未测：所有音频格式、实际声卡听感、全部手势/按键、>120s自然音频（授权目录无此文件）。已补独占本轮输出真正重命名、删除成功及UI停止剩余提交。删除后清理当前结果/失效迟到读取守卫已修，11-49轮删除后等待1800ms无410，当前失效试听清空，专项通过。

三栏新布局左参数、中四图、右任务历史，普通滚轮与最终几何追加记录。

本轮规则先读取 V2 手册及实际源码建立，逐项覆盖表见同目录规则文件。完整程度仍为 **in_progress**，不能把下列成功流程扩展为全部控件、平台或设备验收。输入只使用指定自然录音目录，原文件SHA在宿主退出复核。M06/M07合成产物为正常产品输出，未冒充自然输入。

公共生产文件/任务适配器经替代QWebChannel连接独立Chrome与真实本机服务。数据目录为本轮独占新目录，不操作已有任务库。所有时间为本机观察，包含UI等待/轮询，未声称实时硬期限。原音只读、无V2修改、无全局环境/DDL/commit/push。Linux/远程未开放，实际Qt模块联合验收由主代理另列。

命令：`node --test frontend/tests/m06.test.ts frontend/tests/m07.test.ts frontend/tests/m08.test.ts` 17通过；`node tests/e2e/p17-m06-m09.cjs output/validation/p17/source/m06-m09-inputs.json --extended` 完整扩展流程退出0；`python scripts/p17_m06_m09_readback.py <M08运行目录>` 23组结果、37WAV、6组逐步与拼接PCM完全相等。证据：各模块 `browser-2026-10-01T11-25-04-530Z/browser-report.json`（旧布局完整控件）和 `browser-2026-10-01T11-30-57-286Z/`（新增三栏再次完整扩展成功）。

首次扩展两次因测试定位问题中断：重复“使用说明”按钮需要限定M08模块；异步错误提示需要等待。保留失败目录，不计入通过。最后一轮无pageerror。

## 追加软件控件与大屏证据

`browser-2026-10-01T11-41-04-895Z` 为整页三栏、扩展控件及恢复流程成功轮。独立Chrome模拟1920×1000、1366×768、1280×720、2560×1360、3840×2080 CSS视口；1920/2560/3840的模块scrollHeight等于clientHeight，1280允许响应滚动。普通滚轮实发后module scrollTop未改变。高分辨率是模拟视口，未称实体4K设备验证。表格/侧栏独立滚动保留完整控制。大屏PNG文件为 `large-2560.png`、`large-3840.png`，普通整页图 `three-column.png`。

真实产物最终回读：该轮26个结果组、36WAV（已删除本轮指定输出被排除），6组连续统combined仍与每个step的PCM顺序拼接完全相等。原输入SHA退出复核不变。

长音首次优先单独测 `browser-2026-10-01T11-35-43-506Z`：冷波形193ms/F0及历史4188ms，热波形204ms/F0及历史3965ms。当时另有本轮独立验收进程并行，记录实际资源竞争而不声称纯净独占基准。最后计时脚本另加解码worker耗时观察，未改产品解码器。

## 最终收口（产品源码已冻结）

最终命令 `node tests/e2e/p17-m06-m09.cjs output/validation/p17/source/m06-m09-inputs.json --extended --recovery`，运行目录 `browser-2026-10-01T11-49-52-478Z`，退出0、四模块 `success:true`、无pageerror。各模块目录包含1920真实数据/六组结果、1366/1280、2560/3840、浅深色截图及逐步骤记录。构建交给主代理统一执行，本子代理未全局build。

`python scripts/p17_m06_m09_readback.py output/validation/p17/M08/browser-2026-10-01T11-49-52-478Z` 退出0：31结果组、41WAV哈希/尺寸/率/帧/有限值核对通过，7组连续统PCM逐样本拼接完全相等，metadata同时记录parselmouth和reaper。全部输入SHA宿主退出复核不变。

11-49波形/解码观察：冷短音90ms首显、worker15.90ms、F0+历史1891ms；长音202ms首显、worker19.30ms、F0+历史3559ms。原始记录保留在browser-report与transport-timings，worker解码计时为测试Vite注入观测，产品解码模块未改。

最终独立长音冷/热分阶段补测：`browser-2026-10-01T11-52-18-335Z` 冷长音首显211ms、解码worker24.30ms、F0+历史3898ms；同进程热长音首显212ms、解码20.00ms、F0+历史3581ms。读原音桥耗时顺序见transport-timings；页面首显包含传输/worker启动/渲染，不能直接与各阶段相加代替端到端值。

## 普通入口与播放补充验收

命令 `node tests/e2e/p17-inputs-playback.cjs output/validation/p17/source/m06-m09-inputs.json`，`browser-2026-10-01T12-03-27-065Z` 四报告success:true、退出0、无pageerror。仅补普通入口及播放所需单次产品产物，不重跑全量科学矩阵。使用headless `--mute-audio`，验证播放按钮转暂停、时钟前进、暂停及停止，不声称实际声卡听感。没有操作前台。

M06源录音/合成产物、M07源/目标、M08原音/合成音、M09重建产物统一播放条均通过。各报告playback字段保留实际时钟。M07实际AppShell方法与来源弹层及Esc通过；目录取消受控返回null保持列表，真实独占空目录禁用分析，再打开录音目录恢复。M08同样空目录/取消恢复通过。原生OS选择对话框未打开，取消证据限定桥返回语义。

浏览器本地文件路径：实际生产previewFiles与各页面文件input，M07多WAV导入、M08单文件及webkitdirectory目录导入、M09自然录音派生PNG/JPEG/BMP逐个读入并解码130×129通过。图像来自既有真实录音谱图相同灰度像素的格式编码，未生成自然音频测试输入。认证服务器上传/持久资源链没有由本次本地导入证据替代。

最初补测曾因60ms固定等待尚未看到播放时钟、异步目录刷新未等待、同名aria-label定位选择器歧义中断；改为等待实际状态/限定select后通过。这些为测试同步/选择器修正，未改产品源码。证据图：各模块 `playback.png`，M07 `inputs-cancel-playback.png`，M08 `browser-import.png`，M09 `browser-import-png.png`、`browser-import-jpg.png`、`browser-import-bmp.png`。
