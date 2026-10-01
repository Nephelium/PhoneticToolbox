# P17 M09 逐项验收报告

输入是指定自然录音计算的灰度幅度语谱图，生成方法scipy.signal.spectrogram（256点窗/192重叠），-30..0dB映射白至黑，上下翻转使低频在下。按该图真实0.5324375s/0–8000Hz标定，32迭代、10ms窗、44100Hz、seed0实际重建与4文件保存回读通过。图像相位丢失，未声称重建等于自然原音。

实际Qt专属捕获已实现，交互及状态恢复限定验证：隐藏工具箱后留在原生选角层，顺序选四角，Enter确认、Backspace/右键撤销、R重选、Esc取消。第四角不立即锁定。拒绝自交/退化；按QScreen逻辑坐标到物理像素裁切再归一化，将图片与四角一次带回，保留导入图手动选角。原窗口正常/最大化/最小化状态在成功或取消后恢复。截图仅在内存，确认后仅裁切区域交给provider。网页无此原生能力。

命令 `python scripts/p17_m09_capture.py <真实录音语谱图.png> output/validation/p17/M09/qt-capture-20261001-r2` 退出0。原计划背景为本轮真实语谱图测试窗。r2独立像素复看发现前台被现有M08结果截图窗口覆盖，实际保存的是本任务M08界面截图，不能作为真实语谱图背景捕获通过。r3–r6增加灰度像素断言后均明确失败，未保存非预期像素。停止争抢前台，不关闭用户正在看的窗口。物理屏DPR1.5，窗口逻辑1707×1067，四点返回/最大化取消/最小化取消、无效交叉拒绝、撤销/重选的Qt交互通过，预期背景内容捕获受前台竞争限制；1/1.25/1.5/2裁切映射单独通过。第一轮错误地要求浮点裁切归一化边界精确为0/1；修订为按floor/ceil像素边界的≤1像素误差后通过，未放宽产品算法。

额外修复换图时current任务归属立即清理、尺寸读取/错误反馈按epoch避免旧读取覆盖。原图与四角导入仍可人工调整。新布局左标定、中输入/重建/音频、右任务/导出。

未通过：指定真实谱图背景像素捕获。未测：多物理屏、Wayland/macOS、截图OS异常恢复注入、认证服务器上传、预算极限、实际声卡听感。已补真实128迭代重建UI取消、重试成功。截图映射结构检查不冒充在所有DPI物理设备实际运行。

本轮规则先读取 V2 手册及实际源码建立，逐项覆盖表见同目录规则文件。完整程度仍为 **in_progress**，不能把下列成功流程扩展为全部控件、平台或设备验收。输入只使用指定自然录音目录，原文件SHA在宿主退出复核。M06/M07合成产物为正常产品输出，未冒充自然输入。

公共生产文件/任务适配器经替代QWebChannel连接独立Chrome与真实本机服务。数据目录为本轮独占新目录，不操作已有任务库。所有时间为本机观察，包含UI等待/轮询，未声称实时硬期限。原音只读、无V2修改、无全局环境/DDL/commit/push。Linux/远程未开放，实际Qt模块联合验收由主代理另列。

命令：`node --test frontend/tests/m06.test.ts frontend/tests/m07.test.ts frontend/tests/m08.test.ts` 17通过；`node tests/e2e/p17-m06-m09.cjs output/validation/p17/source/m06-m09-inputs.json --extended` 完整扩展流程退出0；`python scripts/p17_m06_m09_readback.py <M08运行目录>` 23组结果、37WAV、6组逐步与拼接PCM完全相等。证据：各模块 `browser-2026-10-01T11-25-04-530Z/browser-report.json`（旧布局完整控件）和 `browser-2026-10-01T11-30-57-286Z/`（新增三栏再次完整扩展成功）。

首次扩展两次因测试定位问题中断：重复“使用说明”按钮需要限定M08模块；异步错误提示需要等待。保留失败目录，不计入通过。最后一轮无pageerror。

## 追加软件控件与大屏证据

`browser-2026-10-01T11-41-04-895Z` 为整页三栏、扩展控件及恢复流程成功轮。独立Chrome模拟1920×1000、1366×768、1280×720、2560×1360、3840×2080 CSS视口；1920/2560/3840的模块scrollHeight等于clientHeight，1280允许响应滚动。普通滚轮实发后module scrollTop未改变。高分辨率是模拟视口，未称实体4K设备验证。表格/侧栏独立滚动保留完整控制。大屏PNG文件为 `large-2560.png`、`large-3840.png`，普通整页图 `three-column.png`。

真实产物最终回读：该轮26个结果组、36WAV（已删除本轮指定输出被排除），6组连续统combined仍与每个step的PCM顺序拼接完全相等。原输入SHA退出复核不变。

## 最终收口（产品源码已冻结）

最终命令 `node tests/e2e/p17-m06-m09.cjs output/validation/p17/source/m06-m09-inputs.json --extended --recovery`，运行目录 `browser-2026-10-01T11-49-52-478Z`，退出0、四模块 `success:true`、无pageerror。各模块目录包含1920真实数据/六组结果、1366/1280、2560/3840、浅深色截图及逐步骤记录。构建交给主代理统一执行，本子代理未全局build。

`python scripts/p17_m06_m09_readback.py output/validation/p17/M08/browser-2026-10-01T11-49-52-478Z` 退出0：31结果组、41WAV哈希/尺寸/率/帧/有限值核对通过，7组连续统PCM逐样本拼接完全相等，metadata同时记录parselmouth和reaper。全部输入SHA宿主退出复核不变。

`python -m pytest -o addopts= desktop/tests/test_p17_m09_capture.py -q` 10通过：1/1.25/1.5/2物理像素映射、交叉/退化/不足点拒绝、正常/最大化/最小化在grab失败后的恢复。该测试以空Pixmap和几何点验证结构，不读取桌面，不冒充真实谱图像素验收。指定谱图背景捕获仍因前台竞争未通过，未再操作用户前台。

## 普通入口与播放补充验收

命令 `node tests/e2e/p17-inputs-playback.cjs output/validation/p17/source/m06-m09-inputs.json`，`browser-2026-10-01T12-03-27-065Z` 四报告success:true、退出0、无pageerror。仅补普通入口及播放所需单次产品产物，不重跑全量科学矩阵。使用headless `--mute-audio`，验证播放按钮转暂停、时钟前进、暂停及停止，不声称实际声卡听感。没有操作前台。

M06源录音/合成产物、M07源/目标、M08原音/合成音、M09重建产物统一播放条均通过。各报告playback字段保留实际时钟。M07实际AppShell方法与来源弹层及Esc通过；目录取消受控返回null保持列表，真实独占空目录禁用分析，再打开录音目录恢复。M08同样空目录/取消恢复通过。原生OS选择对话框未打开，取消证据限定桥返回语义。

浏览器本地文件路径：实际生产previewFiles与各页面文件input，M07多WAV导入、M08单文件及webkitdirectory目录导入、M09自然录音派生PNG/JPEG/BMP逐个读入并解码130×129通过。图像来自既有真实录音谱图相同灰度像素的格式编码，未生成自然音频测试输入。认证服务器上传/持久资源链没有由本次本地导入证据替代。

最初补测曾因60ms固定等待尚未看到播放时钟、异步目录刷新未等待、同名aria-label定位选择器歧义中断；改为等待实际状态/限定select后通过。这些为测试同步/选择器修正，未改产品源码。证据图：各模块 `playback.png`，M07 `inputs-cancel-playback.png`，M08 `browser-import.png`，M09 `browser-import-png.png`、`browser-import-jpg.png`、`browser-import-bmp.png`。
