# M15 纯客户端感知实验验收

2026-09-27。开发功能与正式工作台接线已完成，`verified` 限定以下 Windows Chrome、实际 Qt 与合成材料证据。完整跨平台状态仍为 `in_progress`，离线 B 按用户批准保留方案。没有运行 M15 服务器 API、账号、SQL 数据迁移或远程计算。

正式入口：统一工作台 → 标注与实验 → 感知实验。无需登录。实现位于 `frontend/src/modules/perception/`，同一个 Vue 页面和 Runner 同时用于浏览器与 Qt。不是测试页替代交付。

## 功能及来源

[逐项来源映射](../modules/evidence/M15-source-map.md)覆盖说明书 6.1–6.2、原 HTML 与 service/models。音频四种顺序、原 RT 开放阶段、范围首匹配等保留；0 值、失败继续、重复回答、复杂配置丢序列等缺陷逐项登记。文本/图片预解码、两次 rAF 后开放作答的修正已经用户明确批准。

V2 基准先从未修改的原函数体独立执行，未导入 V3 生成 expected。证据为 `output/validation/m15-baseline/baseline.json`，包含四范式、responding 后起算、0 ISI 被改为 500、播放错误继续、缺失等待 100 ms、闭区间洗牌和首匹配分段。它是源码函数执行基准，不是完整旧 React 页面或声卡时序测量。

F01–F07 已实现：四范式，音频/图片/TXT 素材与分组预览，完整角色/hash 关联，序列编辑/排序/范围洗牌，参数/分段按键/阶段提示，问卷，资源助手及项目/XLSX 往返，独立会话、本地提交、恢复、三格式结果和关闭保护。范围洗牌新增 mulberry32-v1，保存 seed、before/after 和最终实际试次顺序。

## 实际命令和结果

命令在仓库根目录执行，未更改系统音量或默认设备。浏览器与 Qt 使用低振幅合成刺激，试音的声音影响已提前说明。

| 检查 | 命令 | 结果 / 证据 |
| --- | --- | --- |
| V2 独立基准 | `node tests/e2e/m15-baseline.cjs` | 通过，见上述 baseline.json |
| M15 逻辑与状态机 | `node --test frontend/tests/m15*.test.ts` | 21 项通过，包括写失败、重复键、早到键、播放失败和恢复 |
| 全前端定向回归集合 | `npm --prefix frontend test` | 172 项通过，0 fail/skip；`output/validation/m15-final/frontend-tests.log` |
| 类型 / 构建 | `npm --prefix frontend run typecheck` / `npm --prefix frontend run build` | 通过；M15 独立 JS 分包约 549.82 KB，gzip 181.35 KB，CSS 3.86 KB。保留超过 500 KB 的构建提示，未放宽阈值 |
| 正式 Chrome 运行 | `node tests/e2e/m15.cjs` | 9 组通过；`output/validation/m15-browser/1790488889778/report.json` |
| 正式 Chrome 编辑/恢复 | `node tests/e2e/m15-recovery.cjs` | 8 组通过；`output/validation/m15-recovery/1790488415896/report.json` |
| 浏览器真实存储及受控故障 | `node tests/e2e/m15-runtime.cjs` | 15 组通过；`output/validation/m15-runtime/1790488465885/report.json` |
| 实际 Qt Workbench | `.venv/v3-dev/Scripts/python.exe -X utf8 scripts/verify_m15_qt.py` | 最终 13 步通过；`output/validation/m15-qt/aab76c2b48a24902a0fcb06171f56181/report.json` |
| WSL 静态资源 | `wsl -d NInfer --exec /home/ninfer/ptb-p11-20260926/venv/bin/python /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/scripts/verify_m15_linux_static.py` | 通过；`output/validation/m15-linux-static/20260927-060201/report.json` |

Chrome 使用实际安装的 Windows Google Chrome 152.0.7977.83，以 headless 模式执行。每个端到端测试先生成独立固定生产构建，避免其他任务的共享文件改动触发 HMR。早期一次开发服务验证被并行修改触发的 HMR 打断，保留失败记录，改为固定构建后复验通过。

正式恢复测试曾发现 computed 对同一个可变会话对象缓存，导致完成后的关闭提示不更新。已改为按 revision 发布快照并通过真实关闭回归。额外修正恢复素材被双计容量、同 hash 多角色缓存缺少解码溯源字段，均有定向用例。

Qt 复验 `10ae4fc5e8e44f3787f8bd3df07c7de2` 曾在阶段 3 超时：脚本在本地存储仍 busy 时注入文件，导入未开始。脚本改为先等待页面 `aria-busy=false`，最终 `0de52e5b79b0479886b0d998846bb55f` 全部通过。原失败证据保留，没有放宽产品检查或隐藏失败。

`npm --prefix frontend run ui-data:check` 与定向 `git diff --check` 通过。V2 HTML、说明书、service/models 四份 SHA-256 最终复核与起始值一致。全库 `scripts/validate_docs.py` 检查 962 文件、333 来源、41 任务，保留 3 条非 M15 错误：README 中旧 M10-R5 EXE 链接、MFA 手册与 M11 映射指向尚未生成的 m11-report.md。没有 M15 新增失效链接；没有修改其他任务文件来令全库检查表面通过。原始输出为 `output/validation/m15-final/docs-check.json`。

## 证据覆盖及强度

| 项目 | 实际证据 | 边界 |
| --- | --- | --- |
| 四范式 / 0 ISI | Chrome 正式页面 X、AX、ABX、AXB，真实 Web Audio；导出完整角色与计划顺序，playing 阶段按键不接受 | 没有声学回环，不等于物理输出顺序测量 |
| 计时与调度 | 独立 OfflineAudioContext 样本值/间隔；真实 AudioContext 的计划时间与作答窗口回读 | OfflineAudioContext 是离线渲染，不是设备时延 |
| 键盘 / 阶段 | repeat、长按跨阶段、组合键、IME、大小写、早到事件、单次响应、手动阶段/播放；分段首匹配和范围拒绝 | 浏览器自动化按键和适配器事件，不是物理按键测量 |
| 素材 | 真实 IDB Blob、同名不同内容 TXT 与图片呈现、损坏音频解码拒绝、缺 Blob 停止、预算分块 | 测试缩小预算来触发分块，未用 512 MiB 项目压测峰值 |
| 编辑/导出 | 真实目录模板、范围洗牌、项目 JSON 和序列 XLSX 往返；JSON/XLSX/CSV 下载后由程序回读 | 用户实际保存须明确确认，页面不将点击下载当保存成功 |
| 本地存储 | 真实 IDB 事务/CAS，真实第二标签 Web Lock 拒绝；每 trial 的 running/完成边界 | QuotaExceededError 是对真实 IDB put 的受控注入，未填满整块磁盘 |
| 恢复 / 异常 | 实际 reload、已有问卷/答案保留、活动 attempt interrupted；显式再呈现形成新 attempt；context suspend 与受控 blur | 未做强制进程崩溃、断电或真实休眠循环；pagehide 不保证来得及提交 |
| 视觉 | Chrome 浅/深主题、900×700 与 150% 页面缩放、中文/IPA、专注视图截图 | 未覆盖所有显示器/DPI/操作系统字体回退 |
| Qt C | 实际 Workbench、ptbapp 静态资源、Web Locks/IDB/AudioContext、原生 QTest 手势和按键；X 试次完成与 XLSX/JSON 文件回读 | 测试替原生保存对话框提供路径；未冒充人工点击或全四范式 Qt 验收；无 EXE 打包 |
| Linux | 现有 NInfer 的原生 CPython 3.11.14 核验构建入口、M15 JS/CSS/字体与 SHA-256 | PATH 未发现 Node/Chromium；不据此声称全机未安装。未验证 Linux 浏览器、音频设备或 HTTP 部署 |

runtime 测试页只用于注入故障，产品不导入该页面。Chrome 报告 `external=[]`，断网期间四范式及 TXT 均完成并导出。Qt 资源报告提供实际 ptbapp URL，未用 Chrome 结果替代 Qt 能力检查。Qt 有既存 libpng iCCP/tRNS 资源警告，测试无可见弹窗，不代表实验出错。

截图位于 Chrome 证据目录中的 `focus-text-light.png`、`design-dark.png`、`design-dark-150.png`。已检查中文、IPA、专注状态和小窗口滚动。

## 时间字段定义

RT 方法版本为 `response-open-to-handler-performance-ms/v1`，单位毫秒。`rtMs = keyHandledPerfMs - responseOpenPerfMs`。音频整段计划播放结束后检测 AudioContext 时钟，再开放回答，保持旧作答阶段起算语义。视觉资源完成读取/解码并呈现两次 rAF 后开放。

| 字段 | 时钟 / 单位 | 含义与限制 |
| --- | --- | --- |
| `timing.timeOrigin` | performance 的页面时间原点，epoch ms | 识别同一页面时钟域；恢复不跨原点计算 RT |
| `preparedPerfMs` | performance，ms | 刺激准备完成后的计时快照 |
| `audioMapping.audioSeconds` / `performanceMs` / `bracketMs` | AudioContext 秒 / performance 毫秒 | 前后夹取 performance 得到映射中心与取样跨度。不同量纲不直接相减 |
| `planned[].startAudioSeconds/endAudioSeconds` | AudioContext，秒 | Web Audio start 调度及 buffer 时长。默认预留 50 ms 调度准备，提示音开启时另有原 500 ms 提示间隔；刺激间 ISI 依配置含 0 |
| `planned[].scheduledAtAudioSeconds/scheduledAtPerfMs/scheduleLateByMs` | AudioContext 秒 / performance 毫秒 / 毫秒差 | 提交节点调度前取样，late 为相对计划起点已迟到的非负时长；可算计划提前量。只描述 JS 调度提交，不代表物理声音开始 |
| `outputTimestamp` | 浏览器返回的 context 秒 / performance 毫秒 | getOutputTimestamp 可观测估计，无值为 null |
| `outputStartEstimatePerfMs` | performance，ms | 用上述 outputTimestamp 映射计划起点。不是耳机实际发声测量 |
| `observedEnded[]` | 回调到达时 performance ms / AudioContext 秒 | 只记录 ended 回调观测，不用于声学终点或 RT 起点 |
| `endDetectedPerfMs` | performance，ms | 界面轮询发现音频时钟达到最终计划终点的处理时间 |
| `visualFramePerfMs` | requestAnimationFrame timestamp，ms | 第一帧呈现回调；第二次 rAF 后开放回答。不是物理屏幕显现时刻 |
| `responseOpenPerfMs` | performance，ms | 实际作答窗口开放、RT 起点 |
| `responseDelayFromMappedEndMs` | performance，ms | 相对配对映射计划终点的响应开放迟延诊断。受音频时钟/映射精度影响，不是设备零延迟证明 |
| `keyEventPerfMs` | event.timeStamp，ms | 可识别为当前页面单调域时记录；epoch 型/不可靠值为 null |
| `keyHandledPerfMs` | performance，ms | 接受有效键时处理时间；与事件时间分开保存 |
| `startedAt/finishedAt/createdAt/events.at` | ISO UTC 墙钟 | 只作日历记录，不计算反应时 |

没有可靠数据的字段使用 null，不填 0 伪装零延迟。播放失败/缺刺激/中断不会产生正常 completed attempt。导出含全部 A/B/X 身份、hash、原 WAV 采样率、实际解码率/声道/时长；压缩格式原采样率未知记 null。浏览器可能重采样，程序没有归一化、裁剪、主动声道转换。

## 资源、隔离与离线

容量限制详见 [操作说明](../manual/perception.md)。128 MiB 是保留的解码刺激预算，包含图像像素估算；GC、浏览器内部缓冲及其他模块不在这个数值内，不能宣称整个浏览器峰值小于 128 MiB。大项目顺序预检后分试次准备，单试次过预算阻止呈现。未打包含素材 ZIP，避免无界解包。

本机存储请求可能被拒绝或清理。配置恢复与刺激齐全分开判定，结果可在缺刺激时单独导出。session UUID、participant UUID、配置快照、revision CAS 和 Web Locks 防止会话混写。本地大 Blob 不进 localStorage。正常关闭有异步保存和未确认导出保护，强制关闭/断电不能承诺拦住。

AppShell 按需加载，M15 未打开时不加载 SheetJS 分包。正式运行捕获按键、冻结主题切换、无试次动画。公共播放器和可见活动任务冲突会阻止预检，用户需确认其他录制/计算已暂停；不会终止其他任务/进程。可检测的 blur/visibility/context/device/fullscreen/freeze/resume/长调度间隙都有事件策略；真实设备拔插、退出全屏和休眠硬件场景尚未验收，不能声称能检测所有故障。

- A 已通过：页面/分包/字体/刺激准备完成后断网，完成正式实验和导出。
- C 已定向通过：实际 Qt 内置 ptbapp 资源使用同一运行器，无 M15 Python/服务依赖。
- B 按用户批准留具体方案：[ADR-M15-001](../decisions/ADR-M15-client.md)。本轮没有注册 Service Worker，不保证关闭浏览器后断网冷启动，也不保证 file://。

服务器资源/远程计算状态为不适用：服务器仅分发公开静态文件，M15 不占用服务端计算槽、账号空间或 3 天保留额度。未执行云端生产部署。物理耳机声音、屏幕显现和键盘端到端时延未测；没有依据宣称达到任何毫秒精度。

## 交付及后续

[计划](../plans/modules/M15-perception.md)、[架构决定](../decisions/ADR-M15-client.md)、[来源映射](../modules/evidence/M15-source-map.md)、[用户手册](../manual/perception.md)、[统筹摘要](m15-coordination-summary.md)共同构成交付。保留其他任务的共享差异。未修改 V2/用户语料/旧结果，未 push、部署、执行现存库迁移、修改全局依赖或打包 EXE。
