# M05 迁移与实机报告

2026-09-27，完整模块 **in_progress**。Windows 的基线、浏览器候选、正式离线任务/结果和本机设备分项已有验证，未将其扩大为所有平台、所有姿态或科研同步均通过。

## 结论与入口

正式入口 `scripts/Start-M05-Workbench.ps1` → 唇形提取；长文件/磁盘采集入口见 [操作说明](../manual/lip-extraction.md)。[六功能组映射](../modules/evidence/M05-source-map.md)、[基线](m05-baseline-report.md)、[浏览器比较](m05-browser-report.md)、[方法决策](../decisions/ADR-M05-methods.md)分别保留。

浏览器模型的 12 组等价检查全部未通过，继续标 candidate。原 legacy 后端独立检测公开素材通过；本轮明确授权录制的 3 段自然视频共 **1023 帧**，V3 解码 RGB、原始关键点、有效帧指标和 mask 与原 V2 独立捕获逐值一致（filter off）。filter on 的公式迁移有独立固定输入测试，VFR 改用实际 PTS 导致的滤波时间差异另列，不能宣称旧 `i/fps` 时间轴等价。

## 实际证据

所有下述路径以 `output/validation/m05/` 为根，输出保留失败尝试。真实录制仅本机使用，属于用户明确授权的私有测试，不进入公开 fixture 或来源素材包。

| 范围 | 证据 | 结果 |
| --- | --- | --- |
| 原 V2 双进程 oracle | `v2-1.json.gz`、`v2-2.json.gz`、公开冻结 `tests/fixtures/m05/v2.json.gz` | 180 帧和解析滤波序列一致，expected 不由 V3 生成 |
| 本地 HTTP、SQLite、子进程、完整发布、取消、恢复、上传中断 | `wiring-632d9b2f168444f0b31c2b8c1aa76f7a/report.json` | 5 组通过，9 次中断上传释放预留但不删除原片段；只操作复制的测试库，无现存库 DDL |
| Chrome 正式页面、历史、GIF、offset、关闭、三种编码模式 | `host-206fbb05c89343b3be10dbed1b08ac03/browser-report.json` | 9 组通过；替换 QWebChannel 传输，实际平台适配/API/任务/写盘均执行 |
| Chrome 真实默认摄像头/音频输入 | `host-57b793770d484222a2cc0280f2123ca0/device-report.json` | 三模式各约 12 秒，原始编码和参数完成本地保存 |
| 自然视频独立旧后端比较 | 同目录 `natural-v2.json.gz`、`natural-comparison.json` | 340/342/341 帧，RGB/关键点/指标/mask 差异均 0；含 7 帧真实检测丢失 |
| 点位显示修复及 60 fps 请求 | `host-e178f09712cc4374ad34bd3938b2326c/device-report.json`、`corrected-live-overlay.png` | 完整网格贴合对应帧；60 请求协商 30，解码约 28.43 fps；不称 60 fps 实现 |
| 实际 Qt 工作台/设备/权限/文件桥 | `qt-product-df17fac3c80642769f613f6c1d5fe64b/report.json` | Qt 6.11.2 / Chromium 140；正式任务、完整保存、三种真实采集均通过 |
| 原生磁盘采集适配 | `native-device-20260927-02/capture.json`、`decode-check.json` | 178/178 视频帧回读，时间递增；约 29.75 fps，队列峰值约 2.77 MB，无队列拒绝；真实麦克风非静音信号，ADC 时间不可用 |
| 5 分钟公开长片段 | `long-1a372aff4e5f4346926dc597f6b97b04` | 1500 解码帧完整处理，约 14.86 s，峰值内存 994217984 B，采样临时磁盘峰值 178090407 B |
| 640 输入 + 1080 MP4 | `quality-f0ab221e61db4d10968b1c358fac560f` | 本地 2 GiB 进程预算通过，峰值 1115815936 B，约 2.046 s；先前 1 GiB 失败保留，未扩大服务器预算 |
| 容器旋转 | `rotation-69f09449fc3f4de6b1e673cc8015cf8d` | V2 OpenCV 自动应用旋转；90 度样例 RGB 与 V3 精确一致，冻结 source/pixel hash |

Chrome 默认输入与 Qt 默认输入均为立体声混音，音轨幅度极低，不能据此宣称自然语音同步通过。原生适配明确选择麦克风阵列后，8 秒测试获得 297600 个音频采样，峰值约 0.7165、RMS 约 0.1731；这证明取得非静音输入，不证明内容/声学延迟正确。

## 用户指出的问题与修复

1. **点位显示**：原实现 `_draw_overlay` 画 TESSELATION∪CONTOURS 和全部 478 点，V3 初版只画三个指标轮廓。异步点位还叠在更新的 live video 上，停止后可能把画布重置为 640×480。现恢复完整拓扑，显示实际送入模型的同一图像，尺寸逐帧保留。加入独立 oracle 拓扑测试、真实 Worker 人为延迟下输入图像与显示图像逐像素对应测试、停止保持 512×512 回归。测量索引/公式未变。
2. **Qt `Invalid state`（首次诊断，后续补充见下节）**：两次实机验收脚本用 `QMessageBox.done(Yes)` 关闭对话框，静态 question 未取得 clickedButton，产品收到拒绝。记录为 `MediaAudioVideoCapture / armed=true / denied_user`。修正为实际点击 Yes 按钮后，三模式成功；但此前没有覆盖同页拒绝后重试，不能据此认定所有 `Invalid state` 已解决。
3. **原生 ADC=0**：WDM-KS 驱动报告 `inputBufferAdcTime=0`，旧适配把负映射截成重复 0 PTS，第一次失败。修正为记录 ADC 不可用和显式估计锚点，保留样本时钟及原始回调时间；绝不标记精确同步。第一次失败目录 `native-device-20260927-01` 保留。
4. **其他工程修复**：Qt CJS MIME 别名、输入分块 HTTP 大小限制、Windows 状态文件原子替换竞争、保存目标不存在时的路径检查、MP4 edit list 时间基的音频 offset 量化，均保留原失败证据与复验。

## 资源、时间与隐私

- 固定版本资源和 SHA256 在 `resources/m05/resources.json`、`legacy-runtime.json`、公共资源清单。模型本机静态安装约 23 MB。页面懒加载，独立 Worker 单帧在途。
- Chrome 缓存测试 `cache-report.json`：冷启动约 131 ms、热启动约 118 ms；部分资源重验证，WASM仍重新读取。已初始化 Worker 断网后可继续，新 Worker 断网启动失败，因此未宣称网页有持久离线安装能力。Qt 使用本地 scheme 静态资源，不依赖此缓存。
- 请求 GPU、独立 WebGL renderer 和模型实际硬件加速分别记录；成功请求 delegate 不当作 GPU 性能证明。没有固定 FPS 承诺。
- 原始录制、处理帧清单、时间戳、检测有效性、展示缺口、推理跳过数量各自保留。实际 60 fps 请求降为 30 fps、解码约 28 fps 是观测结果。推理遗漏不冒充新测量。
- 离线输入页面128 MB/CLI 1 GB，输出512 MB，子进程2 GiB/30分钟，临时700 MB；逐帧流式处理。原始缺失与兼容补全分开。浏览器128 MB媒体/32 MB候选记录，存储估计不代表下载磁盘空间。
- 原视频、V2、现有语料/旧包未改，私有视频没有上传。公共素材来源和条件为 NASA/scikit-image 固定资源；私有录制来源由本轮明确授权记录在本地输出。

## 命令与尚未通过的门

```powershell
$env:PYTHONPATH='D:/PhoneticToolbox/PhoneticToolbox_v3/backend/src;D:/PhoneticToolbox/PhoneticToolbox_v3/packages/phonetic_core/src;D:/PhoneticToolbox/PhoneticToolbox_v3/desktop/src'
& '.venv/m05/Scripts/python.exe' -m pytest -c tests/pytest.ini tests/parity/test_lip_extraction.py backend/tests/test_m05_video.py backend/tests/test_m05_exports.py -q
node --test frontend/tests/m05.test.ts
npm --prefix frontend run typecheck
npm --prefix frontend run build
node tests/e2e/m05.cjs
& '.venv/m09-ui/Scripts/python.exe' -X utf8 scripts/verify_m05_wiring.py
& '.venv/m09-ui/Scripts/python.exe' -X utf8 scripts/verify_m05_product_qt.py --devices
```

物理设备命令会短时打开摄像头/音频，只能在明确授权下运行。不是每次文档检查都应重新录制。

| 平台/项目 | 状态 |
| --- | --- |
| Windows Chrome 与实际 Qt 本机采集/任务/导出 | 上述限定范围通过 |
| 自然素材完整侧脸、快速开合、遮挡分段标注与真值 | 有授权实拍和失检证据，尚无逐场景人工金标准 |
| 声学音唇延迟、长时间漂移与传感器实际掉帧 | 未校准；容器 PTS、样本时钟、合成 offset 编码验证不能替代物理测量 |
| 设备物理拔出、长时间后台/睡眠、极限长录制、强杀断电 | 部分逻辑保护和受控测试，完整实机矩阵待验 |
| WSL / 实际 Linux 科学任务 | 只读环境核查：已有 NInfer/P11 Python 3.11.14 缺 numpy/mediapipe/av，未安装或改系统；未准入 |
| 服务器小规格、远程实验室节点 | M05 能力关闭，不宣称自动回退；离线后续复用公共准入规则，摄像头不跨机接管 |
| Firefox/Safari/其他设备/GPU | 未验证 |
| EXE/发布/生产/数据库迁移 | 未执行 |

公共构建保留其他模块的大 chunk 警告。文档全检已有 README 历史 M10-R5 EXE 缺失链接，未为通过检查删除历史内容。完整 M05 继续 in_progress，不能把一次实机采集成功写成全面稳定。

## 最终回归记录

- M05 Python：30 项通过，含旧指标、防抖、旋转、ADC 缺失、原 V2 offset 建议、时间轴和六种动画导出回读。公共任务/本地文件/契约定向回归：24 项通过。
- 前端完整单元集：173 项通过；类型检查、生成契约、UI 数据检查通过。生产构建通过，其他模块大 chunk 警告保留。
- 最新真实 API 重跑：`wiring-09acfdce244f4965a7ff8b2fa7db8da2` 与 `wiring-dd18a12d5f7f41eaaac488e2bc59bce6` 均通过。较早 `wiring-7c3adfc7a8ad4be9bbff66fa623bd728` 出现一次子进程已成功、发布失败的泛化错误，原记录缺少堆栈，尚不能确定原因；现已增加受控本地诊断，后续三次均通过，不用复跑替代对原失败的解释。
- 原生麦克风录制经正式 CLI 全部 178 帧分析、音频解码、V2 偏移建议和带音频 MP4 导出回读通过，见 `native-device-analysis-20260927-01/export-readback.json`。建议只显示供人工决定，没有自动应用。
- 静态架构检查和契约快照零漂移，`git diff --check` 无错误（仓库已有 LF/CRLF 提示）。文档全检仅保留前述历史 M10-R5 EXE 缺失链接。原 V2 五份源码/手册 hash 复核未变。
- WSL 检查是环境缺依赖事实，不是 Linux 验收成功。上述 private-provenance 清单和 hash 随本地实机输出保存，禁止将真实视频当公共 fixture。

## 2026-09-27 追加：拒绝后反复 Invalid state

用户再次提供实时预览启动失败截图。补测在 `qt-product-2cdf53e478b6490399a4d816853f2583/report.json` 复现：第一次请求为 `MediaVideoCapture / denied_user`；同页第二次重新开始没有触发 `permissionRequested`（events 为空），直接在 getUserMedia 返回 Invalid state。原修复只改了提示和测试对话框，漏掉 Qt 当前页面仍保存的媒体拒绝状态。

`m05_permissions.py` 现保存本窗口、本站点、本模块三类媒体权限对象，在下一次显式开始时对仍有效的对象调用 `reset()`，随后照常询问。过期门控的本源拒绝也可重试；其他站点/非媒体权限不重置，不自动 grant。依据 [Qt QWebEnginePermission 文档](https://doc.qt.io/qt-6/qwebenginepermission.html)：媒体权限不跨会话持久化，但对象在当前页面仍有效，reset 恢复询问状态。

验证命令（需前文 PYTHONPATH）：

```powershell
& '.venv/m09-ui/Scripts/python.exe' -X utf8 scripts/verify_m05_product_qt.py --devices --permission-retry --hidden
& '.venv/m09-ui/Scripts/python.exe' -m unittest discover -s desktop/tests -p test_m05_permissions.py -v
```

实际 Qt 6.11.2 验证 `qt-product-c5fb1cfed2954746ab2ca3e222416912/report.json` 通过：拒绝两次均重新询问，随后两次允许预览/停止，之后三种录制完成编码与本地写盘。4 项单元回归通过，包括过期门控、其他来源/权限不受影响、失效对象不重置。git diff --check 无错误（已有换行提示保留）。

为避免干扰用户，最后复验隐藏主窗口，并在测试进程中注入 Yes/No 对话框返回值；实际 getUserMedia、Qt 权限对象、设备与编码器未替换。隐藏窗口呈现计数为 0，**本轮只验证授权恢复与录制链路，不证明可见预览帧率或推理输出**。所有测试实例已退出。较早 `qt-product-090dad214abf41b99d73777922e56103` 因遗漏子进程 PYTHONPATH 在握手阶段失败，`qt-product-0a53d15917ec43f2862ccc21dc109e29` 因测试途中模式变化未完成预期序列，均保留且不作为通过证据。

修复在 Python 桌面适配层，现有已启动窗口不会热更新；需重新启动 M05 开发入口。未更新旧 EXE，未改模型/科学公式或权限安全策略。
