# M05 唇形提取 Implementation Plan

**Goal:** 保留六组旧功能，在用户设备上提供有明确方法身份的浏览器实时预览，并完成原检测后端的离线分析路径。
**Architecture:** 纯指标/防抖属于 phonetic_core；JS 是同一固定规范的轻量后端实现。摄像头、媒体编码、解码和写入在 adapter。共享任务、配额、远程准入与 AppShell 串行接线。
**Tech Stack:** 现有 Vue/TypeScript/Qt；legacy MediaPipe 0.10.14；候选 Web tasks-vision 0.10.14、face_landmarker float16/1，本地 JS/WASM/model/hash。独立项目环境，不修改旧环境。
**Status:** in_progress。用户于 2026-09-27 明确授权阶段 A→B→C→D；不重复请求开工。科学等价未证实的 Web 后端保持 candidate。

## 所有权与顺序

- M05 独占 `packages/phonetic_core/src/phonetic_core/lip/`、`frontend/src/modules/lip-extraction/`、`frontend/src/platform/m05.ts`、`backend/src/ptb_worker/m05_*`、`desktop/src/ptb_desktop/m05_*`、M05 专属脚本、测试、报告与资源。
- 不编辑 M11 文件。2026-09-27 已告知 M11 先占用共享文件，M05 接线前等待释放并重读差异。
- A/B 阶段公共文件不改，C/D 已在 M11 释放后串行接线：AppShell、main API、job models/executor、host/task_bridge、生成契约、source registry、台账。后续每项串行合并，禁止覆盖现有改动。
- 无 push/发布/生产部署/现存库 DDL/系统或全局依赖变化/V2 写入/用户语料写入/主 EXE。

## A 独立基准及规范

1. 读取 V2 说明书 7.1–7.4 和实际 metrics、GUI、service、IO。写 `docs/modules/evidence/M05-source-map.md`，区分说明书差异、当前入口和验证。
2. `scripts/m05_prepare_fixtures.py` 生成有来源的公开图像派生测试视频及解析关键点序列；保存原素材、输入、解码帧与来源哈希。派生视频只作工程测试，不能替代自然说话、侧脸、真实快速开合唇验收。
3. `scripts/capture_m05_v2.py` 由原 V2 解释器只读加载原 metrics 和 GUI 中独立类/函数，双轮捕获，绝不加载 V3 expected。显式记录 legacy 离线填充与实时短缺失插值差异。
4. 迁移 `metrics.py`、`stabilizer.py`，原样迁移和新协议/错误处理分开文件。冻结 `spec.json` 索引与配置；Python/JS 对同一份规范交叉验证。

## B 浏览器可行性与预设比较门

1. 资源从官方 npm 固定 tarball、官方 model revision 1 获得，清单记录 SHA256；运行不访问 CDN/latest，模块打开时才加载。
2. `worker.js` 用 Dedicated Worker 和显式 CPU/GPU delegate；记录初始化/推理失败，禁止静默回到主线程。一次至多一帧 in-flight，不排无界队列。
3. 同一 PNG 解码 RGB 输入序列分别喂给原 FaceMesh 与 Web IMAGE/VIDEO，比较原始 landmarks、原指标、过滤后指标、mask 和时间。VIDEO 单脸自身平滑与旧防抖分别开关/报告。
4. **冻结门槛（测前）：** Python 原样迁移同平台同输入逐值/NaN mask 精确一致；JS 同输入坐标先 float32 对齐，指标有限值 `abs_error <= max(1e-6, abs(expected)*2e-6)`，mask/字段精确一致。这个阈值只覆盖 float32/求和顺序，不能用于新旧检测模型。
5. **检测等价门：** 同帧检测有效性与时间/处理帧清单精确一致；关键点及原指标按同输入迁移容差检查。没有实验论证的更宽模型容差。本门失败即 candidate，不修改阈值、不阻断 C/D 独立功能。
6. Chrome 与 Qt 分别记录 Worker/WASM/WebGL/CPU/GPU、耗时、RSS/浏览器可观测内存、主线程心跳、资源请求/缓存。冷/热加载与断网本地资源分开。

## C 离线核心及任务

1. 流式解码逐帧 PTS；非单调或无 PTS 拒绝正式结果，禁止 frame_index/fps 伪造时间。保留编码时间基、音频起点、旋转处理、尺寸和输入 hash。
2. `legacy-facemesh/0.10.14 + lip-metrics-v2/1` 保留公式、防抖；真实 PTS 时间语义 `decoded-pts/1` 与旧 `frame-index/fps` 对照单列。
3. 每帧记录 detected、imputed、reason、raw/processed；旧补全值只作明确兼容轨，不算新观测。JSONL/CSV 流式输出，限尺寸/帧数/输出/运行时，取消/失败不发布成功 manifest。
4. offset 存一次，`audio_relative_time + lip_manual_offset`，应用/仅保存(0)/取消分开，原始记录不可覆盖。
5. 复用公共任务、文件身份、预留、fencing、最后完整发布。本地未上传视频不计云端配额；上传必须用户显式选择。远程仅离线任务，能力未通过不开放。

## D 页面、设备、回放及收口

1. 公共 ModuleFrame/Toolbar/Section/Status、字体、单一播放与关闭保护。设备刷新保留 ID，用户点击才请求媒体；镜像只改变 CSS 预览。
2. 三模式：实时候选预览；原始录制；先录后算。MediaRecorder 与推理独立，采集/编码/推理/显示的可观测率分列，未知不编造。
3. 有界存储与显式保存流程。stop→finalizing→saved，写入/下载失败可重试。后台节流、切标签、关页、权限拒绝、拔出测试。
4. 已有视频/批量、结果回读、offset、动画、MP4/GIF 与 1080/720/540 质量。设备能力不可用时给出原因及实际桌面路径，不删功能。
5. Windows Chrome/实际 Qt、Linux、其他浏览器/GPU/真实设备/实验室节点单列状态。

## 验收命令（脚本完成后执行，当前不表示通过）

```powershell
& '.venv/m05/Scripts/python.exe' -X utf8 scripts/m05_prepare_fixtures.py
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/capture_m05_v2.py
& '.venv/m05/Scripts/python.exe' -m pytest -c tests/pytest.ini tests/parity/test_lip_extraction.py -q
node --test frontend/tests/m05.test.ts
npm --prefix frontend run typecheck
npm --prefix frontend run build
node tests/e2e/m05.cjs
& '.venv/m09-ui/Scripts/python.exe' -X utf8 scripts/verify_m05_qt.py
```

报告入口：`docs/testing/m05-baseline-report.md`、`m05-browser-report.md`、`m05-report.md`；用户说明 `docs/manual/lip-extraction.md`。证据放 `output/validation/m05/`。记录失败与未测，不扩大 verified。
