# M05 说明书实时预览补充

`M05-MANUAL-LIVE` 当前为 `verified`，范围限定为唇形说明书的预览文字、作者提供的完整演示录屏、生成阅读资源及 Windows 开发环境播放。本次不修改模型、测量算法、产品 UI 或其他模块，冻结 EXE 未更新。

## 已补文字

- `m05-preview-steps`：解释预览中阶段、真人画面、候选网格、九项数值、四条曲线及未检出反馈。
- `m05-status-table`：补预览中状态与预览模式编码大小为 0.0 MB 的含义。
- `m05-replay-steps`、`m05-playback-controls-table`：说明本次实时录制可直接回放，候选已测帧与正式完整解码帧分别核对。
- `m05-high-framerate-replay-note`：说明只采集模式先回放媒体，网格待离线分析；预览结束清除临时测量。
- 工程仅更新 M05 搜索条目，保留其他章节、素材及同期修改。保留 `m05-video-demo` 锚点，将标题改为实时预览演示，加入 `m05-live-preview-video` 节点及实际视频引用。

对照了 V2 第 7 章本机只读 HTML 切片、当前 `LipExtractionPage.vue`、`capture.ts`、`replay.ts`、模块 README 和 M05-R3 报告。沿用 V3 的三模式、停止与另存分开、候选/正式模型身份，不恢复 V2 的自动停止保存或自动偏移估算表述。

## 实际采集与素材状态

Windows 当前 Startup-20261008-R1 应用中，作者允许采集并亲手接受权限弹窗后，实体摄像头实时预览显示 1280×720、协商 30 fps，中央网格、参数与曲线可见变化。麦克风未启动。结束预览后画面清空，另存录制不可用。

第一次使用 FFmpeg `gdigrab` 对窗口录制了 20 秒。文件可解码，但 2、10、18 秒的检查画面没有连续更新，因此不作为动态演示，不加入正式素材登记。诊断原件与截图留在本机忽略的作者目录。该次失败不能据此判断产品摄像头预览失效，也不证明原因已修复。

作者随后主动停止 Computer Use，改为自行录制并发送素材。此后未继续控制窗口或启动设备。最终收到作者提供的屏幕录制，最新明确要求为完整放入，无须剪辑。最终视频保留权限确认、真人预览、网格和曲线变化，以及原有音轨内容；未剪时间、裁画面、缩分辨率或改帧率。桌面原件保持不变。

源视频为 H.264/AAC MP4，4,016,476 字节、400 帧、1920×1080、30 fps、13.333333 秒。先压缩 MP4 后，Qt WebEngine 6.11.2 实测返回 `DEMUXER_ERROR_NO_SUPPORTED_STREAMS`，改为 WebM VP9/Opus 后播放通过。正式素材为 `assets/software-only/m05-live-20261010/m05-live-preview.webm`，2,525,223 字节，减少约 37.1%，仍为 400 帧、1920×1080、30 fps；WebM 容器时长精度为毫秒，登记 13.333 秒。采用 VP9 CRF 34、Opus 32 kb/s，未使用时间截取、裁剪或缩放参数。抽查压缩后动态画面，网格、数值和四条曲线可辨认。

素材标记 `distribution: software-only` / `git: false`，SHA-256 为 `25c5a4bd308b917c106c68e7e2398d2623c523d9601cde691c32ee2b67d3f17a`。原始私有路径与转码记录仅存本机忽略的作者目录；MP4 编码检查文件也移入该目录，不参与正式阅读资源分发。真人媒体未进入 Git，也未上传外部服务。

## 验证与未完成项

最终工程严格结构/媒体摘要校验通过：20 章、954 个素材，零错误、零警告。`scripts/manual/build.py` 生成软件分发阅读资源通过，`npm --prefix frontend run build` 重建 `frontend/dist` 通过，保留既有大分块提示。M05 搜索条目保持既有锚点规则，其他章节搜索与素材不因本次修订改变，同期 M10 改动保留。

- FFprobe 核对全部 400 帧、尺寸、帧率及媒体时长，FFmpeg 全片解码无错误。
- 隔离的 Qt WebEngine 页面使用当前桌面 `Assets` 处理器和 `ptbapp` 地址读取最终 WebM。实际播放时间前进，拖至末段后正常触发 ended，无媒体错误。测试关闭用户手势限制以驱动离屏播放，未改产品设置，不访问摄像头、麦克风或应用数据库。
- 独立 Chrome 进程读取本次 `frontend/dist`，从使用说明进入唇形提取与原有小节锚点。视频渲染、播放、暂停、跳至 10.8 秒、继续播放到结尾均通过；搜索真人实时预览可定位新视频锚点。查看实际阅读器截图，标题、视频、图注和下载入口可读。
- 详细媒体、Qt 和浏览器回执及页面截图保留在本机忽略的 `output/manual-work/m05-provided-video/`。
- 最终复核桌面原件 SHA-256 未变，源工程、生成阅读资源及 `frontend/dist` 的正式视频字节与登记摘要一致。

实际命令：

```powershell
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/manual/validate.py --project manual --strict --report output/manual-work/m05-provided-video/manual-validation.json
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/manual/build.py --project manual --output frontend/public/manual --distribution software
npm --prefix frontend run build
ffmpeg -v error -i manual/assets/software-only/m05-live-20261010/m05-live-preview.webm -f null -
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 output/manual-work/m05-provided-video/qt-media.py
node output/manual-work/m05-provided-video/browser-check.cjs
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/validate_docs.py
git diff --check -- README.md manual/chapters/m05.json manual/project.json manual/README.md manual/AUTHORING.md manual/V2_STYLE_GUIDE.md frontend/src/modules/lip-extraction/README.md docs/plans/task-ledger.json docs/project-status.md docs/testing/m05-manual-live-report.md
```

最终文档核查 1811 个文件、零当前错误，限定改动文件的差异格式检查通过；历史快照旧链接独立列示。本次说明书补充已完成，现存冻结 EXE 尚不包含该视频和文字，下次打包需使用更新的阅读资源。完整 EXE、实体音视频同步、长录制、自然语料测量准确性和其他平台未在本次复验范围内。
