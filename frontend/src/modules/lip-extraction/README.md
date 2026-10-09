# 唇形提取 · M05

本目录提供摄像头采集、候选唇形预览、正式已有视频分析、同步回放与人工偏移保存。名称和持久模块 ID 见[模块注册表](../../app/registry.ts)。完整逐控件正文维护在[可编辑说明书章节](../../../../manual/chapters/m05.json)，既有开发说明见[模块说明](../../../../docs/manual/lip-extraction.md)。本页依据当前源码与最新 M05/P19 验收资料校订，M05 视频演示由作者另行录制，媒体由主编集中维护。

## 工作流程

- **已有视频离线分析**：支持七类视频扩展名及逐文件批量任务，正式路径按实际解码 PTS 处理完整帧，保留检测缺失。
- **仅采集后处理**：模式选择高帧率先录后算。摄像头与麦克风只采集编码，停止并另存后，明确选择所存视频进行离线分析。
- **边采集边处理**：模式选择实时录制。录制媒体与候选测量同时进行，推理繁忙时可跳帧，记录身份保持 `candidate`。
- **设备准备**：实时预览只请求摄像头，结束清除临时测量，不录音视频且不保存唇形。

## 最短完整流程

1. 选择模式、摄像头和目标麦克风，开始后核对实际音轨名称、电平、画面与网格。实时预览不含音频。
2. 点击结束录制，等待编码及音频时间轴检查收尾。停止不会自动导出，也不会自动提交正式离线计算。
3. 用右侧视频播放器同步回放中央面部，按需打开检查偏移量，比较真实音频波形与唇形参数。
4. 点击另存录制。实时录制默认勾选视频与动画，可四种组合；音频与完整已测唇形固定保存。高帧率模式直接保存视频、WAV 与回执。
5. 选择输出目录，检查保存成功提示及回执。取消或失败保留当前录制，可调整选项重试。
6. 对已有视频或已保存只采集媒体进行离线分析，检查完整帧和缺口，再按当前偏移或零人工偏移另存。已完成任务可从本地历史恢复。

## 控件、输出与限制

| 类型 | 内容 |
| --- | --- |
| 初始采集参数 | 实时预览，摄像头系统默认，自动选择麦克风，30 fps，CPU，防抖开启，截止 15 Hz，镜像开启 |
| 请求范围 | 帧率 1–240 fps；截止频率 1–240 Hz；人工偏移 ±2000 ms，按 1 ms 编辑 |
| 录制输出 | 可选原视频/面部网格动画、PCM16 WAV、实时完整 `capture.m05-preview.json`、预算内交换 `.lip.json`、`recording-export.json` |
| 正式输出 | `measurements.csv`、`legacy-compatibility.csv`、`frames.jsonl`、预览/manifest、可用音轨及音频时间记录、预算内交换文件 |
| 偏移另存 | 原帧不改，安全同名交换文件携带所选偏移，另留未调整文件及 `saved-manifest.json` |
| 可视化输出 | MP4/GIF、1080/720/540 三质量；动画重采样不增加测量帧，GIF 没有声音 |
| 页面预算 | 每视频 128 MB，每批和当前页 100 项；录制媒体 128 MB、候选参数 32 MB、30 分钟；目录导出 512 MB，单份动画 128 MB |

当前目录保存以 Windows 桌面本机适配器为准，远程 M05 尚未开放。M05 与 M16 共用设备占用保护。正式历史读取对应本机分析任务，未另存的录制不能从该历史恢复。

## 科研解释边界

- 浏览器 Face Landmarker 与 legacy Face Mesh 0.10.14 未通过等价门。CPU/GPU 控件用于实时浏览器模型，不改变正式离线的冻结方法身份。
- 面宽/面高为 x/y 投影差，外唇宽为固定轮廓点的最大减最小跨度。`area` 为 `(outer−inner)/face`，`circularity` 为 `4π(outer−inner)/outer_perimeter²`，不能写成真实肌肉、口腔面积或毫米测量。
- `open=(y14−y13)/face_height` 保留有符号值。缺失与兼容补值分开，`imputed` 不能计为新检测。
- 额外防抖与模型内部平滑分开记录。镜像、面部放大、偏移弹窗的各轨归一化只影响显示。
- 人工偏移按 `audio_relative_time + lip_manual_offset` 应用一次，原视频和 WAV 不平移。默认 0 和 1 ms 编辑步长不证明物理同步或设备精度。
- 约 12 ms 的起点差异仅为特定合成时钟实验的证据，不外推实体设备，不硬编码为补偿。请求、协商与实际解码帧率也分别解释。
- 重复/倒退视频 PTS 在录制保存时可保留原容器并提示，正式离线任务继续严格检查时间戳，不制造时间轴。

## 源码与验证入口

| 文件 | 职责 |
| --- | --- |
| [LipExtractionPage.vue](LipExtractionPage.vue)、[port.ts](port.ts) | 模式、结果、保存与平台能力 |
| [capture.ts](capture.ts)、[inference.ts](inference.ts)、[audio-input.ts](audio-input.ts) | 媒体权限、设备、候选推理和输入信号 |
| [metrics.ts](metrics.ts)、[stabilizer.ts](stabilizer.ts)、[overlay.ts](overlay.ts) | 固定指标、额外防抖和网格显示 |
| [OffsetDialog.vue](OffsetDialog.vue)、[alignment.ts](alignment.ts)、[replay.ts](replay.ts) | 实际波形、人工偏移和完整帧回放 |
| [唇形核心](../../../../packages/phonetic_core/src/phonetic_core/lip/) | 正式指标与序列规则 |
| [本机保存适配](../../../../desktop/src/ptb_desktop/m05_bridge.py)、[媒体导出](../../../../backend/src/ptb_worker/m05_media_export.py) | 目录授权、完整写入、编码及回执 |

从仓库根目录使用[源码启动器](../../../../scripts/Start-M05-Workbench.ps1)。公共前端检查为 `npm --prefix frontend run typecheck`、`npm --prefix frontend run test -- --run`、`npm --prefix frontend run build`。定向入口为[采集交互检查](../../../../tests/e2e/m05-r3.cjs)和[Qt 检查](../../../../scripts/verify_m05_r3_qt.py)。

[最新 M05 报告](../../../../docs/testing/2026-10-04-m05-r3-report.md)记录限定 Windows 开发态、合成输入的三模式、四组合保存、回放与文件回读；[最新公共入口报告](../../../../docs/testing/2026-10-05-p19-r14-r16-report.md)记录方法与引用、帮助等入口。报告的界面和成品检查不能扩大为实体摄像头/麦克风同步、自然语料准确性、长录制、Linux 媒体链或跨机器完整发行通过。文档检查只验证结构、来源与链接，不代替产品运行验证。

## 方法与来源

参见[同步审计](../../../../docs/references/m05-r3-timing-audit.md)、[来源映射](../../../../docs/modules/evidence/M05-source-map.md)与[统一来源登记](../../../../third_party/source-registry.json)。官方说明包括 [Face Landmarker Web guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/web_js)、[冻结 Face Mesh 文档](https://github.com/google-ai-edge/mediapipe/blob/v0.10.14/docs/solutions/face_mesh.md)、[逐帧显示回调规范](https://wicg.github.io/video-rvfc/)及[媒体录制规范](https://www.w3.org/TR/mediastream-recording/)。模型、媒体二进制和代码许可按登记的具体对象分别核对。
