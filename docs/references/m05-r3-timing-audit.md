# M05-R3 音频、视频与唇形时间审计

2026-10-04。结论：默认 offset=0 不能当作物理同步已经校准。软件中确定的回放接线、时间基准叠加及异常视频保存问题已修复，实体相机/麦克风的延迟和漂移仍未测量。

## 三类时间必须分开

1. `requestVideoFrameCallback.mediaTime` 是送到显示管线的视频帧 PTS。`presentationTime` 是提交合成时间，`captureTime` 是可选采集时间。该接口的显示回调属于尽力同步，可能晚一个显示刷新周期，不能拿回调抵达时间当传感器曝光时间。来源：[WICG requestVideoFrameCallback 草案](https://wicg.github.io/video-rvfc/)，查阅于本日。
2. MediaRecorder 将同一 MediaStream 的音视频编码到一个容器。`start()` 调用、异步编码开始、第一视频帧与第一音频包不具有相同零点的接口保证。`BlobEvent.timecode` 描述分块时间，不能替代每帧的采集时刻。来源：[W3C MediaStream Recording](https://www.w3.org/TR/mediastream-recording/)，查阅于本日。
3. 唇形参数带有模型及滤波的时间响应。Web Face Landmarker 的 VIDEO 模式、numFaces=1 内部平滑与本项目额外防抖是不同层次。Google 文档说明 numFaces=1 启用平滑，但没有给出可直接减去的固定延迟值。来源：[Google Face Landmarker Web guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/web_js)，查阅于本日。额外防抖代码沿用 V2 的自适应增益与门限，不能用一个固定毫秒数概括所有轨迹。

## 确认并修复的问题

| 问题 | 旧行为及影响 | 本轮处理 |
| --- | --- | --- |
| 录制回放未连接动画 | 新录制播放器没有 play/pause/seek/逐帧回调，中间保留最后一次实时结果 | 录制与离线结果均由实际播放器时钟驱动完整帧回放 |
| 候选时间零点混用 | `time_s = mediaTime − start 时观测的 video.currentTime`，导出又加 `first_video_pts` | 新记录标记 `candidate_time_base=recording_start_estimate`，导出使用 `time_s − first_audio_pts`，不再重复加视频首帧时间。旧无标记记录保留旧解释 |
| 离线源视频回放多加首帧 | 离线帧时间已经是容器 PTS 减 anchor，播放器原始时间又加 first_video | 使用 `video.currentTime − anchor − manual_offset`，反向定位使用 `row.time_s + anchor + manual_offset` |
| 当前录制偏移不能保存 | 原实时采集元数据将人工偏移固定成 0 | 分别保存实时录制与各个离线结果的偏移，回放/交换文件/动画一致，原始帧时间不改 |
| 未选视频仍依赖视频转码 | 即使只要音频/唇形，仍会先转全部视频，视频 PTS 异常阻断全部保存 | 未选视频跳过视频编码。重复/倒退 PTS 视频按原容器逐字节保存并提示，不重写时间戳。正式离线分析的严格时间检查仍保留 |

独立数值例：视频首帧 0.100 s，音频首采样 0.050 s，候选帧时间为 0.100/0.230/0.470 s。新交换文件的音频相对时间为 0.050/0.180/0.420 s。人工偏移 +0.123 s 仅存入 metadata，读取时加一次。原算法会再多加 0.100 s。此例由独立合成 PTS/PCM 构造和文件回读测试确认。

## 浏览器独立时间实验

输入是 64×64 黑白二进制帧编号和 AudioContext 音频脉冲。真实 Chrome MediaRecorder 编码，独立 PyAV 解码识别画面编号，按相同编号比较候选时间和编码视频 PTS。推理输出是实验替身，用 0/37/95 ms 等待隔离推理耗时；这项实验不测试 Face Landmarker 的测量质量。

| 初始化等待 / 推理等待 | 无歧义匹配帧 | 候选时间减编码视频 PTS，最小 / 中位 / 最大 |
| --- | ---: | --- |
| 0 / 37 ms（修复前） | 30 | 11.666 / 12.046 / 12.611 ms |
| 500 / 95 ms | 18 | 11.404 / 11.834 / 12.261 ms |
| 1200 / 0 ms | 72 | 11.562 / 12.089 / 12.529 ms |

证据：`output/validation/m05-r3/clock-audit.json` 及其中三份 `independent-clock-audit.json`。脚本为 `tests/e2e/m05-clock.cjs`、`scripts/audit_m05_r3_clocks.py`。

这确认当前起点观测仍有约 12 ms 的软件误差。它没有随本实验中的推理等待增加到 95 ms，符合给输入帧记时、推理结束后保持原时间戳的实现。实验没有证明该误差在别的设备、浏览器、帧率或长录制下恒定，代码未写死 −12 ms 补偿。新元数据将起点误差记为 unknown，physical_sync_verified=false。

音频脉冲也已解码回读，但 AudioContext 调度与 Canvas 定时器本身具有不同延迟，不能把二者的事件差直接视为硬件校准值。第一次使用低亮度红色编码的试验受有损压缩影响，无法可靠识别编号，未用于上述结论，原文件保留。

## 音频和滤波核查

- 独立 WAV 保留解码 PCM 样本。原有 `decoded_samples` 策略忽略包时间戳的小幅抖动，不插入或丢弃样本。现在额外记录最大及末端 PTS/样本钟残差。该残差不等于物理设备漂移，`drift_s` 仍为 null。若源音轨真的存在大间断，这一策略不构成对间断的同步修复。
- 原始视频中的音频、单独 WAV 和动画中的音频有不同容器/编码。MP4 中的 AAC 是派生编码，不能声称与 WAV 逐样本相同。WAV 的已知 PCM 测试逐值一致，视频 PTS 和动画帧数/音轨分别解码核查。
- 指标公式、点索引与额外防抖未改。冻结 V2 基线验证仍通过。`open` 是带符号的点 14−点 13 纵向距离，经面高归一化，小负值可能出现，未强行截成 0。
- 面部放大、镜像、参数比较图的归一化都只影响显示。动画导出按 30 fps 重采样，未增加科研测量帧；实时 Face Landmarker 保持 candidate 来源身份。

## 仍需设备校准的部分

使用与研究相同的相机、麦克风、分辨率、帧率和浏览器录制可同时见到与听到的校准事件，并在开头、中段和结尾重复，才能估计固定偏差与漂移。1 ms 调整步长只代表编辑精度，不代表 30 fps 视频具有 1 ms 测量分辨率。发声起点与唇部运动本身可能不同步，不宜仅凭两条曲线相似就判定设备延迟。此次没有启动实体设备，也没有读取截图对应的个人录制。
