# M05 六功能组与科学行为映射

2026-09-27。状态 in_progress；本表的源码事实已只读核对，V3 入口仍按实际验收逐项填充。原 V2 与继承源码均不修改。

| ID / 说明书 | 原源码入口 | V3 文件归属 / 验收 |
| --- | --- | --- |
| F01 / 7.1、7.3 保存位置、相机、音频、刷新 | LipGUI._refresh_devices / _on_camera_changed / _on_audio_device_changed / _select_save_directory | m05 platform + LipExtractionPage；设备稳定 ID、刷新不改选、权限拒绝/拔出、只停止自有流 |
| F02 / 7.1、7.3 实时参数、防抖、停止保存 | _on_video_tick、LandmarkStabilizer、start_recording/stop_recording/_save_recording_files | Worker candidate + 原离线后端；真实处理清单、filter on/off、最后写入失败、后台/切标签 |
| F03 / 7.1、7.4 原视频、高帧率先录后算 | start_raw_recording/stop_raw_recording、_start_hfps_recording/_finish_hfps_recording | 采集编码与推理分离；原始媒体保留，真实时间/模式互斥/停止后 finalize |
| F04 / 7.1 上传离线、批量场景 | upload_video、_recognize_video_frames、_save_offline_recognition | m05 offline/task；流式 PTS 每帧、进度、取消、失检、尺寸/长片段、批量逐项 manifest |
| F05 / 7.4、13.4 offset | LipOffsetAdjustDialog._on_apply/_on_skip，estimate_lip_audio_offset_seconds，io.resolve_lip_time_axis | offset 独立元数据；应用、仅保存 0、取消三个动作；时间只加一次 |
| F06 / 7.1 回放、导出、7.2 与参数估计关联 | LipAnimationDialog、_save_video/_save_gif、_quality_profile、_resampled_landmarks；services.io.lip | 页内回放、受控本地历史转换；1080/20、720/24、540/28(边长/CRF)，GIF 与视频实际解码回读 |

## 不可静默处理的说明书差异

1. 说明书将面宽/面高称为欧氏距离，metrics.py 实为 `abs(x454-x234)` 和 `abs(y152-y10)`。按源码迁移，坐标是二维像素投影。
2. 说明书外唇宽描述为 61/291 差；代码是 16 个外唇点 x 的 max-min。保留代码定义。
3. 说明书称离线失检最多两帧插值、长缺失为 NaN。`_recognize_video_frames` 实际前值保持所有后续失检，前导缺失用首有效结果回填；最多两帧线性插值属于实时 `_save_recording_files`。分开冻结，V3 补全值不能算新观测。
4. 说明书强调真实时间，上传视频代码仍是 frame_index/fps。V3 decoded-pts/1 是明确时间轴修正，与 legacy 基准分开。
5. 原始录制的说明书承诺无实时推理；`_on_video_tick` 只有 high-fps 分支绕开 FaceMesh，raw 模式仍经过推理。V3 采集/推理分离按用户新要求实施并记录。
6. 说明书“自动保存图表”和“面积肌肉+口腔开口面积”的措辞不能当源码事实：保存主路径是 WAV/PKL/timestamps；area 分子为 outer polygon area 减 inner polygon area。指标既非物理面积也非三维生理量。
7. 说明书批量场景不等于旧 GUI 已有多文件选择；当前 upload_video 是单文件。本轮批量是新增入口，复用逐文件规则。

## 固定科学定义

坐标由 lm.x*解码宽、lm.y*解码高转 float32，丢弃 z。外唇/内唇各 16 点，面轮廓 36 点，顺序直接取 metrics.py。无头姿旋转归一化或测量镜像变换。V2 OpenCV 默认自动应用容器旋转元数据，V3 保留 0/90/180/270 度显示旋转，记录编码与有效尺寸；负 open 保留。area=(outer-inner)/face；circularity=4π(outer-inner)/outer_perimeter²；总宽为外宽+内宽。零分母为 NaN，不归零、不裁负面积/开口。

LandmarkStabilizer 默认 min_cutoff=15、beta=.08、d_cutoff=1；全 478 点，FACEMESH_TESSELATION∪CONTOURS 邻接均值、float32、dt 最大 .25、噪声速度 .11*face_scale、运动门 .010/.024*face_scale。不等同单一 Butterworth 低通。不可用简单 EMA 冒充旧防抖。

时间 offset 为 audio_relative_time + lip_manual_offset，metadata 优先 audio_first_frame_time，其次 companion start_time；仅缺 anchor 的旧相对时间归零。原 IO 的有效值插值跨缺失另属于 M01 兼容行为，不能把它当检测成功。

## 证据边界

NASA 公开肖像的平移/遮挡/旋转/分辨率/VFR 派生片段仅为工程压力测试，无自然说话、真实侧脸或真实张口金标准。没有自动开启用户摄像头或读取研究视频。真正设备/自然语料验收需授权素材，不能靠衍生静态照片替代。

## 2026-09-27 产品证据更新

- F01：M05 页面稳定设备 ID 与显式权限，Chrome 真实默认设备已录制；Qt 实机三模式采集和本地保存也已通过。权限拒绝通过受控测试，物理拔出待实测。
- F02：固定本地 Worker，完整 V2 面部拓扑与 478 点；同帧画面叠加、停止保持尺寸回归已通过。浏览器仍为 candidate，不替代正式结果。
- F03：Chrome 三模式原始录制、finalize、原始媒体及候选参数本地写盘通过；60 fps 请求协商为 30 fps，实际解码约 28.43 fps。没有无掉帧承诺。
- F04：真实 HTTP/SQLite/受控子进程/原始视频分块/取消/恢复/逐文件结果与 SHA256 回读通过。公开 5 分钟 1500 帧全段完成。三段明确授权自然录制共 1023 帧的解码 RGB、原始关键点、有效帧指标、mask 与独立 V2 精确一致（filter off）。
- F05：应用 offset、仅保存 0、取消、关闭保护及回读通过；AAC 音频映射的 offset 编码量化修复有精确有理数测试。互相关建议算法原样迁移并接 UI，仅在单声道/600万采样预算内提供显式建议，不自动应用，不宣称生理同步。
- F06：历史任务读取、完整结果保存、实际 GIF 导出 UI 链路通过；MP4/GIF 六种格式×质量组合解码回读通过。原始视频播放依赖浏览器编解码能力，不能播放时有明确替代说明。

最新完整边界见 [M05 报告](../../testing/m05-report.md)。
