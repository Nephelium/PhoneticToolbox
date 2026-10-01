# P17 M05 唇形提取逐项验收规则

状态：限定 Windows 已执行，逐行状态见下表，未验范围保留。基线 0779bee；执行记录见同目录 M05-report.md。

已读原始手册 `../PhoneticToolbox_v2/Phonetic_Export/index.html` 第 7.1–7.4 节，及实际源码 `phonetic_toolbox/gui/widgets/lip_gui.py`。V2 源码 SHA-256：`6656012fb4469466d3a41ec9eb36c139be63b6cf058906b8d3bcf6dd7e6fbc79`。

现行P17要求优先于旧手册：1920×1000 CSS主工作区常用操作单页，普通滚轮不推动整页，小窗保持可达；波形连续、自适应可见振幅。V2原始目录自动保存改为受控显式保存。历史通过不算本轮实测。

## 可复现功能门

| ID | 手册 / V2定位 | V3控件及前置状态、步骤和判据 | 证据 / 状态 |
| --- | --- | --- | --- |
| M05-F01 设备与模式 | 7.1–7.4 / `lip_gui.py` | 刷新设备，摄像头/麦克风/帧率/CPU-GPU/四模式逐一选择；设备缺失明确错误不伪造预览 | partial · X03软件选择/刷新已验；真实设备选择 user-deferred · 井井明确将设备采集留给 EXE 人工验收 |
| M05-F02 防抖与镜像 | 7.1–7.4 / `lip_gui.py` | 开关防抖、截止频率、镜像；只改显示/指定滤波，记录元数据；不能承诺物理零延迟 | partial · X03开关/频率字段已验；录制元数据 user-deferred · 井井明确将设备采集留给 EXE 人工验收 |
| M05-F03 预览启动/停止 | 7.1–7.4 / `lip_gui.py` | 开预览→停止并收尾→重开；释放所属设备，三模式录制中设备设置锁定 | user-deferred · 井井明确将设备采集留给 EXE 人工验收 |
| M05-F04 三模式录制 | 7.1–7.4 / `lip_gui.py` | 实时参数/原始视频/高帧率先录后算，逐模式开始停止并保存；设备采集依用户明确要求 user-deferred | user-deferred · 井井明确将设备采集留给 EXE 人工验收 |
| M05-F05 保存与放弃 | 7.1–7.4 / `lip_gui.py` | 原始录制、候选参数记录、正式离线分析、核对下载、放弃/取消；未保存保护不能误清已存结果 | user-deferred · 井井明确将设备采集留给 EXE 人工验收 |
| M05-F06 真实离线视频 | 7.1–7.4 / `lip_gui.py` | 从批准目录选择真实视频、多选、分析/取消；逐帧真实时间、缺失/补全计数及参数语义与V2一致 | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-F07 历史 | 7.1–7.4 / `lip_gui.py` | 刷新历史/选择/读取；空结果、失败、重复读取、切换时迟到结果不覆盖 | partial · 最终Qt空历史刷新按钮通过；历史结果读取 blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-F08 结果与偏移 | 7.1–7.4 / `lip_gui.py` | 结果选择、播放/暂停、帧滑条、offset边界、填建议、应用并保存/无偏移保存/取消；保存偏移准确且不改原视频 | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-F09 动画导出 | 7.1–7.4 / `lip_gui.py` | 1080/720/540、MP4/GIF，回读帧/时长/尺寸，取消保留旧文件 | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-F10 真实同步 | 7.1–7.4 / `lip_gui.py` | 检查音视频采集时序/首次PTS/人工唇动对照；软件时间戳不能充当物理同步验证，本轮 user-deferred | user-deferred · 井井明确将设备采集留给 EXE 人工验收 |

## 每个实际控件的清单

下表为初次审阅时逐模板控件索引，行号可能随最终布局移动；控件名与绑定为稳定定位。每种普通控件检验正常路径及关键失败/取消，额外数值组合不等同于新的未测控件。动态v-for控件逐文件/参数/任务实例应用同一判据。共享WaveformViewport、AudioTransport、TaskPanel及参数抽屉按总规则复用，不能据模板存在判通过。

| ID | V3源文件/行 | 控件及绑定 | 实际状态 |
| --- | --- | --- | --- |
| M05-C001 | `LipExtractionPage.vue:76` | `<select v-model="mode" :disabled="recording"><option value="preview">实时预览（候选）` | verified · X03 四模式/数值上下界/开关/枚举/帮助 |
| M05-C002 | `LipExtractionPage.vue:76` | `<button :disabled="recording&#124;&#124;busy&#124;&#124;dirty" @click="start">{{mode==='preview'?'打开预览':'开始录制'}}` | user-deferred · 井井明确将设备采集留给 EXE 人工验收；可见空态保存/停止/放弃禁用已验 |
| M05-C003 | `LipExtractionPage.vue:76` | `<button :disabled="!state&#124;&#124;!['opening','previewing','recording'].includes(state.phase)" @click="stop">停止并收尾` | user-deferred · 井井明确将设备采集留给 EXE 人工验收；可见空态保存/停止/放弃禁用已验 |
| M05-C004 | `LipExtractionPage.vue:76` | `<button @click="help=!help">帮助` | verified · X03 四模式/数值上下界/开关/枚举/帮助 |
| M05-C005 | `LipExtractionPage.vue:76` | `<button @click="emit('references')">方法与引用` | verified · X07 |
| M05-C006 | `LipExtractionPage.vue:82` | `<select v-model="camera" :disabled="recording"><option value="">系统默认` | user-deferred · 井井明确将设备采集留给 EXE 人工验收；可见空态保存/停止/放弃禁用已验 |
| M05-C007 | `LipExtractionPage.vue:83` | `<select v-model="microphone" :disabled="recording"><option value="">系统默认` | user-deferred · 井井明确将设备采集留给 EXE 人工验收；可见空态保存/停止/放弃禁用已验 |
| M05-C008 | `LipExtractionPage.vue:84` | `<button :disabled="recording" @click="refresh">刷新设备` | verified · X03 四模式/数值上下界/开关/枚举/帮助 |
| M05-C009 | `LipExtractionPage.vue:84` | `<input v-model.number="fps" type="number" min="1" max="240" :disabled="recording"/> fps` | verified · X03 四模式/数值上下界/开关/枚举/帮助 |
| M05-C010 | `LipExtractionPage.vue:85` | `<input v-model="filter" type="checkbox" :disabled="recording"/>特征点防抖` | verified · X03 四模式/数值上下界/开关/枚举/帮助 |
| M05-C011 | `LipExtractionPage.vue:85` | `<input v-model.number="cutoff" type="number" min="1" max="240" :disabled="recording"/> Hz` | verified · X03 四模式/数值上下界/开关/枚举/帮助 |
| M05-C012 | `LipExtractionPage.vue:86` | `<select v-model="delegate" :disabled="recording"><option>CPU` | verified · X03 四模式/数值上下界/开关/枚举/帮助 |
| M05-C013 | `LipExtractionPage.vue:86` | `<input v-model="mirror" type="checkbox"/>镜像预览` | verified · X03 四模式/数值上下界/开关/枚举/帮助 |
| M05-C014 | `LipExtractionPage.vue:94` | `<button :disabled="!['ready','failed'].includes(state?.phase??'')&#124;&#124;!state?.dirty" @click="saveRecording">保存原始录制` | user-deferred · 井井明确将设备采集留给 EXE 人工验收；可见空态保存/停止/放弃禁用已验 |
| M05-C015 | `LipExtractionPage.vue:94` | `<button :disabled="!['ready','failed'].includes(state?.phase??'')" @click="saveMetadata">保存候选参数与时间记录` | user-deferred · 井井明确将设备采集留给 EXE 人工验收；可见空态保存/停止/放弃禁用已验 |
| M05-C016 | `LipExtractionPage.vue:94` | `<button :disabled="state?.phase!=='ready'&#124;&#124;!canAnalyze&#124;&#124;busy" @click="analyzeRecording">对录制做正式离线分析` | user-deferred · 井井明确将设备采集留给 EXE 人工验收；可见空态保存/停止/放弃禁用已验 |
| M05-C017 | `LipExtractionPage.vue:94` | `<button v-if="(mediaRequested&#124;&#124;localSaved)&amp;&amp;(metadataRequested&#124;&#124;metadataSaved)&amp;&amp;state?.dirty" @click="capture?.markSaved();mediaRequested=metadataRequested=false;notice='已按用户确认标记保存。'">已核对录制及参数下载` | user-deferred · 井井明确将设备采集留给 EXE 人工验收；可见空态保存/停止/放弃禁用已验 |
| M05-C018 | `LipExtractionPage.vue:94` | `<button :disabled="recording&#124;&#124;busy&#124;&#124;!dirty" @click="confirmDiscard=true">放弃未保存内容` | user-deferred · 井井明确将设备采集留给 EXE 人工验收；可见空态保存/停止/放弃禁用已验 |
| M05-C019 | `LipExtractionPage.vue:95` | `<button @click="discard">放弃` | user-deferred · 井井明确将设备采集留给 EXE 人工验收；可见空态保存/停止/放弃禁用已验 |
| M05-C020 | `LipExtractionPage.vue:95` | `<button @click="confirmDiscard=false">取消` | user-deferred · 井井明确将设备采集留给 EXE 人工验收；可见空态保存/停止/放弃禁用已验 |
| M05-C021 | `LipExtractionPage.vue:97` | `<input type="file" accept=".mp4,.mov,.avi,.mkv,.wmv,.m4v,.webm" multiple :disabled="busy&#124;&#124;recording" aria-label="选择离线视频" @change="choose"/><button :disabled="busy&#124;&#124;recording&#124;&#124;!files.length&#124;&#124;` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-C022 | `LipExtractionPage.vue:97` | `<button v-if="busy" @click="abort?.abort()">取消本次任务` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-C023 | `LipExtractionPage.vue:97` | `<input v-model="cloudOptIn" type="checkbox"/>明确上传所选视频至当前账号，进行正式离线分析（占用服务器额度）` | not_applicable · Windows 本机入口不提供此网页上传/下载控件，网页服务另列未准入 |
| M05-C024 | `LipExtractionPage.vue:98` | `<button :disabled="busy" @click="refreshHistory">刷新本地历史任务` | verified · 真实Qt空历史刷新，3e2a15f6e9d3487d8217e3b5516ce1f7/qt-report.json |
| M05-C025 | `LipExtractionPage.vue:98` | `<select v-model="historyId" aria-label="历史唇形任务"><option v-for="job in history" :key="job.id" :value="job.id">{{job.name}}` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-C026 | `LipExtractionPage.vue:98` | `<button :disabled="busy&#124;&#124;!historyId" @click="loadHistory">读取历史结果` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-C027 | `LipExtractionPage.vue:99` | `<select v-model.number="selection" :disabled="busy" aria-label="选择唇形结果" @change="replayIndex=0;stopReplay();draw()"><option v-for="(result,i) in results" :key="result.id" :value="i">{{result.name}}` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-C028 | `LipExtractionPage.vue:101` | `<button @click="replaying?stopReplay():play()">{{replaying?'暂停动画':'播放动画'}}` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-C029 | `LipExtractionPage.vue:101` | `<input v-model.number="replayIndex" type="range" min="0" :max="Math.max(0,rows.length-1)" aria-label="回放帧" @input="stopReplay();draw()"/><span>{{(rows[replayIndex]?.time_s??0).toFixed(3)}} s` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-C030 | `LipExtractionPage.vue:101` | `<input v-model.number="offset" :disabled="busy" type="number" min="-2" max="2" step="0.001"/> s` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-C031 | `LipExtractionPage.vue:101` | `<button :disabled="busy" @click="saveResult('apply')">应用偏移并保存` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-C032 | `LipExtractionPage.vue:101` | `<button :disabled="busy" @click="saveResult('save_without_offset')">保存但不应用偏移` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-C033 | `LipExtractionPage.vue:101` | `<button @click="notice='已取消本次偏移保存，原始结果保留。'">取消偏移保存` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-C034 | `LipExtractionPage.vue:102` | `<button :disabled="busy" @click="offset=Number(selected.metadata.offset_suggestion.offset_seconds.toFixed(3))">填入偏移建议` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-C035 | `LipExtractionPage.vue:103` | `<select v-model="quality" aria-label="动画导出质量"><option value="high">高清 1080` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-C036 | `LipExtractionPage.vue:103` | `<button :disabled="busy&#124;&#124;!port.exportAnimation" @click="exportAnimation('mp4')">导出视频` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-C037 | `LipExtractionPage.vue:103` | `<button :disabled="busy&#124;&#124;!port.exportAnimation" @click="exportAnimation('gif')">导出 GIF` | blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |

## 公共、状态及性能验收

| ID | 可复现操作与判据 | 状态 |
| --- | --- | --- |
| M05-G01 | 空态与真实数据：1920×1000、1366×768、1280×720，100/125/150%缩放、浅深色、侧栏拖宽，截图及scrollHeight/clientHeight记录；普通滚轮不带动整页，小窗保存/取消可达 | verified · 四档CSS视口/浅深色/右栏折叠，真实Qt DPR1.5；其他OS/实体屏blocked |
| M05-G02 | 原始录音加载冷/热计时；按钮反馈20次、拖动缩放20次，记录median/p95/max与最后状态；目标见P17 G-P01–03 | user-deferred · 井井明确将设备采集留给 EXE 人工验收 |
| M05-G03 | 全长、2/8/32/64/128倍至采样级连续波形、自适应振幅、零/正/负刻度，试听不变；科学显示抽点不进入导出 | not_applicable · M05空态无音频波形；真实结果 blocked · 批准目录无真实视频/lip.json，不能生成输入替代 |
| M05-G04 | 全长/选区试听、暂停/停止、重复点击、音量、切声道、换文件/页签；只有一个播放归属 | user-deferred · 井井明确将设备采集留给 EXE 人工验收 |
| M05-G05 | 缺输入、非法输入、错误后恢复、取消、迟到、关闭未保存，逐项记录受控注入与真实失败 | partial · 缺输入/非法值/取消/重试/草稿按C/F行证据，额外故障不扩为通过 |
| M05-G06 | 原文件前后SHA-256一致；输出只写output/validation/p17/M05，不得写原目录 | verified · originals-unchanged.json true；输出在本轮独占M01-M05目录，Mxx/evidence-index.json定位 |

V2 实际函数索引（用于追溯）：`__init__`, `set_neighbors`, `reset`, `_alpha`, `_neighbor_average`, `filter`, `__init__`, `_build_ui`, `_setup_video_pipeline`, `_refresh_devices`, `_enumerate_devices_worker`, `_on_devices_detected`, `_open_camera`, `_on_camera_combo_changed`, `_on_audio_combo_changed`, `_restart_live_audio_stream`, `_set_device_controls_enabled`, `_on_filter_toggled`, `_filter_settings`, `_load_filter_cutoff_setting`, `_on_filter_strength_changed`, `_update_filter_range_from_camera`, `_current_min_cutoff`, `_build_mesh_neighbors`, `_setup_live_audio_stream`, `set_theme`, `_select_save_directory`, `_default_save_directory`, `_on_video_tick`, `_show_frame`, `_on_hfps_frame`, `_draw_overlay`, `_draw_metrics_overlay`, `_open_help`, `_open_animation_player`, `upload_video`, `toggle_hfps_recording`, `_start_hfps_recording`, `_finish_hfps_recording`, `_recognize_video_frames`, `_fill_leading_nans`, `_save_offline_recognition`, `start_raw_recording`, `stop_raw_recording`, `_load_recording_bundle`, `_point_in_bounds`, `start_recording`, `stop_recording`, `_save_recording_files`, `_interpolate_short_numeric_gaps`, `_interpolate_short_landmark_gaps`, `closeEvent`, `__init__`, `_build_ui`, `_apply_theme`, `_clean_time_axis`, `_apply_initial_view`, `_on_param_changed`, `_on_manual_toggled`, `_current_offset`, `_on_apply`, `_on_skip`, `_on_scroll`, `_on_press`, `_on_release`, `_on_motion`, `_limit_visible_audio_points`, `_audio_envelope`, `_estimate_offset_seconds`, `_redraw`, `__init__`, `_build_ui`, `_infer_render_size`, `_normalize_range`, `_time_to_index`, `_render_frame`, `_render_points`, `_crop_to_face_region`, `_show_frame`, `_on_slider_changed`, `_toggle_play`, `_on_playback_state_changed`, `_on_player_position_changed`, `_sync_frame_from_player`, `_selected_indices`, `_quality_profile`, `_resampled_landmarks`, `_save_video`, `_save_gif`, `closeEvent`

## 本轮证据定位与现行布局

- 完整科学/主流程：`output/validation/p17/M01-M05-5cb4879d0220451da22cf911ce0043e1/report.json`，实际原生产物回读为同目录`readback.json`。
- 普通交互X01–X08：`output/validation/p17/M01-M05-f3d0eb386cd94cb3bc05ad939264ab78/report.json`，exit 0，页面错误0。
- 尾测X09：`output/validation/p17/M01-M05-be70b5cb449347c4a6ca8cbe695bdb7e/report.json` 中已完成checks；X09–X11及X08均完成，exit 0、页面错误0。
- 最后普通控件X12：`output/validation/p17/M01-M05-58f99d3ee30746759b579ca0bcd2d093/report.json`，exit 0、错误0；M05空历史刷新：`output/validation/p17/M01-M05-3e2a15f6e9d3487d8217e3b5516ce1f7/qt-report.json` success=true。
- 最终构建真实隐藏Qt：`output/validation/p17/M01-M05-86b574624cd247ee8f349bfa4610a162/qt-report.json`，success=true。
- 公共播放器/参数X13：`output/validation/p17/M01-M05-4595abcd041e427cbf7d3b143a245dbc/report.json`，80项逐一勾选、音量/空格/范围/进度，exit 0。
- 共享波形：`shared-waveform/1790855238430/report.json`，真实50000点、4组、24次Ctrl滚轮；共享语谱：`shared-spectrogram/1790853257028/report.json`；共享三栏：`shared-workbench/1790854845430/report.json`，均相对`output/validation/p17`。
- M03为明确的左右两栏例外，其他四页三栏。1920×1000、2560×1360、3840×2080为模拟CSS视口；1280×720保持滚动。右侧长历史/高级操作允许内部滚动。Qt DPR1.5，实体屏报告1707×1067，不冒充1920实体屏。
