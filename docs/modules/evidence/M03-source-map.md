# M03 说明书与源码审阅

2026-09-12。状态：源码/说明书映射已完成，算法迁移 planned。本轮未运行旧 EGG 算法、读取自然语料或创建数据库。对照相邻 v2 `Phonetic_Export/index.html` 的 3.1–3.4 全文与本工程继承源码。说明书纯文本仅留在忽略的 `output/validation/20260912-closeout/manual-text.txt`，未复制其图片或全文进发行物。

## 全部功能映射

| 功能 | 手册/实际入口 | 核心或服务 | v3 目标与验收 |
| --- | --- | --- | --- |
| F01 双声道 WAV、交换、归一化 | §3.1/3.3，EGGWidget.load_wav_file / toggle_channel_flip / _load_data_from_path | EGGAnalysisService.load_file | EggAnalysisPage 文件工具与数组输入适配。恰好双声道，左 EGG/右音频，交换重新分析，两个峰值独立到0.7，原文件不变，A01–A03 |
| F02 原始/滤波 EGG、高通、音频波形 | §3.2，toggle_egg_display_mode / handle_highpass_slider_change / update_zoom_plots | load_file、get_events_segment、filters.apply_highpass_filter/apply_lowpass_filter | 共同时间状态、信号来源标签和参数快照，A04–A06 |
| F02 语谱图窗长、灰度范围 | §3.3，update_roi_plots / handle_spec_window_change，spec_vmin/max | GUI 内 Matplotlib.specgram：NFFT=int(fs*window_ms/1000)，75%重叠 | EGG 自身谱图适配，不能直接用固定5ms的M01 Praat预览替换，A07 |
| F03 GCI/GOI独立斜率/尺度、峰谷、自动 | §3.3，toggle_gci_method/toggle_goi_method/toggle_auto_prominence / _trigger_reanalysis | find_gci_goi_peak_min_criterion、analyze_events/get_events_segment | 核心显式配置、事件单位秒与实际信号源，A08–A10 |
| F03 CQ/SQ与微观事件同步 | §3.2/3.3，update_roi_plots / update_zoom_plots | calculate_cq_sq_segment、calculate_cq_sq | 时间轨与独立数值导出，保留公式/缺失mask，局部分析差异见D04，A11–A12 |
| F03 声门移动标记 | 源码追加，toggle_glottal_detection | detect_glottal_movement：Praat F0斜率±1000 Hz/s，同类间隔≥0.1s | 同名旧功能保留但说明为F0变化启发式，不直接宣称测得器官位移，A13 |
| F04 Praat/GCI两类F0及显示开关 | §3.3，toggle_f0_visibility/toggle_f0_correction / _update_f0_contour | calculate_praat_f0、_calculate_gci_f0 | 分别保留真实帧/事件中点时间，不把两类F0合并，A14–A15 |
| F04 逆滤波阶数、对比窗、WAV | §3.3，run_inverse_filtering / InverseFilteringResultDialog | apply_simplified_cp_inverse_filtering | 共同工作台内对比视图和双WAV结果集合，当前缺失承诺见D03，A16–A17 |
| F05 60秒总览、起点/时长、定位、红线 | §3.1/3.2，plot_timeline / on_timeline_click / on_left_plot_click | time_vector与当前ROI | 总览/主ROI/微观窗口三层分开，同源时标，A18 |
| F05 50ms细查、10–200ms缩放、拖动 | §3.2，on_zoom_scroll/on_zoom_drag、on_left_scroll/on_left_drag | GUI局部波形与事件 | 保留点击联动及键盘数值输入，A19 |
| F05 ROI播放、停止 | §3.2，play_audio/stop_audio | 已归一化的音频角色声道 | 共用播放能力，仍播放当前ROI，交换后声音角色正确，A20 |
| F05 单文件保存CSV及三PNG | §3.3，save_analysis / _save_csv_data / _save_plots | GUI数据表outer join、150dpi导出 | 保留时间列集合与来源、白底三图、时间文件名，原输出冲突另名保护，A21 |
| F06 目录、静音阈值、高通、交换、方法、双F0、可选三图 | §3.4，EggBatchDialog / BatchWorker.run / save_batch_plots | 全局事件、GCI时间网格、插值、归一化音频绝对值20ms滑动均值mask | 共用持久任务和完整文件集合，单文件/批次规则显式分开，A22–A24 |
| F06 进度、取消、失败继续 | 源码 BatchWorker.cancel/run | 旧批次仅文件间查取消 | 新所属worker在文件内可取消，取消不得变为成功空结果，A25–A26 |
| 全局 主题、帮助、来源、错误、双端 | 根规则与六功能组 | P04/P06/P07 | 共同界面/任务/配额，逐端真实操作，A27–A30 |

目标文件、阶段与30项测试定义见 [M03实施计划](../../plans/2026-09-12-m03-implementation.md)。以上目标入口尚未实现。

## 已核实默认值

- `models/config.py:EGGConfig`：峰/谷显著度0.01，自动开启，自动下限0.01；高通25Hz、低通1000Hz；GCI/GOI均slope；criterion_level=0.25；谱窗20ms，显示-70至-10dB。
- `EGGWidget.__init__`：不交换、显示滤波EGG，Praat F0/GCI F0/声门移动初始均关闭；总览60秒，微观50ms。
- `EggBatchDialog`：静音阈值0.01，高通25Hz，不交换，GCI slope、GOI scale，两类F0默认保留，图片默认不生成。不要将单文件配置的GOI默认值套到批次。
- `load_file`：time_vector=np.arange(N)/fs，现有file_duration取最后采样时刻(N-1)/fs。归一化影响实际分析和逆滤波输入，不能仅称显示缩放。
- 自动峰显著度为局部max(abs(signal))*0.6，下限0.01，窗200ms/步100ms。峰/谷先在EGG信号找，随后再在相关区间用差分寻找斜率事件。手册的直接在微分信号找峰谷说法不精确。

## 迁移前必须单列的差异

| ID | 源码事实 | 计划处理与边界 |
| --- | --- | --- |
| D01 | 手册说单文件默认GCI斜率、GOI尺度，实际EGGConfig两者均slope；批处理默认才是slope/scale | 原行为基准分别捕获，v3先保留两套实际默认并注明。改默认必须另行决定 |
| D02 | `calculate_cq_sq`的SQ=(去接触时长-接触建立时长)/接触时长，取值趋于[-1,1]；核心README将SQ写成比值 | 保留计算公式，v3文档明确“旧实现的SQ不对称指标”，不冒充通常比值定义；CQ仅保留严格0.05<CQ<0.95，缺失为NaN |
| D03 | 手册承诺逆滤波自动输出_ORIG.wav/_IF.wav。当前run_inverse_filtering仅打开对比对话框，该GUI文件无WAV写出调用 | v3增加显式保存两WAV，保留归一化音频来源说明，使用受控结果集合，作为补齐承诺单列验收 |
| D04 | calculate_cq_sq_segment滤波路径从已处理信号截取±100ms后再次滤波；get_events_segment从raw截±50ms后滤波 | 不能把两个调用合并后宣称数值不变。A阶段同时捕获，B阶段先显式legacy策略；统一处理需要独立数值差异报告与审阅 |
| D05 | calculate_praat_f0丢弃底层实际帧时间，手工生成0.005+0.01*n；底层compute_praat_f0_track已有pitch.xs() | 基准保存旧时间与实际时间两份，v3以真实时间输出且标记行为修正，不静默移动数值 |
| D06 | GCI/GOI scale分支均硬编码0.25，criterion_level参数未用于阈值；min_f0也未用于峰距，仅max_f0起作用 | 不创造可调但无效的控件。保留0.25实际行为，参数语义在文档里说明 |
| D07 | 单文件CSV是多个时间网格outer join，批次在CQ/GCI时刻插值F0，再按音频绝对值包络mask；其图片重算CQ/SQ并不应用该mask | 不将批次CSV和图中非静音点说成完全相同。raw/native轨与导出策略分列，保持NaN/插值来源，比较时统一明确策略 |
| D08 | EGG filters.filtfilt失败会警告后返回原输入，事件异常或取消可返回空列表，批次保存图片异常可吞掉 | 新状态/错误要显式区分空事件、失败、取消、仅部分产物。正常数值保持，异常不得沿用伪成功 |
| D09 | file_duration少一个采样周期，局部CQ计算可返回ROI外padding点，单文件CSV未再次裁切CQ行 | 元数据区分sample_duration与last_sample_time，输出区间遵循明确半开规则；与旧输出的变更单列审阅 |

上述为本地源码事实，不是对生理测量有效性的验证。本轮不执行上述科学行为修正。

## 来源与现有基准

现有 P03 的 `SYN-EGG-44100`、私有 EGG-01/EGG-05 已有独立Windows服务级基线，见 [捕获协议](../../baseline/capture-protocol.md)和 `tests/fixtures/manifest.json`。自然语料路径仅在忽略的本机证据中，后续需再核对哈希；当前未读音频、未重捕获。P03不能覆盖本表的ROI、图片、CSV、IF双WAV与完整GUI行为。

`PENDING-EGG`仍为local-evidence-only，手册引用《汉语韵律的嗓音发声研究》不足以证明本代码的具体方法/许可来源，后续须核对原文页码、公式和代码来源，不能将手册推荐设置表述成已获独立科学验证。依赖链包含NumPy、SciPy、Parselmouth、Pandas、Matplotlib与WAV读取；当前filters还导入pywt，但EGG实际只用高/低通，不应机械携带无关去噪算法。
