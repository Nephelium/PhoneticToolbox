# M04 V2 说明书与源码映射

2026-09-13，M04-A。已读相邻V2 `Phonetic_Export/index.html` 的11.1整节，包含内部11.6/11.7小标题，锚点 `s1773837409220`。源码及说明书哈希见 [基准清单](../../../tests/fixtures/m04/manifest.json)。仅公开合成输入进入基准，不复制手册图片或全文。

## 四组功能

| 组 | 说明书操作 | V2 实际入口 | V3 计划入口及验收 |
| --- | --- | --- | --- |
| F01 | 输入/输出目录、刷新、选择WAV | `LPCSpectrumWidget._browse_input/_browse_output/_refresh_files/_on_file_selected` | 统一文件能力/文件列表，保存目录独立授权；切文件作废旧结果/标注，A01/A02 |
| F01 | 双声道读取 | `read_wav_float_mono`，整型按位深转换后各声道平均 | 计算使用原平均规则，记录真实采样率/样本数；试听原声道语义单列，A03 |
| F01 | 同名TextGrid、循环层级 | `read_sibling_textgrid/next_tier_name` 与 `_on_textgrid_button_clicked/_draw_textgrid` | 桌面同目录关联，网页按资源显式关联；共用时间标注，A04/A05 |
| F02 | 阶数/频率上限/Min/Max/动态y轴 | `LPCSpectrumConfig`、控件初始化、`compute_spectrum` | 独立草稿及任务快照，动态与手动范围明确，A06/A07 |
| F03 | 浏览、缩放、平移、Shift框选 | `_on_scroll/_on_press/_on_motion/_on_release` | 公共波形可选手势适配；普通滚轮页面滚动，Ctrl滚轮缩放，保留Shift框选及明确清除选区，A08/A09 |
| F03 | 无框选开始处理 | `_start_lpc_processing` 使用 `ax.get_xlim()` | 明确记录波形可见范围为本次ROI，频率轴不作为时间，A10 |
| F03 | 有框选开始处理 | 同函数 `int(t*fs)`，`[start_idx:end_idx]` | 保留截断取整/半开样本区间并写入快照，A11 |
| F03 | LPC计算与显示 | `compute_lpc_spectrum`、`_plot_lpc_curve` | 纯核心输出真实1024点谱线，公共科学图表渲染，A12–A14 |
| F03 | 返回波形 | `_return_to_waveform`、`_last_wave_xlim/_last_lpc_result` | 两种视图独立状态，回波形保留上次结果和时间范围，A15 |
| F04 | 播放/停止 | `_play_current_file/_stop_audio/_on_player_position_changed` | 公共播放器，明确试听范围与声道；关页/切文件停止，A16 |
| F04 | 自动生成PNG、选区标签命名 | `_create_export_figure`、`save_plot_figure/extract_label_in_range` | 每次处理生成持久PNG结果，桌面保存/网页下载，300DPI与碰名保护，A17/A18 |
| F04 | 帮助与错误 | `_open_help`、各QMessageBox入口 | 公共说明/错误/任务反馈，取消和失败保留源数据，A19/A20 |

## 固定数值规则

- 默认阶数50（1–200），频率上限8000Hz（100–48000），纵轴−5至35dB（各控件−200至100），动态范围默认关闭。
- 输入转float64，多声道逐样本求平均。PCM16除32768，PCM32除2147483648，uint8减128再除128，浮点原值保留。
- 预加重0.97，首样本保留；全ROI Hamming窗，`np.correlate(..., mode='full')`正延迟部分，自相关Toeplitz求解。分母 `[1, -coeff...]`，`freqz(1.0,a,worN=1024,fs=fs)`。
- 频率网格0至Nyquist之前，共1024点。幅度 `20*log10(abs(response)+1e-10)`，没有乘预测误差增益，也没有原始FFT叠加；不称校准声压级。
- 动态纵轴使用 `freq <= freq_max_hz` 的谱值，最小/最大各外扩5dB，不改变谱数组。默认阶数50时51样本失败，52样本可求解；静音/NaN/Inf在已测场景明确失败。

## 已发现差异及迁移处理

| 差异 | 实际证据 | 处理决定 |
| --- | --- | --- |
| 旧规划列SRC-PRAAT为算法来源 | `core/acoustic/lpc.py`直接调用NumPy/SciPy；服务无Praat计算 | 更正模块技术描述；包聚合导入Praat不构成本算法依赖。引用深查按要求暂停 |
| 手册称频率上限控制计算 | 两种频率上限下1024点数组逐字节相同 | 标为显示上限，保持原网格与动态范围规则 |
| 手册称自动读取TextGrid | 实际由按钮触发查找/解析，选文件只清缓存和层名 | 自动发现关联候选，读取与层切换明确可操作；不悄悄套用旧文件标注 |
| 标签末端与手册泛称选区内不同 | `(0,1)` 和 `(.15,.85)` 均排除末区间，重复`b`去重 | 初版保留原提取规则并说明；不静默改文件标签 |
| 手册称普通单击可清框选 | 普通按下/释放没有清空 `_selection_range_sec` | 明确新增清除选区入口补手册承诺 |
| 波形模式试听忽略框选 | 先取可见范围，只有谱图模式才取框选；原QMediaPlayer播放源文件 | 保留可见范围试听，并独立提供明确选区试听，默认分析范围可辨 |
| 谱图状态再次处理可能把Hz当秒 | 无框选分支直接读取当前 `ax.get_xlim()` | 保存波形时间范围/本次ROI，重算继续用时间快照，禁止复用频率轴 |
| 固定PNG同名覆盖 | 原`savefig`直接写固定文件名 | 使用现有受控结果/保存机制，保留建议文件名并避免覆盖 |
| 原规划提及原谱和LPC图例 | 实际只画LPC曲线 | 初版保留单曲线，原FFT叠加不作为迁移已完成功能 |

以上GUI行为来自源码静态核对，M04-A未运行完整GUI交互。数值、WAV、TextGrid和PNG的真实双轮基准见 [报告](../../testing/m04-baseline-report.md)。
