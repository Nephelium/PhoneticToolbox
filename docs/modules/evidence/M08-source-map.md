# M08 说明书、源码与实现映射

2026-09-26。已读相邻 V2 `Phonetic_Export/index.html` 第 5.1–5.4 节全文。继承源码仅作为迁移来源和独立基准生成输入，V3 运行时不导入旧目录。

## 六组功能

| 组 | 说明书 / 原符号 | V3 实现 | 正常 / 错误证据及边界 |
| --- | --- | --- | --- |
| F01 单文件 | 5.1/5.2；`load_audio`、`synthesize_sound`、`save_audio_smart`；`synthesize_from_pitch` | `PitchManipulationPage.vue` 文件/参数、原音/合成音播放器；`m08_synthesis.py` 原路径；`m08_rules.next_name` 编号 | Windows 4 个合成配置波形/真实时间轴精确对比；Chrome 真音频、0.8 倍速、当前视野和整段、保存编号/播放停止。零/非有限倍率、曲线长度/ROI 错误、输入/输出预算单测。正式 FileProvider 仍缺 MP3/FLAC 与 M08 adapter，完整 F01 不标完成。 |
| F02 曲线与参照 | 5.1/5.2；`update_pitch_data`、`restore_pitch_data`、`import_f0_sequence`、`update_axis_range`、`add_ref_line`、`clear_ref_lines`、`save_comparison_plot` | `PitchCurve.vue`、`state.ts`、`HistoryPlot.vue`；公共主题/字体/PNG painter | Shift 绘制、Ctrl 恢复、有声 mask、反向笔画沿旧规则；导入第二列并均匀插值、视野不重置。间断/无声/非有限输入拒绝。真实 Chrome 手绘/导入，真实历史 PNG 与浅深色已查看。 |
| F03 批次文件 | 5.2/5.3；`_get_current_batch_files`、`delete_current_batch_files`、`rename_current_batch_files`、`update_comparison_plot` | 页面 currentBatch 按源 ID/两位小数时间前缀分组；M08Port 明确结果 ID；renamePlan 共同前缀替换及大小写冲突预检 | Chrome 在专属合成目录完成真实重命名/删除，源 WAV 保留；非法路径拒绝。历史从已保存 PCM16 重新提取 F0。正式 owner/配额/下载/删除与原子编号需平台接线，不把 test adapter 当生产存储验收。 |
| F04 批量变速变调 | 5.4；`BatchProcessorDialog`、`BatchProcessorWorker.run`、`ManipulationService.process_single_file` | 页面文件夹入口/多选/独立参数/逐文件任务；`m08_transform.py`，`m08_jobs.execute` | 三个可执行 V2 配置精确对比，另两个非零 Hz 原始失败保留；修复 Hertz 后 150×1.2+20 的独立约 200 Hz 检验。Chrome 双音频成功及损坏 WAV 单独失败。停止剩余提交及任务取消接口保留，正式硬取消/持久队列未接。 |
| F05 批量基频 | 5.3；`batch_linear_save`、`generate_batch_linear`、`add_knot`、`clear_knots` | `m08_batch.py` 纯 iterator、`m08_rules.validate_controls`；页面批量基频及快照 | 16 种起终点连接组合与 order 拐点、offset 输出 WAV 精确对比；前端验证说明书 6×6×6=216 组合、常量/顺序/逆序。Chrome 实际四文件输出。越界/重复时间/对角长度不一致/超过256组合明确拒绝，不截断结果。 |
| F06 拐点与帮助 | 5.3；`KnotEditorDialog.add_row/del_rows/save_changes`、`_open_help_doc` | 公共 ModalDialog 中拐点表与模块帮助、references emit | 首末行不可删除，中间行添加/选择删除，保存先排序且 mode 跟随行，再检查时间/列表。Chrome 首末行禁删、添加/保存/四输出，纯状态测错误。方法入口待公共 AppShell 注册后统一打开 SRC-PRAAT。 |

## 保留的数值与数据规则

- 加载 `Sound.to_pitch()` 的默认参数与 `pitch.xs()`，0 表示无声，绘图断开，不能补零曲线。
- 单文件：`extract_part(..., preserve_times=True)`；PitchTier 仅添加正 F0；Manipulation 固定 0.01 s/75/600；`abs(speed-1)>0.01` 时 DurationTier 取 1/speed；overlap-add 重合成。当前视野的边界是原代码实数时间，无额外采样取整。
- 文件夹：先 `Lengthen (overlap-add)`，再 Multiply frequencies，最后 Shift frequencies。倍率/Hz 的 0.01 阈值原样保留。
- 批量基频：order/reverse 先数值排序，reverse 倒序，所有对角列表长度一致；full 沿输入顺序做笛卡尔积；constant 取首值。拐点显式加入 PitchTier，offset 用最近原始 F0，加法不改成倍数。批量基频不应用单文件 speed。
- 导入文本逐行处理。一列取该列，两列及更多取第二列。原对话框提示空格/逗号分隔容易使人误以为单行多个频率全部导入，实际代码只取第二项；V3 提示明确每行一个值或两列，未静默改解析规则。
- 原手绘反向笔画依索引升序写入由上一频率到当前频率的插值；本轮照原规则保留，若要修正反向笔画方向需独立行为审阅。
- 单次保存名 `<stem>_<start:.2f>_<end:.2f>_modified_<max+1>.wav`；批量包含 seg/kn/lin/offset/Fpath/knot/combo 与尾号。当前已实现建议名与完整配置，正式存储需原子分配。

## 明确的适配/修正

科学原样迁移为 `m08_synthesis.py`、`m08_batch.py`、`m08_transform.py`。batch 仅将文件扫描/写入移到外层，以 iterator 输出原名称、Sound、控制值。其他校验/导入/命名规则放独立 `m08_rules.py`。

1. **M08-FIX01**：V2 `Shift frequencies(...,"Hz")` 在 Parselmouth 0.4.7/Praat 6.1.38 报 Unit 错误。原样函数默认仍复现失败，handler 显式选 `Hertz`，保留先乘后加与阈值。未修改 V2。
2. **M08-FIX02**：非法倍率/无限值/范围/组合预算拒绝，逐文件异常可见。取代旧静默回落、批次吞异常与覆盖同名风险，不变更合法输入算法。
3. **M08-FIX03**：保存绑定生成快照；当前视野变化不重命名旧结果。删除列出明确结果 ID。重命名按 owner 结果表保留可追踪历史，避免旧版改前缀后记录消失。
4. 普通滚轮滚动，Ctrl+滚轮缩放遵循已批准 V3 公共规范。合成波形按实际输出时长显示，避免旧界面用原音视野截掉减速结果。历史明确零起点对齐，PNG 附版本文件名。

## 独立基准

`tests/support/m08_baseline.py` 只加载原 V2 文件或 AST 提取原 service 方法，不调用 V3 生成 expected。公开合成 16 kHz/0.6 s 双谐波音频；原数据与配置、源码 SHA-256 在 `tests/fixtures/m08/v2.json`，数组在 `v2.npz`。66 数组、571,595 值，包含输入/中间轴/原结果，两个旧 Hz 错误另存。

Praat 变速重合成存在随机状态，初次不固定种子时3项精确比较失败。基准与测试在每个适用案例前调用 `random_initializeWithSeedUnsafelyButPredictably(42)`，生产默认不被该测试策略改变。官方语义见 [Praat 随机种子说明](https://praat.org/manual/_random_initializeWithSeedUnsafelyButPredictably_.html)。失败历史保留于工作记录，不通过扩大容差解决。

Windows/Linux、组件/正式宿主、科学/资源/文件权限状态分别见 [验收报告](../../testing/m08-report.md)。
