# M03 开发态功能清单复核

2026-09-13。**verified（限定本轮功能映射与Windows独立Chrome定向操作）**。复核范围为原V2六功能组、说明书3.1–3.4和实际GUI/批次源码。完整M03仍in_progress；本报告不把入口存在等同于全平台验收。实施方案见[功能复核计划](../plans/2026-09-13-m03-function-review.md)。

## 复核依据

直接只读核对相邻V2的 `phonetic_toolbox/gui/widgets/egg_widget.py`、`gui/dialogs/egg_batch_dialog.py`，并读取3.1–3.4完整说明书文本。原手册HTML SHA-256为 `a46c984929bcd8073ff1daf4e6b382a6685d4ca19e0fee0e29de3ecf0fb39ad5`，与原登记相同；`third_party/egg-migration.json` 所列八份原文件逐份哈希一致。本轮无外部方法、代码或依赖引入。代码许可仍未闭合，不能用哈希一致推断授权。

下表的已有证据是历史限定验证，本轮没有重跑全部科学数值。源码符号及D01–D10差异的细节保留在[来源映射](../modules/evidence/M03-source-map.md)。

| 功能组 / 手册操作 | V2实际符号 | V3入口与行为 | 证据及当前边界 |
| --- | --- | --- | --- |
| F01 / 3.1 双声道载入、交换、独立归一化 | load_wav_file、toggle_channel_flip、load_file | 目录列表/上传、交换声道、数组核心；总览是原音频，科研/试听为归一化结果 | [核心](m03-core-report.md)、[试听](m03-playback-report.md)，A01–A03；实际声卡另验 |
| F01 / 3.3 帮助与来源 | show_help_dialog | 使用说明、方法与来源及公共致谢 | [方法审阅](../references/m03-method-audit.md)，A29来源未闭合 |
| F02 / 3.2 音频与EGG微观、原始/滤波、高低通 | toggle_egg_display_mode、handle_highpass_slider_change、update_zoom_plots | V2两列四图、EGG图上滤波控件、来源提示 | [页面](m03-ui-report.md)、[E1](m03-report.md)，A04–A06 |
| F02 / 3.3 谱窗与dB上下限 | handle_spec_window_change、update_roi_plots | 第二行谱窗5–50ms和dB范围 | 核心/页面/E1，A07；维持EGG专属PSD，未换成M01 Praat图 |
| F03 / 3.3 GCI与GOI独立方法 | toggle_gci_method、toggle_goi_method、analyze_events | 两下拉各选斜率/尺度0.25，实际默认slope/scale | 核心独立四组合，A08/A10；不提供无效阈值旋钮 |
| F03 / 3.3 峰谷显著度、自动 | toggle_auto_prominence、handle_peak_prominence_change、handle_valley_prominence_change | 自动峰与手动峰/谷、非法参数提示 | 核心及[批次反馈](m03-batch-feedback-report.md)，A09 |
| F03 / 3.2 CQ/SQ与事件、声门活动 | calculate_cq_sq_segment、detect_glottal_movement | CQ/SQ双轴、微观事件、F0变化标记 | 核心/页面，A11–A13；独立mask及三种旧局部滤波规则不合并；标记不代表器官位移 |
| F04 / 3.3 Praat与GCI F0独立显示 | toggle_f0_visibility、toggle_f0_correction | 独立开关、各自真实时间轴 | 核心/E1，A14/A15；单文件CSV始终保留双F0 |
| F04 / 3.3 逆滤波阶数、对比、双WAV | run_inverse_filtering、InverseFilteringResultDialog | LP自动/指定、前后波形和频谱、原音频/IF双试听与保存/下载 | [任务导出](m03-jobs-report.md)、试听、[结果反馈](m03-result-feedback-report.md)，A16/A17；简化CP来源未闭合 |
| F05 / 3.1 总览60秒、长音频滑条 | plot_timeline、on_timeline_slider_change | 最下方紧凑总览、图下双声道/缩放/适合窗口，超过60秒滑条导航 | [长文件](m03-long-report.md)、[紧凑布局](m03-overview-report.md)，A18；全段计算120秒/576万帧 |
| F05 / 3.1 单击总览定位 | on_timeline_click | 本轮补回单击移动选区且保留时长，末尾贴边；拖动可另选区间，松开更新 | 本轮真实任务回读；此前零长度漏项已复现，不用旧总览截图替代该行为验证 |
| F05 / 3.2 起点/时长、红线定位、微观缩放/拖动 | update_roi_plots、on_left_release、on_zoom_scroll/on_zoom_drag | 数值控件、点击联动、Ctrl滚轮、拖动和键盘 | [E3范围](m03-e3-report.md)，A19；原源码5–5000ms覆盖过时手册10–200ms |
| F05 / 3.2 播放、停止 | play_audio、stop_audio | 公共选区播放器及停止/暂停，交换/关闭停止 | 试听专项，A20；浏览器节点样本已验，物理声卡仍待 |
| F05 / 3.3 CSV/三PNG及名称 | save_analysis、_save_csv_data、_save_plots | 当前ROI任务、源名称/时间保存、完整产物与同名保护 | E1、任务导出、字体预检、结果反馈，A21；保留原outer join缺失行 |
| F06 / 3.4 目录、交换、高通、方法、F0、静音阈值、可选图片 | EggBatchDialog、BatchWorker.run | 文件集合/全选、独立批次配置、完整滤波文件、逐任务保存 | E1、批次反馈，[E2网页](m03-e2-report.md)，A22–A24；图与CSV静音mask策略不同 |
| F06 / 3.4 进度、取消、失败继续 | cancel_processing、BatchWorker.cancel/run | 处理记录内取消/重试；本轮补提交期间取消并防止取消上一批 | 本轮延迟真实接收响应验证，任务导出/E2，A25/A26；已完成产物保留 |

## 本轮两项修复

总览原公共组件只支持拖选，单击会令start=end。EGG现在显式启用clickMovesSelection，其余页面保持默认。小于3px位移作为单击，移动现有时长；点击末尾向前贴边保留完整时长，避免空选区或越界。正常松开后提交一次实际预览；已有任务繁忙时保留新选区并提示结束后更新，避免重叠任务。原V2可将ROI终点留在文件外，本轮沿用V3的有界选区约束。

上一轮冻结批次提交控件时禁用了返回/Escape，但缺少替代取消入口。本轮加入取消提交，立即退出弹窗、停止后续文件；用独立提交ID集合取消本轮已接收任务，迟到接收也请求取消，上一批不受影响。当前提交请求尚未返回时禁用开启新批次；取消请求失败要在主页面报告，不宣称已撤回成功任务。

## 保留的V3调整与剩余项

- 单文件通过目录/上传列表选择；导出先形成持久结果，再选择目录或下载，不自动写回源目录。IF双WAV补齐手册承诺，原GUI实际未写WAV。批次图片按实际V2默认关闭，手册默认生成图的文字已过时。
- 参数编辑后显式更新，旧图与试听失效；总览定位/拖选及四图手势在空闲时更新。普通滚轮滚动页面，Ctrl滚轮缩放。使用V3公共主题/字体/弹窗，四图与紧凑总览遵守井井已审阅布局。
- 已复核六功能组都有实际入口和关联证据，未发现其他遗漏控件。A01–A30保留[联合记录](m03-report.md)中的限定状态，不能将本轮检查扩大为全部科学路径重测。完整M03仍in_progress。
- 当前剩余主要是A20物理声卡、A28原生多屏/DPI、A29方法文献及代码许可链、A30生产环境等证据。已有Windows开发态Chrome/Qt、本机托管PG与自然录音的历史证据保持有效范围。本轮不启动生产/设备/发行新范围。
- EXE、相关探针和打包均暂停，M04不自动推进。下一项为方法/源码来源未决项的集中收口；需要原始授权材料的项如实保留，不以UI完成冲掉。

## 验证记录

`npm --prefix frontend run test` 65 passed；`run typecheck`、`run build`通过。`node tests/e2e/m03-batch-feedback.cjs` 6组、`node tests/e2e/m03-overview.cjs` 3组通过，证据分别为 `output/validation/m03-ui/chrome-52986b280b1144f69b9a360a3152cc0f/report.json` 与 `chrome-94121f04e4544f629ca5d6aa4613aa61/report.json`。

新增 `node tests/e2e/m03-function-review.cjs`：首轮 `chrome-b055eaac0e1147e49e976cd88742c26f` 实际复现总览单击时长变零；第一轮修复 `chrome-27f2a6d29cd74884929ad06d28ff1b0b` 的3组通过。扩展M01/M02检查时，`chrome-ab75f5dd8a634aa08daf5c309406c74f` 因测试错误地在M01正文寻找公共底部时间控件而失败；改为直接检查真实波形选区的显示宽度。所有失败产物保留。

正常更新、批次和取消走真实本机服务/科学子进程；故障仅为延迟真实接收响应，不伪造科学结果。已有测试schema隔离副本，未执行DDL。未改V2、科学包、全局环境、语料或EXE，未push/发布；本轮未复跑完整科学对照、Qt和托管账号流程。

最终新增回归5组通过，0页面错误：`output/validation/m03-ui/chrome-98010f76c63a445380703d7e5816fc3c/report.json`。其中M01使用实际WAV，M02使用既有显式合成表格页面，仅验证公共波形默认单击/拖选，不扩大为M01/M02完整流程。最终工作台截图已查看。

`.venv/m09-ui/Scripts/python.exe scripts/validate_docs.py` 检查575文件、330来源、32任务，errors=[]，历史快照失效链接单列；`scripts/check_architecture.py` errors=[]。`npm --prefix frontend run contracts:check` / `run ui-data:check` 无漂移，`git -c core.safecrlf=false diff --check`通过。
