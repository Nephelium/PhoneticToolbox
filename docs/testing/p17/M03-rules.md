# P17 M03 EGG 信号分析逐项验收规则

状态：限定 Windows 已执行，逐行状态见下表，未验范围保留。基线 0779bee；执行记录见同目录 M03-report.md。

已读原始手册 `../PhoneticToolbox_v2/Phonetic_Export/index.html` 第 3.1–3.4 节，及实际源码 `phonetic_toolbox/gui/widgets/egg_widget.py`。V2 源码 SHA-256：`20dd2874be54ae4202fc2aecaf05484227e3aff48bb99d5a08cfca5bb6629daf`。

现行P17要求优先于旧手册：1920×1000 CSS主工作区常用操作单页，普通滚轮不推动整页，小窗保持可达；波形连续、自适应可见振幅。V2原始目录自动保存改为受控显式保存。历史通过不算本轮实测。

## 可复现功能门

| ID | 手册 / V2定位 | V3控件及前置状态、步骤和判据 | 证据 / 状态 |
| --- | --- | --- | --- |
| M03-F01 加载与声道 | 3.1–3.4 / `egg_widget.py` | 加载真实双声道EGG；左EGG右音频，交换两次回到原状态；各声道归一化峰值0.7，不把归一化振幅称声压 | verified · 主流程/X03，真实双声道交换两次恢复 |
| M03-F02 自动首显与状态 | 3.1–3.4 / `egg_widget.py` | 选择录音自动打开有界会话并显示四图；失败可重试，更新不增加持久任务，切换文件时迟到结果不得覆盖 | verified · 主流程 + X04错误重试；20次更新不创建正式任务 |
| M03-F03 选区数值 | 3.1–3.4 / `egg_widget.py` | 起点/时长输入、首尾越界与空值；有效范围保留，非法值不发错误任务 | verified · 主流程40秒/.5秒ROI；状态测试越界/非有限值 |
| M03-F04 微观窗口 | 3.1–3.4 / `egg_widget.py` | 5/50/200/5000ms与边界；微观中心、两右图、GCI/GOI同步，波形可见窗自动纵轴 | verified · 20次实际50/100ms更新 + X04非法0恢复，范围单测 |
| M03-F05 四图手势 | 3.1–3.4 / `egg_widget.py` | 逐张单击/左拖/滚轮/方向键/+/-操作20次；左图共同主ROI，右图共同微观窗口，最后图像不倒退 | verified · 四张各执行滚轮/拖动/右方向键；20次更新另计 |
| M03-F06 总览 | 3.1–3.4 / `egg_widget.py` | 总览单击保留选区时长、末尾贴边；拖动新选区、声道/缩放/适合窗口、60秒滑动导航正确 | verified · X12总览单击保持时长、拖动重选、缩放适合 |
| M03-F07 EGG滤波 | 3.1–3.4 / `egg_widget.py` | 原始/滤波、高通/低通输入、默认2000Hz；非法频率不计算，原始数据不写回 | verified · X03原始/滤波/高低通变更恢复，默认2000 |
| M03-F08 事件参数 | 3.1–3.4 / `egg_widget.py` | 峰/谷显著度、自动开关、GCI/GOI方法逐项改动；自动模式手动峰输入禁用，CQ/SQ缺失分别保留 | verified · X03/X09 + 独立CQ/SQ状态测试 |
| M03-F09 语谱参数 | 3.1–3.4 / `egg_widget.py` | 窗长5–50ms、dB上下限、Praat/GCI F0开关分别组合；谱/F0坐标含单位 | verified · X03窗长/dB/F0开关逐一恢复 |
| M03-F10 正式导出 | 3.1–3.4 / `egg_widget.py` | 保存CSV/三图，任务完成后打开并保存全结果；CSV保留两类F0、PNG回读与选区一致 | verified · CSV501×5、3张PNG及原生保存回读 |
| M03-F11 逆滤波 | 3.1–3.4 / `egg_widget.py` | 稳定元音ROI≤1秒/48000样本、LP自动/指定阶数；打开四图与双试听，保存PNG/ORIG/IF及元数据并回读 | verified · 自动LP真实IF四图/双WAV；X09非法0拒绝，X12指定LP20真实IF |
| M03-F12 独立批次 | 3.1–3.4 / `egg_widget.py` | 选择文件/全选、静音阈值、高通、交换、GCI/GOI、双F0、提交/取消；逐文件结果和失败明确 | verified · X03/X04/X09独立设置、完整77秒批次、取消重试/保存 |
| M03-F13 历史/错误/草稿 | 3.1–3.4 / `egg_widget.py` | 刷新、取消、重试、查看、重新读取、保存/下载、返回；错误在所属窗口，关闭草稿保护和保存失败提示 | verified · 历史/任务/草稿X03/X04/X08/X09及X11读取中断重读/提交取消 |

## 每个实际控件的清单

下表为初次审阅时逐模板控件索引，行号可能随最终布局移动；控件名与绑定为稳定定位。每种普通控件检验正常路径及关键失败/取消，额外数值组合不等同于新的未测控件。动态v-for控件逐文件/参数/任务实例应用同一判据。共享WaveformViewport、AudioTransport、TaskPanel及参数抽屉按总规则复用，不能据模板存在判通过。

| ID | V3源文件/行 | 控件及绑定 | 实际状态 |
| --- | --- | --- | --- |
| M03-C001 | `EggAnalysisPage.vue:233` | `<button @click="save">保存参数草稿` | verified · X03 + X08 |
| M03-C002 | `EggAnalysisPage.vue:233` | `<button v-if="context.files.choose" @click="choose" :disabled="loading"><AppIcon name="folder"/>打开 WAV 目录` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C003 | `EggAnalysisPage.vue:233` | `<input ref="picker" type="file" accept=".wav" hidden multiple @change="addFiles"/><button @click="picker?.click()">导入 WAV` | not_applicable · Windows 本机入口不提供此网页上传/下载控件，网页服务另列未准入 |
| M03-C004 | `EggAnalysisPage.vue:233` | `<button @click="refresh" :disabled="!directory&&context.files.kind==='desktop'">刷新文件` | verified · 尾测 X09 |
| M03-C005 | `EggAnalysisPage.vue:233` | `<button @click="config.flip_channels=!config.flip_channels" :disabled="!source">交换声道` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C006 | `EggAnalysisPage.vue:233` | `<button @click="batchError='';batchOpen=true" :disabled="!files.length&#124;&#124;!tasks?.egg&#124;&#124;batchBusy">批量分析` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C007 | `EggAnalysisPage.vue:233` | `<button @click="help=true">使用说明` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C008 | `EggAnalysisPage.vue:233` | `<button @click="emit('references')">方法与来源` | verified · X07 |
| M03-C009 | `EggAnalysisPage.vue:234` | `<select aria-label="EGG 音频文件" :value="source?.id??''" :disabled="loading" @change="load(files.find(f=>f.id===($event.target as HTMLSelectElement).value)!)"><option value="" disabled>选择双声道 WAV` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C010 | `EggAnalysisPage.vue:235` | `<button v-if="error&&source" @click="updateNow" :disabled="previewBusy">重试预览` | verified · X04，非法输入恢复及真实任务取消/重试 |
| M03-C011 | `EggAnalysisPage.vue:236` | `<input v-model="config.signal_mode" type="checkbox" true-value="filtered" false-value="raw"/>滤波` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C012 | `EggAnalysisPage.vue:236` | `<input v-model.number="config.highpass_cutoff" aria-label="EGG 高通频率" type="number" min="1" max="47999" step="1"/> Hz` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C013 | `EggAnalysisPage.vue:236` | `<input v-model.number="config.lowpass_cutoff" aria-label="EGG 低通频率" type="number" min="1" max="47999"/>` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C014 | `EggAnalysisPage.vue:239` | `<summary>处理记录 · {{visibleJobs.length}} 个任务` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C015 | `EggAnalysisPage.vue:239` | `<button v-if="batchBusy&#124;&#124;batchIds.length" @click="cancelBatch">取消本次批量任务` | verified · X04，非法输入恢复及真实任务取消/重试 |
| M03-C016 | `EggAnalysisPage.vue:239` | `<button v-if="tasks?.saveJob&&completedBatch.length" :disabled="savingBatch" @click="saveBatch">保存本次批量结果（{{completedBatch.length}}）` | verified · 尾测 X09 |
| M03-C017 | `EggAnalysisPage.vue:239` | `<button @click="poll">刷新记录` | verified · 尾测 X09 |
| M03-C018 | `EggAnalysisPage.vue:239` | `<button @click="view(job)">查看 {{labels[job.id]??job.id.slice(0,8)}}` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C019 | `EggAnalysisPage.vue:242` | `<summary>参数快照` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C020 | `EggAnalysisPage.vue:242` | `<button v-if="resultReadError" :disabled="resultLoading&#124;&#124;savingResult" @click="view(resultJob)">重新读取结果` | verified · 尾测 X11，受控故障/延迟，仅真实产物回读 |
| M03-C021 | `EggAnalysisPage.vue:242` | `<button v-if="tasks?.saveJob" :disabled="savingResult&#124;&#124;resultLoading&#124;&#124;!resultConfig" @click="saveResult(resultJob)">选择目录保存完整结果` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C022 | `EggAnalysisPage.vue:242` | `<button v-for="f in resultJob.result_manifest.files" :key="f.id" :disabled="resultLoading&#124;&#124;!resultConfig" @click="download(f.id,f.name)">下载 {{resultNames[f.name]??f.name}}` | not_applicable · Windows 本机入口不提供此网页上传/下载控件，网页服务另列未准入 |
| M03-C023 | `EggAnalysisPage.vue:242` | `<button @click="closeResult">返回分析` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C024 | `EggBatchPanel.vue:6` | `<input v-model="settings.flip_channels" type="checkbox"/>交换 EGG / 音频声道` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C025 | `EggBatchPanel.vue:6` | `<input v-model="settings.generate_images" type="checkbox"/>同时导出三张图片` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C026 | `EggBatchPanel.vue:6` | `<input v-model="settings.keep_praat_f0" type="checkbox"/>Praat F0` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C027 | `EggBatchPanel.vue:6` | `<input v-model="settings.keep_gci_f0" type="checkbox"/>GCI F0` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C028 | `EggBatchPanel.vue:6` | `<input v-model.number="settings.silence_threshold" type="number" min="0" max="1" step=".001"/>` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C029 | `EggBatchPanel.vue:6` | `<input v-model.number="settings.highpass_cutoff" type="number" min="1" max="47999"/>` | verified · 尾测 X09 |
| M03-C030 | `EggBatchPanel.vue:6` | `<select v-model="settings.gci_method"><option value="slope">斜率` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C031 | `EggBatchPanel.vue:6` | `<select v-model="settings.goi_method"><option value="scale">尺度` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C032 | `EggBatchPanel.vue:6` | `<input type="checkbox" :checked="all" :disabled="busy" @change="selected=all?[]:files.map(f=>f.id)"/>全选（{{selected.length}} / {{files.length}}）` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C033 | `EggBatchPanel.vue:6` | `<input v-model="selected" type="checkbox" :value="file.id" :disabled="busy"/><span>{{file.name}}` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C034 | `EggBatchPanel.vue:6` | `<button v-if="busy" @click="emit('cancel')">取消提交` | verified · 尾测 X11，受控故障/延迟，仅真实产物回读 |
| M03-C035 | `EggBatchPanel.vue:6` | `<button :disabled="busy" @click="emit('close')">返回工作台` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C036 | `EggBatchPanel.vue:6` | `<button class="primary" :disabled="busy&#124;&#124;!selected.length" @click="emit('submit',files.filter(f=>selected.includes(f.id)),settings)">{{busy?'正在提交…':'提交所选文件'}}` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C037 | `EggControls.vue:10` | `<input aria-label="EGG 选区起点" :value="start" type="number" min="0" :max="duration" step=".001" @change="emit('range',Number(($event.target as HTMLInputElement).value),Number(($event.target as HTMLInputElement).value)+end-start)"/>` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C038 | `EggControls.vue:11` | `<input aria-label="EGG 选区时长" :value="Number((end-start).toFixed(6))" type="number" min=".001" :max="duration" step=".001" @change="emit('range',start,start+Number(($event.target as HTMLInputElement).value))"/>` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C039 | `EggControls.vue:12` | `<input v-model.number="config.micro_width_ms" aria-label="EGG 微观窗口" type="number" min="5" max="5000" step="5"/>` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C040 | `EggControls.vue:16` | `<button :disabled="!canExport" @click="emit('save')">保存 CSV / 三图` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C041 | `EggControls.vue:16` | `<input v-model.number="order" type="number" min="1" max="256" placeholder="自动"/>` | verified · 尾测 X09 |
| M03-C042 | `EggControls.vue:16` | `<button :disabled="!canExport" @click="emit('inverse')">逆滤波 IF` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C043 | `EggInverseResult.vue:11` | `<button @click="save()" :disabled="saving">保存四图 PNG` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C044 | `EggInverseResult.vue:11` | `<button @click="save(0)" :disabled="saving">保存此图 PNG` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C045 | `EggInverseResult.vue:11` | `<button @click="save(1)" :disabled="saving">保存此图 PNG` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C046 | `EggInverseResult.vue:11` | `<button @click="save(2)" :disabled="saving">保存此图 PNG` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C047 | `EggInverseResult.vue:11` | `<button @click="save(3)" :disabled="saving">保存此图 PNG` | verified · 完整主流程，四图/正式任务/IF及5个PNG |
| M03-C048 | `EggParameters.vue:6` | `<input v-model.number="config.peak_prominence" type="number" min="0" max="10" step=".001" :disabled="config.auto_prominence"/>` | verified · 尾测 X09 |
| M03-C049 | `EggParameters.vue:6` | `<input v-model="config.auto_prominence" type="checkbox"/>自动` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C050 | `EggParameters.vue:6` | `<input v-model.number="config.valley_prominence" type="number" min="0" max="10" step=".001"/>` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C051 | `EggParameters.vue:7` | `<select v-model="config.gci_method"><option value="slope">斜率` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C052 | `EggParameters.vue:7` | `<select v-model="config.goi_method"><option value="slope">斜率` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C053 | `EggParameters.vue:9` | `<input v-model.number="config.spec_window_ms" type="number" min="5" max="50" step="1"/>` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C054 | `EggParameters.vue:9` | `<input v-model.number="config.spec_vmin" type="number" min="-160" max="20"/>` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C055 | `EggParameters.vue:9` | `<input v-model.number="config.spec_vmax" type="number" min="-160" max="20"/>` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C056 | `EggParameters.vue:10` | `<input v-model="config.keep_praat_f0" type="checkbox"/>Praat F0` | verified · X03，合法变更恢复/批次非法阈值拒绝 |
| M03-C057 | `EggParameters.vue:10` | `<input v-model="config.keep_gci_f0" type="checkbox"/>GCI F0` | verified · X03，合法变更恢复/批次非法阈值拒绝 |

## 公共、状态及性能验收

| ID | 可复现操作与判据 | 状态 |
| --- | --- | --- |
| M03-G01 | 空态与真实数据：1920×1000、1366×768、1280×720，100/125/150%缩放、浅深色、侧栏拖宽，截图及scrollHeight/clientHeight记录；普通滚轮不带动整页，小窗保存/取消可达 | verified · 四档CSS视口/浅深色/右栏折叠，真实Qt DPR1.5；其他OS/实体屏blocked |
| M03-G02 | 原始录音加载冷/热计时；按钮反馈20次、拖动缩放20次，记录median/p95/max与最后状态；目标见P17 G-P01–03 | verified · report.timings实际冷加载/20次更新；未承诺固定延迟 |
| M03-G03 | 全长、2/8/32/64/128倍至采样级连续波形、自适应振幅、零/正/负刻度，试听不变；科学显示抽点不进入导出 | verified · shared-waveform 1790855238430 + 本页实际图；科学数据未改 |
| M03-G04 | 全长/选区试听、暂停/停止、重复点击、音量、切声道、换文件/页签；只有一个播放归属 | verified · 尾测X10真实播放/暂停/继续/停止；X13公共音量/空格/手填/全部/进度；物理声卡听感未验 |
| M03-G05 | 缺输入、非法输入、错误后恢复、取消、迟到、关闭未保存，逐项记录受控注入与真实失败 | partial · 缺输入/非法值/取消/重试/草稿按C/F行证据，额外故障不扩为通过 |
| M03-G06 | 原文件前后SHA-256一致；输出只写output/validation/p17/M03，不得写原目录 | verified · originals-unchanged.json true；输出在本轮独占M01-M05目录，Mxx/evidence-index.json定位 |

V2 实际函数索引（用于追溯）：`__init__`, `apply_theme`, `__init__`, `__init__`, `init_ui`, `set_theme`, `_setup_initial_plots`, `load_wav_file`, `toggle_channel_flip`, `_load_data_from_path`, `_on_load_finished`, `_reset_app_state`, `_cancel_active_task`, `_finish_active_task`, `_on_task_progress`, `_on_task_error`, `_on_task_canceled`, `_get_downsampling_step`, `plot_timeline`, `update_timeline_roi_visual`, `update_roi_plots`, `_update_f0_contour`, `on_timeline_slider_change`, `on_timeline_click`, `on_left_plot_click`, `on_left_scroll`, `on_left_press`, `on_left_release`, `on_left_drag`, `update_zoom_plots`, `on_zoom_plot_click`, `on_zoom_scroll`, `on_zoom_press`, `on_zoom_release`, `on_zoom_drag`, `toggle_egg_display_mode`, `handle_highpass_slider_change`, `play_audio`, `stop_audio`, `toggle_f0_visibility`, `_calculate_f0`, `toggle_f0_correction`, `toggle_gci_method`, `toggle_goi_method`, `toggle_auto_prominence`, `handle_peak_prominence_change`, `handle_valley_prominence_change`, `handle_spec_window_change`, `_trigger_reanalysis`, `_on_events_finished`, `toggle_glottal_detection`, `run_inverse_filtering`, `save_analysis`, `_save_csv_data`, `_save_plots`, `show_batch_dialog`, `show_help_dialog`

## 本轮证据定位与现行布局

- 完整科学/主流程：`output/validation/p17/M01-M05-5cb4879d0220451da22cf911ce0043e1/report.json`，实际原生产物回读为同目录`readback.json`。
- 普通交互X01–X08：`output/validation/p17/M01-M05-f3d0eb386cd94cb3bc05ad939264ab78/report.json`，exit 0，页面错误0。
- 尾测X09：`output/validation/p17/M01-M05-be70b5cb449347c4a6ca8cbe695bdb7e/report.json` 中已完成checks；X09–X11及X08均完成，exit 0、页面错误0。
- 最后普通控件X12：`output/validation/p17/M01-M05-58f99d3ee30746759b579ca0bcd2d093/report.json`，exit 0、错误0；M05空历史刷新：`output/validation/p17/M01-M05-3e2a15f6e9d3487d8217e3b5516ce1f7/qt-report.json` success=true。
- 最终构建真实隐藏Qt：`output/validation/p17/M01-M05-86b574624cd247ee8f349bfa4610a162/qt-report.json`，success=true。
- 公共播放器/参数X13：`output/validation/p17/M01-M05-4595abcd041e427cbf7d3b143a245dbc/report.json`，80项逐一勾选、音量/空格/范围/进度，exit 0。
- 共享波形：`shared-waveform/1790855238430/report.json`，真实50000点、4组、24次Ctrl滚轮；共享语谱：`shared-spectrogram/1790853257028/report.json`；共享三栏：`shared-workbench/1790854845430/report.json`，均相对`output/validation/p17`。
- M03为明确的左右两栏例外，其他四页三栏。1920×1000、2560×1360、3840×2080为模拟CSS视口；1280×720保持滚动。右侧长历史/高级操作允许内部滚动。Qt DPR1.5，实体屏报告1707×1067，不冒充1920实体屏。
