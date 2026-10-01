# P17 M01 参数估计逐项验收规则

状态：限定 Windows 已执行，逐行状态见下表，未验范围保留。基线 0779bee；执行记录见同目录 M01-report.md。

已读原始手册 `../PhoneticToolbox_v2/Phonetic_Export/index.html` 第 2.1 / 7.2 节，及实际源码 `phonetic_toolbox/gui/widgets/parameter_estimation_widget.py`。V2 源码 SHA-256：`98d83c3dafbbca45c823b5e6aef2d113e3239e4f2eb5ad4dfe576da0afcdce5a`。

现行P17要求优先于旧手册：1920×1000 CSS主工作区常用操作单页，普通滚轮不推动整页，小窗保持可达；波形连续、自适应可见振幅。V2原始目录自动保存改为受控显式保存。历史通过不算本轮实测。

## 可复现功能门

| ID | 手册 / V2定位 | V3控件及前置状态、步骤和判据 | 证据 / 状态 |
| --- | --- | --- | --- |
| M01-F01 目录、递归与关联 | 2.1 / 7.2 / `parameter_estimation_widget.py` | 选择目录→切换包含子文件夹→刷新；音频、TextGrid、唇形、参数表按同源文件关联，取消目录选择保留旧列表 | verified · 主流程/X03/X09 + X12受控null取消；唇形单列blocked |
| M01-F02 列表选择 | 2.1 / 7.2 / `parameter_estimation_widget.py` | 单击当前录音、逐项勾选、全选/取消全选；切分只作用于勾选集合，无勾选时当前文件；全列表分析不等于勾选分析 | verified · X03 列表操作 + 完整切分 |
| M01-F03 参数选择80项 | 2.1 / 7.2 / `parameter_estimation_widget.py` | 打开选择输出参数，按分组/搜索选择、全选清空、应用/取消；空选择拒绝提交，取消不修改原配置 | verified · 主流程/X04；逐字段枚举见下表 |
| M01-F04 设置14项 | 2.1 / 7.2 / `parameter_estimation_widget.py` | 逐一修改14项设置并应用/取消；非法范围保持对话框且不发任务，保存草稿与关闭未保存保护可恢复 | verified · 主流程/X04/X08，非法帧移0拒绝 |
| M01-F05 TextGrid层与片段 | 2.1 / 7.2 / `parameter_estimation_widget.py` | 选择层、点击标签、展开完整列表；波形选区与精确时间一致，空标签策略明确 | verified · X03/X09 + 精确切分回读 |
| M01-F06 真实切分 | 2.1 / 7.2 / `parameter_estimation_widget.py` | 复制短录音和原TextGrid到独占目录，选择输出目录并执行切分；回读每段采样及TextGrid边界，文件名含原名/层/标签/时间 | verified · 完整主流程，3段68916样本与原PCM精确一致 |
| M01-F07 参数同步切分 | 2.1 / 7.2 / `parameter_estimation_widget.py` | 先真实参数提取，再勾选同时切分参数结果；回读WAV/XLSX/SQLite时间和字段；历史来源显式标明未核实 | verified · 完整主流程，最近同源父结果同步切分；旧表选择X09 |
| M01-F08 计算/取消/重试 | 2.1 / 7.2 / `parameter_estimation_widget.py` | 真实列表分析，观察运行和完成，期间试听；取消不伪成功，失败重试拥有新任务且旧结果可读 | verified · 全批次实际完成/预算失败及X06取消重试 |
| M01-F09 结果保存与回读 | 2.1 / 7.2 / `parameter_estimation_widget.py` | 保存XLSX和SQLite，回读字段数、时间、空值、acoustic/2与后端；重复保存不同内容不覆盖 | verified · 4组XLSX/SQLite合计42895单元格回读，acoustic/2 |
| M01-F10 旧PKL兼容 | 2.1 / 7.2 / `parameter_estimation_widget.py` | 只有授权真实旧PKL方可转换；输出新lip.json且原PKL哈希不变；无素材则blocked | blocked · 批准目录无真实PKL |

## 每个实际控件的清单

下表为初次审阅时逐模板控件索引，行号可能随最终布局移动；控件名与绑定为稳定定位。每种普通控件检验正常路径及关键失败/取消，额外数值组合不等同于新的未测控件。动态v-for控件逐文件/参数/任务实例应用同一判据。共享WaveformViewport、AudioTransport、TaskPanel及参数抽屉按总规则复用，不能据模板存在判通过。

| ID | V3源文件/行 | 控件及绑定 | 实际状态 |
| --- | --- | --- | --- |
| M01-C001 | `BatchResults.vue:43` | `<select :value="active?.id" :disabled="busy" @change="$emit('select',($event.target as HTMLSelectElement).value)"><option v-for="batch in batches" :key="batch.id" :value="batch.id">{{batch.operation==='acoustic_analysis'?'参数分析':'T` | verified · 尾测 X09 |
| M01-C002 | `BatchResults.vue:46` | `<button v-if="!active.summary.closed" :disabled="busy" @click="$emit('cancel')">取消后续处理` | verified · X06 真实批次取消/单文件重试 |
| M01-C003 | `BatchResults.vue:46` | `<button v-if="desktop&&active.summary.counts.succeeded" :disabled="busy" @click="$emit('save')">保存已完成结果` | verified · 完整主流程 + 原生产物回读 |
| M01-C004 | `BatchResults.vue:48` | `<button v-if="['failed','interrupted','cancelled'].includes(item.state)&&item.job_id&&active.summary.closed" :disabled="busy" @click="$emit('retry',item.job_id)">重试此文件` | verified · X06 真实批次取消/单文件重试 |
| M01-C005 | `BatchResults.vue:49` | `<button v-for="file in job.result_manifest.files" :key="file.id" :disabled="busy" @click="$emit('download',file.id,downloadName(active,job,file.name))">{{downloadName(active,job,file.name)}}` | not_applicable · Windows 本机入口不提供此网页上传/下载控件，网页服务另列未准入 |
| M01-C006 | `ParameterEstimationPage.vue:140` | `<button class="primary" :disabled="busy" @click="choose('input')">选择音频目录` | verified · 完整主流程 + 原生产物回读 |
| M01-C007 | `ParameterEstimationPage.vue:140` | `<button :disabled="busy" @click="choose('association')">选择关联目录` | verified · 尾测 X09 |
| M01-C008 | `ParameterEstimationPage.vue:141` | `<input ref="picker" class="visually-hidden" type="file" accept=".wav" multiple aria-label="选择音频列表" @change="add"/><button class="primary" @click="picker?.click()">添加WAV到列表` | not_applicable · Windows 本机入口不提供此网页上传/下载控件，网页服务另列未准入 |
| M01-C009 | `ParameterEstimationPage.vue:143` | `<input v-model="state.recursive" type="checkbox" :disabled="busy" @change="refresh"/>包含子文件夹` | verified · X03 |
| M01-C010 | `ParameterEstimationPage.vue:144` | `<button :disabled="busy&#124;&#124;(context.files.kind==='desktop'&&!state.input)" @click="refresh">{{busy?'正在读取…':'刷新列表'}}` | verified · X03 |
| M01-C011 | `ParameterEstimationPage.vue:145` | `<input v-model="state.sameDirectory" type="checkbox"/>结果与WAV同目录` | verified · X06 关闭同目录并选择独占保存目录 |
| M01-C012 | `ParameterEstimationPage.vue:145` | `<button :disabled="state.sameDirectory&#124;&#124;busy" @click="choose('output')">选择结果目录` | verified · 完整主流程 + 原生产物回读 |
| M01-C013 | `ParameterEstimationPage.vue:145` | `<button @click="emit('references')"><AppIcon name="book"/>方法与引用` | verified · X07 |
| M01-C014 | `ParameterEstimationPage.vue:149` | `<input aria-label="全选音频" type="checkbox" :checked="audioFiles.length>0&&state.marked.length===audioFiles.length" :indeterminate="state.marked.length>0&&state.marked.length<audioFiles.length" :disabled="!audioFiles.length" @change=` | verified · X03 |
| M01-C015 | `ParameterEstimationPage.vue:149` | `<input v-model="state.marked" type="checkbox" :value="file.id" :aria-label="'选择切分 '+file.name"/><button class="file-row" :class="{selected:state.selected===file.id}" :aria-pressed="state.selected===file.id" @click="selectFile(file` | verified · X03 |
| M01-C016 | `ParameterEstimationPage.vue:152` | `<select aria-label="TextGrid关联" :value="linked.textgrid?.id??''" :disabled="state.wave.loading" @change="changeAssociation('textgrid',$event)"><option value="">不关联` | verified · X03 |
| M01-C017 | `ParameterEstimationPage.vue:152` | `<select aria-label="唇形关联" :value="linked.lip?.id??''" @change="changeAssociation('lip',$event)"><option value="">不关联` | blocked · 批准目录无真实 lip.json/旧 PKL；空列表与禁用可见 |
| M01-C018 | `ParameterEstimationPage.vue:153` | `<summary>关联说明与旧文件兼容` | verified · X03 |
| M01-C019 | `ParameterEstimationPage.vue:155` | `<select v-model="pickleId" aria-label="旧唇形PKL"><option value="">请选择旧文件` | blocked · 批准目录无真实 lip.json/旧 PKL；空列表与禁用可见 |
| M01-C020 | `ParameterEstimationPage.vue:155` | `<button :disabled="converting&#124;&#124;!pickleId&#124;&#124;busy" @click="convertLip">{{converting?'正在转换…':'转换并保存 .lip.json'}}` | blocked · 批准目录无真实 lip.json/旧 PKL；空列表与禁用可见 |
| M01-C021 | `ParameterEstimationPage.vue:159` | `<select v-model.number="state.wave.channel" @change="stop"><option v-for="(_,i) in state.wave.asset.channels" :key="i" :value="i">声道 {{i+1}}` | verified · X12真实77秒双声道切换2/1恢复 |
| M01-C022 | `ParameterEstimationPage.vue:160` | `<select v-model.number="linked.layer"><option v-for="(tier,i) in linked.tiers" :key="i" :value="i">{{tier.name}}` | verified · 尾测 X09 |
| M01-C023 | `ParameterEstimationPage.vue:160` | `<summary>查看完整标签列表与精确时间` | verified · X03 |
| M01-C024 | `ParameterEstimationPage.vue:160` | `<button v-for="(interval,i) in intervals" :key="i" @click="intervalSelect(interval.xmin,interval.xmax)"><span class="ipa-sample">{{interval.text&#124;&#124;'（空标签）'}}` | verified · X03 |
| M01-C025 | `ParameterEstimationPage.vue:160` | `<input v-model="sliceParameters" type="checkbox"/>同时切分参数结果` | verified · 完整主流程 + 原生产物回读 |
| M01-C026 | `ParameterEstimationPage.vue:161` | `<select v-model="parameterSource" aria-label="切分参数来源"><option value="recent">本应用最近一次同源完整结果` | verified · 尾测 X09 |
| M01-C027 | `ParameterEstimationPage.vue:162` | `<select aria-label="历史参数表关联" :value="linked.legacy?.id??''" @change="linked.legacy=state.files.find(f=>f.id===($event.target as HTMLSelectElement).value&&f.kind==='parameter')??null"><option value="">未指定` | verified · 尾测 X09 |
| M01-C028 | `ParameterEstimationPage.vue:164` | `<button :disabled="!taskReady&#124;&#124;taskBusy&#124;&#124;busy" @click="startBatch('textgrid_segment')">保存当前层切分音频` | verified · 完整主流程 + 原生产物回读 |
| M01-C029 | `ParameterEstimationPage.vue:166` | `<button @click="openParameters">选择输出参数` | verified · 主流程 + X04，14字段修改/清空恢复、80候选/全选清空及非法帧移；X13逐个勾选全部80项 |
| M01-C030 | `ParameterEstimationPage.vue:166` | `<button @click="openSettings">编辑14项设置` | verified · 主流程 + X04，14字段修改/清空恢复、80候选/全选清空及非法帧移；X13逐个勾选全部80项 |
| M01-C031 | `ParameterEstimationPage.vue:166` | `<button :disabled="!dirty(state)" @click="save">保存草稿` | verified · 主流程 + X08 保存关闭/重开 |
| M01-C032 | `ParameterEstimationPage.vue:166` | `<button class="primary" :disabled="!taskReady&#124;&#124;taskBusy&#124;&#124;busy&#124;&#124;!audioFiles.length" @click="startBatch('acoustic_analysis')">开始全列表分析` | verified · 完整主流程 + 原生产物回读 |

## 公共、状态及性能验收

| ID | 可复现操作与判据 | 状态 |
| --- | --- | --- |
| M01-G01 | 空态与真实数据：1920×1000、1366×768、1280×720，100/125/150%缩放、浅深色、侧栏拖宽，截图及scrollHeight/clientHeight记录；普通滚轮不带动整页，小窗保存/取消可达 | verified · 四档CSS视口/浅深色/右栏折叠，真实Qt DPR1.5；其他OS/实体屏blocked |
| M01-G02 | 原始录音加载冷/热计时；按钮反馈20次、拖动缩放20次，记录median/p95/max与最后状态；目标见P17 G-P01–03 | verified · report.timings实际冷加载/20次更新；未承诺固定延迟 |
| M01-G03 | 全长、2/8/32/64/128倍至采样级连续波形、自适应振幅、零/正/负刻度，试听不变；科学显示抽点不进入导出 | verified · shared-waveform 1790855238430 + 本页实际图；科学数据未改 |
| M01-G04 | 全长/选区试听、暂停/停止、重复点击、音量、切声道、换文件/页签；只有一个播放归属 | verified · 尾测X10真实播放/暂停/继续/停止；X13公共音量/空格/手填/全部/进度；物理声卡听感未验 |
| M01-G05 | 缺输入、非法输入、错误后恢复、取消、迟到、关闭未保存，逐项记录受控注入与真实失败 | partial · 缺输入/非法值/取消/重试/草稿按C/F行证据，额外故障不扩为通过 |
| M01-G06 | 原文件前后SHA-256一致；输出只写output/validation/p17/M01，不得写原目录 | verified · originals-unchanged.json true；输出在本轮独占M01-M05目录，Mxx/evidence-index.json定位 |

V2 实际函数索引（用于追溯）：`__init__`, `run`, `_emit_progress`, `__init__`, `init_ui`, `_browse_input`, `_browse_output`, `_toggle_output_dir`, `_refresh_files`, `_on_selection_changed`, `_plot_waveform`, `_update_plot`, `_on_scroll`, `_on_press`, `_on_release`, `_on_motion`, `_play_audio`, `_stop_audio`, `_read_textgrid`, `_toggle_segmentation`, `_save_segmented_audio`, `_read_lip_data`, `_open_settings`, `_open_help`, `_open_parameter_selection`, `_open_parameter_help`, `_start_processing`, `set_theme`, `_on_progress`, `_on_finished`

## 本轮证据定位与现行布局

- 完整科学/主流程：`output/validation/p17/M01-M05-5cb4879d0220451da22cf911ce0043e1/report.json`，实际原生产物回读为同目录`readback.json`。
- 普通交互X01–X08：`output/validation/p17/M01-M05-f3d0eb386cd94cb3bc05ad939264ab78/report.json`，exit 0，页面错误0。
- 尾测X09：`output/validation/p17/M01-M05-be70b5cb449347c4a6ca8cbe695bdb7e/report.json` 中已完成checks；X09–X11及X08均完成，exit 0、页面错误0。
- 最后普通控件X12：`output/validation/p17/M01-M05-58f99d3ee30746759b579ca0bcd2d093/report.json`，exit 0、错误0；M05空历史刷新：`output/validation/p17/M01-M05-3e2a15f6e9d3487d8217e3b5516ce1f7/qt-report.json` success=true。
- 最终构建真实隐藏Qt：`output/validation/p17/M01-M05-86b574624cd247ee8f349bfa4610a162/qt-report.json`，success=true。
- 公共播放器/参数X13：`output/validation/p17/M01-M05-4595abcd041e427cbf7d3b143a245dbc/report.json`，80项逐一勾选、音量/空格/范围/进度，exit 0。
- 共享波形：`shared-waveform/1790855238430/report.json`，真实50000点、4组、24次Ctrl滚轮；共享语谱：`shared-spectrogram/1790853257028/report.json`；共享三栏：`shared-workbench/1790854845430/report.json`，均相对`output/validation/p17`。
- M03为明确的左右两栏例外，其他四页三栏。1920×1000、2560×1360、3840×2080为模拟CSS视口；1280×720保持滚动。右侧长历史/高级操作允许内部滚动。Qt DPR1.5，实体屏报告1707×1067，不冒充1920实体屏。

## 14设置与80输出参数逐字段索引

X04对14设置各做合法修改/清空恢复或开关来回，主流程额外验证帧移0拒绝。参数抽屉对全部80项全选/清空/搜索枚举，X13对全部80项逐个取消/勾选恢复；同类型控件正常路径不等于每项科学有效值都可从本录音检出。缺唇形数据对应列可为空。

| 设置字段 | 默认 | 允许范围 | 本轮 |
| --- | --- | --- | --- |
| `energy_window_ms` | 40.0 | 1–1000 / number | verified X04；范围状态测试 |
| `frameshift_ms` | 5.0 | 0.1–1000 / number | verified X04；范围状态测试 |
| `lip_smooth_win_size` | 0 | 0–100 / integer | verified X04；范围状态测试 |
| `max_f0` | 880.0 | 50–2000 / number | verified X04；范围状态测试 |
| `max_formant` | 6000.0 | >0–10000 / number | verified X04；范围状态测试 |
| `min_f0` | 60.0 | 10–1000 / number | verified X04；范围状态测试 |
| `n_periods` | 3 | 1–100 / integer | verified X04；范围状态测试 |
| `num_formants` | 5 | 3–10 / integer | verified X04；范围状态测试 |
| `only_voiced` | True | >– / boolean | verified X04；范围状态测试 |
| `reaper_hilbert` | True | >– / boolean | verified X04；范围状态测试 |
| `reaper_no_highpass` | False | >– / boolean | verified X04；范围状态测试 |
| `silence_threshold` | 0.03 | 0–1 / number | verified X04；范围状态测试 |
| `smooth_win_size` | 10 | 1–100 / integer | verified X04；范围状态测试 |
| `windowsize_ms` | 40.0 | 1–1000 / number | verified X04；范围状态测试 |

| 输出参数 | UI选择 | 科学值范围 |
| --- | --- | --- |
| `pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `pF1` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `pF2` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `pF3` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `pF4` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `pB1` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `pB2` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `pB3` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `pB4` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H2_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H2_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H4_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H4_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `A1_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `A1_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `A2_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `A2_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `A3_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `A3_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1H2u_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1H2u_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H2H4u_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H2H4u_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1A1u_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1A1u_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1A2u_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1A2u_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1A3u_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1A3u_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1A1c_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1A1c_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1A2c_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1A2c_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1A3c_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1A3c_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1H2c_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H1H2c_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H2H4c_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H2H4c_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H2K_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H2K_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H5K_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H5K_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H42Ku_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H42Ku_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H2KH5Ku_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H2KH5Ku_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H42Kc_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H42Kc_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H2KH5Kc_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `H2KH5Kc_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `CPP_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `CPP_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `Intensity` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `HNR05_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `HNR15_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `HNR25_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `HNR35_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `HNR05_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `HNR15_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `HNR25_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `HNR35_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `SHR_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `SHR_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `SpectralSlope_pF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `SpectralSlope_rF0` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `Jitter_Local` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `Jitter_RAP` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `Jitter_PPQ5` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `Shimmer_Local` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `Shimmer_APQ3` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `Shimmer_APQ5` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `Shimmer_APQ11` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `LipArea` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `LipWidth` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `LipOpen` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
| `LipCirc` | verified 全选/清空/枚举X04 + 逐项勾选X13 | 真实完整结果/空值见readback，不合成补值 |
