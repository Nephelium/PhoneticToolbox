# P17 M04 LPC 谱图逐项验收规则

状态：限定 Windows 已执行，逐行状态见下表，未验范围保留。基线 0779bee；执行记录见同目录 M04-report.md。

已读原始手册 `../PhoneticToolbox_v2/Phonetic_Export/index.html` 第 11.1 / 11.6 / 11.7 节，及实际源码 `phonetic_toolbox/gui/widgets/lpc_spectrum_widget.py`。V2 源码 SHA-256：`6075118585718c790abd0b8416ddd6a536ea074a6a1adb662a2dfce1a5f242e5`。

现行P17要求优先于旧手册：1920×1000 CSS主工作区常用操作单页，普通滚轮不推动整页，小窗保持可达；波形连续、自适应可见振幅。V2原始目录自动保存改为受控显式保存。历史通过不算本轮实测。

## 可复现功能门

| ID | 手册 / V2定位 | V3控件及前置状态、步骤和判据 | 证据 / 状态 |
| --- | --- | --- | --- |
| M04-F01 加载、刷新与TextGrid | 11.1 / 11.6 / 11.7 / `lpc_spectrum_widget.py` | 加载短真实录音/同名TextGrid，手动取消关联和下一层；真实边界与层名可读 | verified · 主流程/X09目录刷新/关联/层选择 |
| M04-F02 时间范围 | 11.1 / 11.6 / 11.7 / `lpc_spectrum_widget.py` | 手输起止、标签选择、清除选区；清除后分析波形可见范围，最多48000样本，边界按采样点半开区间 | verified · .2–.4秒8820样本真实ROI，X03清除；边界状态测试 |
| M04-F03 波形与语谱手势 | 11.1 / 11.6 / 11.7 / `lpc_spectrum_widget.py` | 左拖/Shift拖动/滚轮/键盘、语谱选择，与公共P17现行手势一致；时间和频率视图互不污染 | partial · X01共轴20次及公共波形手势；时间/频率独立由状态测试 |
| M04-F04 LPC参数 | 11.1 / 11.6 / 11.7 / `lpc_spectrum_widget.py` | 阶数1/50/200、频率100/8000/48000、dB上下限、动态纵轴；非法阶数/幅度/ROI拒绝且旧结果保留 | verified · X03合法字段/dynamic + X04阶数0拒绝；预算边界状态测试 |
| M04-F05 科学计算 | 11.1 / 11.6 / 11.7 / `lpc_spectrum_widget.py` | 真实短片段计算1024点LPC，记录参数/样本/墙钟，与V2同输入同配置数值比较 | verified · 频率1024+谱值1024与原V2 np.array_equal |
| M04-F06 结果切换 | 11.1 / 11.6 / 11.7 / `lpc_spectrum_widget.py` | 波形↔LPC频谱、参数变更后旧结果标过期；试听结果对应单声道均值片段 | verified · X03波形/频谱/历史，旧结果标记由状态测试 |
| M04-F07 完整保存 | 11.1 / 11.6 / 11.7 / `lpc_spectrum_widget.py` | PNG/选区WAV/JSON实际保存，PNG白底黑线300DPI、TextGrid标签入名，JSON频率与谱值/来源可回读 | verified · PNG2400×1350、8820样本WAV、完整JSON回读 |
| M04-F08 任务和故障 | 11.1 / 11.6 / 11.7 / `lpc_spectrum_widget.py` | 开始、取消、重复提交、失败重试、历史读取、迟到文件切换；切换后的任务不覆盖当前文件 | verified · X04真实任务取消/重试 + X03历史 |
| M04-F09 草稿 | 11.1 / 11.6 / 11.7 / `lpc_spectrum_widget.py` | 保存参数草稿，关闭未保存保护；恢复类型和数值一致，保存失败反馈可见 | verified · X03保存草稿 + X08关闭取消/保存重开 |

## 每个实际控件的清单

下表为初次审阅时逐模板控件索引，行号可能随最终布局移动；控件名与绑定为稳定定位。每种普通控件检验正常路径及关键失败/取消，额外数值组合不等同于新的未测控件。动态v-for控件逐文件/参数/任务实例应用同一判据。共享WaveformViewport、AudioTransport、TaskPanel及参数抽屉按总规则复用，不能据模板存在判通过。

| ID | V3源文件/行 | 控件及绑定 | 实际状态 |
| --- | --- | --- | --- |
| M04-C001 | `LpcSpectrumPage.vue:78` | `<button v-if="context.files.choose" @click="choose">打开 WAV 目录` | verified · 完整主流程 + LPC/V2/产物回读 |
| M04-C002 | `LpcSpectrumPage.vue:78` | `<input ref="picker" type="file" accept=".wav,.TextGrid" multiple hidden @change="addFiles"/><button @click="picker?.click()">导入文件` | not_applicable · Windows 本机入口不提供此网页上传/下载控件，网页服务另列未准入 |
| M04-C003 | `LpcSpectrumPage.vue:78` | `<button @click="refresh">刷新文件` | verified · 尾测 X09 |
| M04-C004 | `LpcSpectrumPage.vue:78` | `<button @click="emit('references')">方法与引用` | verified · X07 |
| M04-C005 | `LpcSpectrumPage.vue:82` | `<select aria-label="LPC 音频文件" :value="source?.id??''" @change="load(($event.target as HTMLSelectElement).value)"><option disabled value="">选择 WAV` | verified · 完整主流程 + LPC/V2/产物回读 |
| M04-C006 | `LpcSpectrumPage.vue:84` | `<select aria-label="LPC TextGrid" :value="grid?.id??''" :disabled="!wave.asset" @change="associate(($event.target as HTMLSelectElement).value)"><option value="">不关联` | verified · 尾测 X09 |
| M04-C007 | `LpcSpectrumPage.vue:85` | `<select v-model="tier" aria-label="LPC 标注层" :disabled="!tiers.length&#124;&#124;gridLoading"><option v-if="!tiers.length" value="">无标注` | verified · 尾测 X09 |
| M04-C008 | `LpcSpectrumPage.vue:85` | `<button :disabled="tiers.length<2" @click="tier=tiers[(tiers.findIndex(t=>t.name===tier)+1)%tiers.length].name">下一层` | verified · 尾测 X09 |
| M04-C009 | `LpcSpectrumPage.vue:89` | `<input v-model="start" aria-label="LPC 选区起点" type="number" min="0" step=".001" :disabled="!wave.asset" @input="editRange"/>` | verified · 完整主流程 + LPC/V2/产物回读 |
| M04-C010 | `LpcSpectrumPage.vue:89` | `<input v-model="end" aria-label="LPC 选区终点" type="number" min="0" step=".001" :disabled="!wave.asset" @input="editRange"/>` | verified · 完整主流程 + LPC/V2/产物回读 |
| M04-C011 | `LpcSpectrumPage.vue:89` | `<button :disabled="!wave.asset" @click="clearSelection">清除选区` | verified · X03 + X04 + X08 |
| M04-C012 | `LpcSpectrumPage.vue:90` | `<button class="primary" :disabled="loading&#124;&#124;gridLoading&#124;&#124;submitting&#124;&#124;!!liveJob&#124;&#124;!tasks?.lpc&#124;&#124;!wave.asset" @click="submit">{{submitting?'正在提交…':'开始分析'}}` | verified · 完整主流程 + LPC/V2/产物回读 |
| M04-C013 | `LpcSpectrumPage.vue:90` | `<button v-if="submitting&#124;&#124;liveJob" :disabled="cancelPending" @click="cancel()">取消分析` | verified · X04 真实取消/重试 |
| M04-C014 | `LpcSpectrumPage.vue:92` | `<select v-model.number="wave.channel" @change="stop"><option v-for="(_,i) in wave.asset.channels" :key="i" :value="i">{{i+1}}` | verified · X12真实77秒双声道预览切换2/1恢复，未提交全长LPC |
| M04-C015 | `LpcSpectrumPage.vue:93` | `<button @click="save">保存参数草稿` | verified · X03 + X04 + X08 |
| M04-C016 | `LpcSpectrumPage.vue:93` | `<input v-model="draft.order" aria-label="LPC 阶数" type="number" min="1" max="200"/>` | verified · X03 + X04 + X08 |
| M04-C017 | `LpcSpectrumPage.vue:93` | `<input v-model="draft.freq_max_hz" aria-label="LPC 频率上限" type="number" min="100" max="48000"/>` | verified · X03 + X04 + X08 |
| M04-C018 | `LpcSpectrumPage.vue:93` | `<input v-model="draft.amp_min_db" aria-label="LPC 幅度下限" :disabled="draft.dynamic_y" type="number" min="-200" max="100"/>` | verified · X03 + X04 + X08 |
| M04-C019 | `LpcSpectrumPage.vue:93` | `<input v-model="draft.amp_max_db" aria-label="LPC 幅度上限" :disabled="draft.dynamic_y" type="number" min="-200" max="100"/>` | verified · X03 + X04 + X08 |
| M04-C020 | `LpcSpectrumPage.vue:93` | `<input v-model="draft.dynamic_y" type="checkbox"/>动态纵轴` | verified · X03 + X04 + X08 |
| M04-C021 | `LpcSpectrumPage.vue:96` | `<button :aria-pressed="mode==='wave'" @click="mode='wave'">波形` | verified · X03 + X04 + X08 |
| M04-C022 | `LpcSpectrumPage.vue:96` | `<button :aria-pressed="mode==='spectrum'" :disabled="!result" @click="mode='spectrum'">LPC 频谱` | verified · X03 + X04 + X08 |
| M04-C023 | `LpcSpectrumPage.vue:96` | `<input v-model="wave.showSpectrogram" type="checkbox" :disabled="!context.files.spectrogram"/>显示语谱图（Praat）` | verified · X01 共轴20次真实语谱画布 |
| M04-C024 | `LpcSpectrumPage.vue:103` | `<button v-if="tasks?.saveJob" :disabled="saving&#124;&#124;reading" @click="saveResult">选择目录保存完整结果` | verified · 完整主流程 + LPC/V2/产物回读 |
| M04-C025 | `LpcSpectrumPage.vue:103` | `<button v-for="file in resultJob.result_manifest.files" :key="file.id" @click="download(file.id,result!.export_names[file.name]??file.name)">下载 {{file.name.endsWith('.png')?'PNG':file.name.endsWith('.wav')?'选区 WAV':'参数与谱值 JSON'}}` | not_applicable · Windows 本机入口不提供此网页上传/下载控件，网页服务另列未准入 |
| M04-C026 | `LpcSpectrumPage.vue:103` | `<summary>查看此结果的参数快照` | verified · 尾测 X09 |
| M04-C027 | `LpcSpectrumPage.vue:104` | `<summary>分析任务与历史结果（{{jobs.length}}）` | verified · X03 + X04 + X08 |
| M04-C028 | `LpcSpectrumPage.vue:104` | `<button v-for="job in jobs.filter(j=>j.state==='succeeded')" :key="job.id" @click="show(job)">查看结果 {{new Date(job.created_at*1000).toLocaleTimeString()}} · {{job.id.slice(0,8)}}` | verified · X03 + X04 + X08 |
| M04-C029 | `SpectrumPlot.vue:16` | `<button aria-label="缩小 LPC 频谱" :disabled="zoom<=1" @click="scale(zoom/2)">−` | verified · 尾测 X09 |
| M04-C030 | `SpectrumPlot.vue:16` | `<button aria-label="放大 LPC 频谱" :disabled="zoom>=64" @click="scale(zoom*2)">+` | verified · 尾测 X09 |
| M04-C031 | `SpectrumPlot.vue:16` | `<button @click="reset">适合频率范围` | verified · 尾测 X09 |

## 公共、状态及性能验收

| ID | 可复现操作与判据 | 状态 |
| --- | --- | --- |
| M04-G01 | 空态与真实数据：1920×1000、1366×768、1280×720，100/125/150%缩放、浅深色、侧栏拖宽，截图及scrollHeight/clientHeight记录；普通滚轮不带动整页，小窗保存/取消可达 | verified · 四档CSS视口/浅深色/右栏折叠，真实Qt DPR1.5；其他OS/实体屏blocked |
| M04-G02 | 原始录音加载冷/热计时；按钮反馈20次、拖动缩放20次，记录median/p95/max与最后状态；目标见P17 G-P01–03 | verified · report.timings实际冷加载/20次更新；未承诺固定延迟 |
| M04-G03 | 全长、2/8/32/64/128倍至采样级连续波形、自适应振幅、零/正/负刻度，试听不变；科学显示抽点不进入导出 | verified · shared-waveform 1790855238430 + 本页实际图；科学数据未改 |
| M04-G04 | 全长/选区试听、暂停/停止、重复点击、音量、切声道、换文件/页签；只有一个播放归属 | verified · 尾测X10真实播放/暂停/继续/停止；X13公共音量/空格/手填/全部/进度；物理声卡听感未验 |
| M04-G05 | 缺输入、非法输入、错误后恢复、取消、迟到、关闭未保存，逐项记录受控注入与真实失败 | partial · 缺输入/非法值/取消/重试/草稿按C/F行证据，额外故障不扩为通过 |
| M04-G06 | 原文件前后SHA-256一致；输出只写output/validation/p17/M04，不得写原目录 | verified · originals-unchanged.json true；输出在本轮独占M01-M05目录，Mxx/evidence-index.json定位 |

V2 实际函数索引（用于追溯）：`__init__`, `init_ui`, `resizeEvent`, `_browse_input`, `_browse_output`, `_refresh_files`, `_on_file_selected`, `_plot_waveform`, `_draw_selection_overlay`, `_play_current_file`, `_stop_audio`, `_on_player_position_changed`, `_on_textgrid_button_clicked`, `_draw_textgrid`, `_start_lpc_processing`, `_plot_lpc_curve`, `_apply_axes_theme`, `_ensure_plot_margins`, `_create_export_figure`, `_open_help`, `_is_shift_pressed`, `_on_scroll`, `_on_press`, `_on_release`, `_on_motion`, `_return_to_waveform`, `set_theme`

## 本轮证据定位与现行布局

- 完整科学/主流程：`output/validation/p17/M01-M05-5cb4879d0220451da22cf911ce0043e1/report.json`，实际原生产物回读为同目录`readback.json`。
- 普通交互X01–X08：`output/validation/p17/M01-M05-f3d0eb386cd94cb3bc05ad939264ab78/report.json`，exit 0，页面错误0。
- 尾测X09：`output/validation/p17/M01-M05-be70b5cb449347c4a6ca8cbe695bdb7e/report.json` 中已完成checks；X09–X11及X08均完成，exit 0、页面错误0。
- 最后普通控件X12：`output/validation/p17/M01-M05-58f99d3ee30746759b579ca0bcd2d093/report.json`，exit 0、错误0；M05空历史刷新：`output/validation/p17/M01-M05-3e2a15f6e9d3487d8217e3b5516ce1f7/qt-report.json` success=true。
- 最终构建真实隐藏Qt：`output/validation/p17/M01-M05-86b574624cd247ee8f349bfa4610a162/qt-report.json`，success=true。
- 公共播放器/参数X13：`output/validation/p17/M01-M05-4595abcd041e427cbf7d3b143a245dbc/report.json`，80项逐一勾选、音量/空格/范围/进度，exit 0。
- 共享波形：`shared-waveform/1790855238430/report.json`，真实50000点、4组、24次Ctrl滚轮；共享语谱：`shared-spectrogram/1790853257028/report.json`；共享三栏：`shared-workbench/1790854845430/report.json`，均相对`output/validation/p17`。
- M03为明确的左右两栏例外，其他四页三栏。1920×1000、2560×1360、3840×2080为模拟CSS视口；1280×720保持滚动。右侧长历史/高级操作允许内部滚动。Qt DPR1.5，实体屏报告1707×1067，不冒充1920实体屏。
