# P17 M02 参数显示逐项验收规则

状态：限定 Windows 已执行，逐行状态见下表，未验范围保留。基线 0779bee；执行记录见同目录 M02-report.md。

已读原始手册 `../PhoneticToolbox_v2/Phonetic_Export/index.html` 第 2.2 节，及实际源码 `phonetic_toolbox/gui/widgets/parameter_display_widget.py`。V2 源码 SHA-256：`5a47c2a92b96d59a10bbbaafca6cea1fe6a5deaba50b76321c691acceb0af89c`。

现行P17要求优先于旧手册：1920×1000 CSS主工作区常用操作单页，普通滚轮不推动整页，小窗保持可达；波形连续、自适应可见振幅。V2原始目录自动保存改为受控显式保存。历史通过不算本轮实测。

## 可复现功能门

| ID | 手册 / V2定位 | V3控件及前置状态、步骤和判据 | 证据 / 状态 |
| --- | --- | --- | --- |
| M02-F01 双目录与过滤 | 2.2 / `parameter_display_widget.py` | 音频目录/参数目录/递归/刷新逐一操作；搜索不存在、中文、恢复空搜索；无参数时波形可用且不生成假曲线 | verified · 主流程/X03/X09，参数表空关联后恢复 |
| M02-F02 初始空图 | 2.2 / `parameter_display_widget.py` | 读取真实参数表；首图无自动曲线，逐项勾选后显式分配才绘图 | verified · 主流程首图0曲线，显式分配 |
| M02-F03 候选筛选 | 2.2 / `parameter_display_widget.py` | 输入参数搜索，切换reaper/correction，全选可见/清空；筛选保留已有图窗和勾选，不丢数据 | verified · X03 |
| M02-F04 图窗管理 | 2.2 / `parameter_display_widget.py` | 新建、目标选择、分配、移除勾选、清空、合并、删除最后一图再新建；无效目标禁用分配，曲线按图窗正确归属 | verified · 主流程/X03/X09；状态测试最后一图删除 |
| M02-F05 图窗缩放 | 2.2 / `parameter_display_widget.py` | 放大图窗→还原；当前选区、时间窗、曲线和标签保持，弹窗可关闭 | verified · X03 放大/还原 |
| M02-F06 参数坐标 | 2.2 / `parameter_display_widget.py` | 检查多曲线叠加、图例、真实时间、不同单位轴、null缺口；显示抽点不得进入导出 | partial · 真实曲线/时间/图像已验；多单位/null数值规则由39项状态测试覆盖 |
| M02-F07 时间交互 | 2.2 / `parameter_display_widget.py` | 起点/长度输入及边界、图上Ctrl滚轮/左右拖动；所有图窗/波形时间同步，普通滚轮遵循当前P17规则 | verified · X01 20次共轴 + X09手填时间窗 |
| M02-F08 TextGrid参数层 | 2.2 / `parameter_display_widget.py` | 将文本层分配到图窗；标签时间与原TextGrid对应，中文/IPA可读，空标记不乱画 | verified · X12原SQLite text_区域/text_word标签真实显示 |
| M02-F09 图像与数值保存 | 2.2 / `parameter_display_widget.py` | 保存当前图默认PNG，SVG显式选项；当前图与整幅PNG分别回读，数值原表与M01产物回读对应 | verified · 主流程PNG及X03 SVG；数值源表由M01回读 |
| M02-F10 配置草稿 | 2.2 / `parameter_display_widget.py` | 保存绘图配置，改动后切换页签、关闭保护，恢复图窗配置；读取失败可重新关联恢复 | verified · 主流程/X08关闭草稿恢复 |

## 每个实际控件的清单

下表为初次审阅时逐模板控件索引，行号可能随最终布局移动；控件名与绑定为稳定定位。每种普通控件检验正常路径及关键失败/取消，额外数值组合不等同于新的未测控件。动态v-for控件逐文件/参数/任务实例应用同一判据。共享WaveformViewport、AudioTransport、TaskPanel及参数抽屉按总规则复用，不能据模板存在判通过。

| ID | V3源文件/行 | 控件及绑定 | 实际状态 |
| --- | --- | --- | --- |
| M02-C001 | `ParameterDisplayPage.vue:46` | `<button @click="choose('input')" :disabled="busy">选择音频目录` | verified · 完整主流程，PNG真实回读 |
| M02-C002 | `ParameterDisplayPage.vue:46` | `<button @click="choose('association')" :disabled="busy">选择参数目录` | verified · 尾测 X09 |
| M02-C003 | `ParameterDisplayPage.vue:46` | `<input ref="picker" type="file" accept=".wav,.xlsx,.ptb.sqlite,.ptb.sqlite3" multiple hidden @change="addFiles"/><button @click="picker?.click()">添加文件` | not_applicable · Windows 本机入口不提供此网页上传/下载控件，网页服务另列未准入 |
| M02-C004 | `ParameterDisplayPage.vue:46` | `<input v-model="recursive" type="checkbox" :disabled="busy" @change="refresh"/>包含子文件夹` | verified · 尾测 X09 |
| M02-C005 | `ParameterDisplayPage.vue:46` | `<button @click="refresh" :disabled="busy">刷新文件` | verified · 尾测 X09 |
| M02-C006 | `ParameterDisplayPage.vue:46` | `<button @click="save">保存绘图配置` | verified · 主流程 + X08 |
| M02-C007 | `ParameterDisplayPage.vue:46` | `<button @click="emit('references')">参数说明与来源` | verified · X07 |
| M02-C008 | `ParameterDisplayPage.vue:48` | `<input v-model="filter" placeholder="搜索音频"/>` | verified · X03 + 主流程，SVG/PNG实际保存 |
| M02-C009 | `ParameterDisplayPage.vue:48` | `<button v-for="file in audios.filter(f=>f.name.toLowerCase().includes(filter.toLowerCase()))" :key="file.id" :class="{selected:file.id===selected}" :disabled="busy" @click="loadAudio(file.id)">{{file.name}}` | verified · 完整主流程，PNG真实回读 |
| M02-C010 | `ParameterDisplayPage.vue:49` | `<select :value="tableId" @change="loadTable(($event.target as HTMLSelectElement).value)" :disabled="busy"><option value="">未关联参数表` | verified · 尾测 X09 |
| M02-C011 | `ParameterDisplayPage.vue:51` | `<input type="number" min="0" :max="wave.asset.duration-wave.asset.duration/wave.zoom" step=".001" :value="wave.offset" @change="wave.offset=Math.max(0,Math.min(wave.asset!.duration-wave.asset!.duration/wave.zoom,Number(($event.tar` | verified · 尾测 X09 |
| M02-C012 | `ParameterDisplayPage.vue:51` | `<input type="number" min=".01" :max="wave.asset.duration" step=".001" :value="wave.asset.duration/wave.zoom" @change="wave.zoom=wave.asset!.duration/Math.max(.01,Math.min(wave.asset!.duration,Number(($event.target as HTMLInputElem` | verified · 尾测 X09 |
| M02-C013 | `ParameterDisplayPage.vue:52` | `<button @click="maximized=maximized===group.id?null:group.id">{{maximized===group.id?'还原图窗':'放大图窗'}}` | verified · X03 + 主流程，SVG/PNG实际保存 |
| M02-C014 | `ParameterDisplayPage.vue:52` | `<button v-if="groups.length>1" @click="remove(group.id)">合并回首图` | verified · X03 + 主流程，SVG/PNG实际保存 |
| M02-C015 | `ParameterDisplayPage.vue:52` | `<button @click="deletePlot(group.id)">删除图窗` | verified · 尾测 X09 |
| M02-C016 | `ParameterDisplayPage.vue:54` | `<input v-model="search" placeholder="搜索参数" aria-label="参数搜索"/><div class="m02-toolbar">` | verified · X03 + 主流程，SVG/PNG实际保存 |
| M02-C017 | `ParameterDisplayPage.vue:54` | `<input v-model="reaper" type="checkbox"/>reaper` | verified · X03 + 主流程，SVG/PNG实际保存 |
| M02-C018 | `ParameterDisplayPage.vue:54` | `<input v-model="correction" type="checkbox"/>correction` | verified · X03 + 主流程，SVG/PNG实际保存 |
| M02-C019 | `ParameterDisplayPage.vue:54` | `<button @click="markVisible" :disabled="!visible.length">全选可见参数` | verified · X03 + 主流程，SVG/PNG实际保存 |
| M02-C020 | `ParameterDisplayPage.vue:54` | `<button @click="marked=[]">清空勾选` | verified · X03 + 主流程，SVG/PNG实际保存 |
| M02-C021 | `ParameterDisplayPage.vue:54` | `<input v-model="marked" :value="name" type="checkbox"/>{{name}}` | verified · X03 + 主流程，SVG/PNG实际保存 |
| M02-C022 | `ParameterDisplayPage.vue:54` | `<button @click="search='';reaper=true;correction=true">清除筛选` | verified · X03 + 主流程，SVG/PNG实际保存 |
| M02-C023 | `ParameterDisplayPage.vue:55` | `<select v-model.number="target" aria-label="目标图窗"><option v-for="group in groups" :key="group.id" :value="group.id">{{group.title}}` | verified · 尾测 X09 |
| M02-C024 | `ParameterDisplayPage.vue:55` | `<button @click="addGroup">新建图窗` | verified · 完整主流程，PNG真实回读 |
| M02-C025 | `ParameterDisplayPage.vue:55` | `<button @click="groups=clearGroup(groups,target)" :disabled="!groups.some(g=>g.id===target&&g.parameters.length)">清空选定图窗` | verified · 完整主流程，PNG真实回读 |
| M02-C026 | `ParameterDisplayPage.vue:55` | `<button @click="deletePlot(target)" :disabled="!groups.some(g=>g.id===target)">删除选定图窗` | verified · 完整主流程，PNG真实回读 |
| M02-C027 | `ParameterDisplayPage.vue:55` | `<button class="primary" @click="assign" :disabled="!marked.length&#124;&#124;!groups.some(g=>g.id===target)">将 {{marked.length}} 项分配到图窗` | verified · 完整主流程，PNG真实回读 |
| M02-C028 | `ParameterDisplayPage.vue:55` | `<button @click="groups=groups.map(g=>({...g,parameters:g.parameters.filter(p=>!marked.includes(p))}))" :disabled="!marked.length">从图中移除勾选项` | verified · X03 + 主流程，SVG/PNG实际保存 |
| M02-C029 | `ParameterFigure.vue:56` | `<select v-model="format" :aria-label="group.title+'图片格式'" class="image-format"><option value="png">PNG` | verified · X03 + 主流程，SVG/PNG实际保存 |
| M02-C030 | `ParameterFigure.vue:56` | `<button @click="save" :disabled="!group.parameters.length&#124;&#124;exporting" :title="'仅保存参数图 '+format.toUpperCase()">保存当前图` | verified · 完整主流程，PNG真实回读 |
| M02-C031 | `ParameterFigure.vue:56` | `<button @click="saveWhole" :disabled="!group.parameters.length&#124;&#124;exporting" title="白底 300 dpi，包含波形、已开启语谱图及此参数图">{{exporting?'正在生成 PNG…':'保存整幅 PNG'}}` | verified · 完整主流程，PNG真实回读 |

## 公共、状态及性能验收

| ID | 可复现操作与判据 | 状态 |
| --- | --- | --- |
| M02-G01 | 空态与真实数据：1920×1000、1366×768、1280×720，100/125/150%缩放、浅深色、侧栏拖宽，截图及scrollHeight/clientHeight记录；普通滚轮不带动整页，小窗保存/取消可达 | verified · 四档CSS视口/浅深色/右栏折叠，真实Qt DPR1.5；其他OS/实体屏blocked |
| M02-G02 | 原始录音加载冷/热计时；按钮反馈20次、拖动缩放20次，记录median/p95/max与最后状态；目标见P17 G-P01–03 | verified · report.timings实际冷加载/20次更新；未承诺固定延迟 |
| M02-G03 | 全长、2/8/32/64/128倍至采样级连续波形、自适应振幅、零/正/负刻度，试听不变；科学显示抽点不进入导出 | verified · shared-waveform 1790855238430 + 本页实际图；科学数据未改 |
| M02-G04 | 全长/选区试听、暂停/停止、重复点击、音量、切声道、换文件/页签；只有一个播放归属 | verified · 尾测X10真实播放/暂停/继续/停止；X13公共音量/空格/手填/全部/进度；物理声卡听感未验 |
| M02-G05 | 缺输入、非法输入、错误后恢复、取消、迟到、关闭未保存，逐项记录受控注入与真实失败 | partial · 缺输入/非法值/取消/重试/草稿按C/F行证据，额外故障不扩为通过 |
| M02-G06 | 原文件前后SHA-256一致；输出只写output/validation/p17/M02，不得写原目录 | verified · originals-unchanged.json true；输出在本轮独占M01-M05目录，Mxx/evidence-index.json定位 |

V2 实际函数索引（用于追溯）：`__init__`, `init_ui`, `_load_last_dirs`, `_browse_wav`, `_browse_xlsx`, `_on_wav_dir_edited`, `_set_wav_dir`, `_refresh`, `_filter_files`, `_update_param_visibility`, `_on_file_selected`, `_load_param_columns_meta`, `_ensure_time_index`, `_load_window_parameter_df`, `_load_window_parameter_df_full_scan`, `_plot`, `_update_wave_plot`, `_convert_to_float_mono`, `_on_scroll`, `_on_press`, `_on_release`, `_on_motion`, `_play_audio`, `_on_player_media_status_changed`, `_on_player_position_changed`, `_save_image`, `_get_total_duration`, `_configure_view_for_current_audio`, `_normalize_view_window`, `_update_position_slider`, `_update_window_label`, `_set_view_window`, `_on_position_slider_changed`, `set_theme`, `_show_param_help`, `_open_help_doc`

## 本轮证据定位与现行布局

- 完整科学/主流程：`output/validation/p17/M01-M05-5cb4879d0220451da22cf911ce0043e1/report.json`，实际原生产物回读为同目录`readback.json`。
- 普通交互X01–X08：`output/validation/p17/M01-M05-f3d0eb386cd94cb3bc05ad939264ab78/report.json`，exit 0，页面错误0。
- 尾测X09：`output/validation/p17/M01-M05-be70b5cb449347c4a6ca8cbe695bdb7e/report.json` 中已完成checks；X09–X11及X08均完成，exit 0、页面错误0。
- 最后普通控件X12：`output/validation/p17/M01-M05-58f99d3ee30746759b579ca0bcd2d093/report.json`，exit 0、错误0；M05空历史刷新：`output/validation/p17/M01-M05-3e2a15f6e9d3487d8217e3b5516ce1f7/qt-report.json` success=true。
- 最终构建真实隐藏Qt：`output/validation/p17/M01-M05-86b574624cd247ee8f349bfa4610a162/qt-report.json`，success=true。
- 公共播放器/参数X13：`output/validation/p17/M01-M05-4595abcd041e427cbf7d3b143a245dbc/report.json`，80项逐一勾选、音量/空格/范围/进度，exit 0。
- 共享波形：`shared-waveform/1790855238430/report.json`，真实50000点、4组、24次Ctrl滚轮；共享语谱：`shared-spectrogram/1790853257028/report.json`；共享三栏：`shared-workbench/1790854845430/report.json`，均相对`output/validation/p17`。
- M03为明确的左右两栏例外，其他四页三栏。1920×1000、2560×1360、3840×2080为模拟CSS视口；1280×720保持滚动。右侧长历史/高级操作允许内部滚动。Qt DPR1.5，实体屏报告1707×1067，不冒充1920实体屏。
