# M01 参数估计：源码与行为对照

2026-09-09；M01 状态 **planned**。本页是迁移前审阅证据，不是 v3 功能验收。实施顺序见 [文件级计划](../../plans/2026-09-09-m01-implementation.md)，逐项操作见 [验收表](M01-acceptance.csv)，完整参数、设置、源文件 SHA-256 和函数行号见 [机器清单](M01-parameter-settings.json)。

## 核对方法与证据等级

- 对当前工作区继承源码做 AST/文本检查，不导入旧 GUI、SettingsService 或算法，不运行旧程序。
- 30 个相关源码/资源文件逐字节与相邻 v2 比较，全部一致；包括 REAPER 二进制和 IRAPT 查表。只记录仓库相对路径，不把本机私人路径或语料写入本页。
- 80 个参数键及显示名、顺序与 `docs/modules/all-80-acoustic-parameters.json` 精确相同；14 个控件的默认值来自 AcousticConfig，范围来自 SettingsDialog，3 个非对话框字段单列。
- P03 已保存的黄金文件提供旧服务数值、列名和后端证据。本轮没有重新捕获数值，也没有把原服务测试解释为 v3 算法通过。
- 下列“源码事实”均可定位到机器清单的函数行号；“拟议行为”必须在实现后通过验收表，才可标 verified。

## 六组功能的实际入口

| 功能 | 旧控件 → 真实调用 | 源码事实 | v3 拟议入口与验证 |
| --- | --- | --- | --- |
| M01-F01 目录/文件 | `edit_input/edit_output/chk_same_dir` → `_browse_input/_toggle_output_dir/_refresh_files` | 默认同目录；刷新仅枚举顶层 `*.wav`，不递归，清空 TextGrid/唇形关联；列表多选 | `DirectoryOrAssetPicker`＋文件栏；桌面授权目录句柄，网页项目资源集合；列表顺序随任务冻结，不能悄悄递归 |
| M01-F02 TextGrid | `btn_read_tg/btn_tg_seg` → `_read_textgrid/_toggle_segmentation`；服务 `analyze_batch` | 显式读取供缓存；层级按钮循环各层后回到无层。批分析独立自动寻找同名 `.TextGrid`，并不依赖显式读取或当前切分层 | 关联工具栏显示自动匹配来源；独立层级选择。批分析保留所有层标签；选层仅控制预览/切分 |
| M01-F02 切分导出 | `btn_save_seg` → `_save_segmented_audio` | 仅对列表选中文件的缓存层导出，跳过空白、sil/eps 及尖括号变体；采样点用 `int(t*fs)`，写入原 dtype/声道；文件名拼接并过滤字符 | 独立“保存切分音频”，明确选中范围，预估片段数；重名、负时间、越界规则单列修正，不用标签直接构造磁盘路径 |
| M01-F02 唇形 | `btn_read_lip` → `_read_lip_data` → `services/io/lip.read_lip_data` | 同名 `.pkl` 只是先建立关联；计算时读取，另可能读取 `_timestamps.pkl`；偏移在时间轴解析后加一次 | 保留四项功能；网页只接受有版本的安全结构化数据，历史 PKL 经受控本地转换。不能将服务器 `pickle.load` 搬入 v3 |
| M01-F03 参数选择 | `ParameterSelectionDialog` → `_open_parameter_selection` → `selected_parameter_keys` → `_apply_selected_parameters` | GUI 默认显式选择80键；全不选后确认会拒绝，取消保留原选择；过滤发生在全部计算完成后 | 复用 ParameterDrawer；先保留计算路径与导出筛选语义，性能优化另提交；服务 `None` 与 GUI 显式80键需分别回归 |
| M01-F04 设置 | `SettingsDialog.save_settings` → `SettingsService.set/save/get_config_object` | 10＋4控件；SettingsService 是进程单例，`save/load` 均为空操作，所谓“保存”不跨启动 | 配置改为模块草稿＋不可变任务快照；按钮解释应用范围，不引入跨账号共享或自动迁移旧配置；跨启动偏好另作明确设计 |
| M01-F05 浏览/试听 | `_on_selection_changed/_plot_waveform/_update_plot/_play_audio` | 首个选中文件显示/试听，播放从0开始播放全文件；图用降采样、滚轮缩放、拖动平移 | 共用 WaveformViewport/AudioTransport；全文件播放入口保留，新增选区播放独立标注；显示采样时间按真实索引，不能用旧图插值位置代表计算时间 |
| M01-F06 批处理 | `_start_processing` → `PEWorker.run` → `analyze_batch` → `save_results` | 处理**全列表**，不是多选项；每文件完成后导出，失败继续下一项；中断只在文件开始前检查；完成项不撤回 | “处理列表中的 N 个文件”；逐文件可恢复任务＋批次汇总；已完成文件保留；取消当前/未开始文件，显示准确数量和错误，不能整批假成功 |

## 计算与适配拆分

下表路径前缀：旧代码均在 `phonetic_toolbox/`；新核心位于 `packages/phonetic_core/src/phonetic_core/`。精确源码行号/哈希见机器清单，以下新路径均为 **planned**。

| 旧文件/函数 | 拟迁入位置 | 迁移约束 |
| --- | --- | --- |
| `models/config.py`: AcousticConfig、AnalysisResult；`models/acoustic_models.py`: PitchTrack | `models/acoustic.py` | 配置与结果分开；任务配置不可变；采样率元数据修正单列；DataFrame 导出适配不能把数据库引入核心 |
| `services/acoustic_service.py`: PARAMETER_MAPPING、CORE_RESULT_FIELD_MAP | `acoustic/catalog.py` | 80键、标签及旧过滤规则保留；额外 SOE 两键单列，不能篡改现有80项基线 |
| `align_track_to_grid/smooth_preserving_gaps` | `acoustic/alignment.py` | 不外推，不跨声学 NaN 段平滑；重复时间取首项；调用前数组不得被修改 |
| `core/acoustic/common.py,energy.py,voicing.py,spectral_batch.py,corrections.py,cpp.py,hnr.py,shr.py,spectral_slope.py,soe.py` | `acoustic/` 同名文件 | 保留数值步骤、运算次序、窗与归一化；只调整导入和配置/输入接口 |
| `f0_irapt.py`、`Sinc_hash_1000.mat` | `acoustic/f0_irapt.py`、`acoustic/data/Sinc_hash_1000.mat` | 用包资源加载或显式表参数，去掉开发机路径；注册查表 hash 和 GPL 来源；wheel 必须含资源 |
| `jitter_shimmer.py`: `_extract_f0_for_wm/compute_jitter_shimmer` | `acoustic/jitter_shimmer.py` | WM 方法不替换为 Praat jitter；IRAPT→Praat 回退路径保留并新增实际后端记录；不凭结果形状推断后端 |
| `f0_praat.py/formants_praat.py` | `acoustic/f0_praat.py/formants_praat.py` | 将 `Sound(path)` 的解码移到外层并做等价验证；不能假定 SciPy 均值单声道与 Praat 原读文件路径必然相同；保留 cc、真实 pitch times、Burg 参数和槽位筛选 |
| `f0_reaper.py` 的重采样/量化与 EST 解析 | `acoustic/reaper_codec.py` | 16k转换、量化、列解析独立回归；不把原生进程管理放入 core |
| `f0_reaper.py` 的进程/临时目录；`reaper_python.py` 的独立算法 | `ptb_worker/native/reaper.py`；核心 `acoustic/reaper_python.py` | native 经 port；Python回退只迁 EpochTrackerPy 及真实依赖，CLI/绘图/任意路径搜索不入核心；两种后端分别命名和验收 |
| `services/acoustic_service.py`: analyze_file 计算段 | `services/acoustic.py` | 显式数组、解码描述、配置、已解析关联数据、backend ports；不得动态 import 旧包 |
| `services/io/textgrid.py`、声学服务标签填充 | `io/textgrid.py`、`acoustic/annotations.py` | 文本解码与纯解析分开；`text_<tier>`、ceil边界、同名层冲突先捕获；切分层不筛掉分析标签列 |
| `services/io/lip.py`: resolve_lip_time_axis/interpolation | `acoustic/lip.py`；外层 `ptb_worker/io/lip.py` | 核心收安全字典/数组与显式音频锚点；本地适配读取兼容数据；保留偏移、稳定排序、去重、缺失点插值和独立平滑的真实行为 |
| `analyze_batch/save_results/services/io/excel.py` | `ptb_worker/acoustic_executor.py`、`ptb_worker/io/parameter_exports.py` | 单文件 XLSX＋新建 `.ptb.sqlite` 组成一次原子结果；不能覆盖用户现存表/库；新旧导出回读比较数值、mask、列顺序、标签 |

`core/acoustic/lpc.py` 属于 M04 共用候选，本轮只索引，不为 M01 顺带迁移完整 LPC 页面。`__init__.py` 只导出实际迁入函数，不能整包引入不需要的可执行程序。

## 参数、设置和旧行为差异

完整80项见机器清单。P03 不含唇形的默认直接服务调用有 `Time_s`＋78参数＝79列：覆盖目录76键，另有 `SOE_pF0/SOE_rF0`。GUI 将80个目录键显式写入配置，`_apply_selected_parameters` 会筛掉两个 SOE；对同一79列 fixture **静态推导**应剩77列，尚需独立 GUI 配置捕获验证。不能把79列服务基线直接当作 GUI 默认导出。

| 设置键 | 默认值 | 旧控件范围/单位 | 解释 |
| --- | --- | --- | --- |
| silence_threshold | 0.03 | 0–1，比值 | 与能量/静音处理关联 |
| energy_window_ms | 40 | 1–1000 ms | 能量窗 |
| frameshift_ms | 5 | 0.1–1000 ms | 时间网格，不随绘图缩放改变 |
| windowsize_ms | 40 | 1–1000 ms | WM jitter/shimmer 调用使用 `max(160, windowsize_ms)`，并非所有算法共用40ms窗 |
| smooth_win_size | 10 | 1–100 点 | 声学平滑并保留缺口 |
| lip_smooth_win_size | 0 | 0–100 点 | 唇形独立平滑，0/1不平滑 |
| only_voiced | true | 布尔 | 控件写“ZCR判定”，最终声学 mask 实际取 pF0/rF0 有效有声帧并集；名称修正不改变算法 |
| n_periods | 3 | 1–100 周期 | 谐波估计窗 |
| num_formants | 5 | 3–10 | Burg 实际至少请求5个候选，只输出F1–F4及B1–B4；不得按控件名猜算法 |
| max_formant | 6000 | 0–10000 Hz | 0是旧控件允许值，不代表算法可成功；新校验需另列差异 |
| min_f0 | 60 | 10–1000 Hz | 虽在REAPER分组，实际也传给Praat与WM链 |
| max_f0 | 880 | 50–2000 Hz | 同上；旧UI没有跨字段 min<max 检查 |
| reaper_hilbert | true | 布尔 | REAPER开关 |
| reaper_no_highpass | false | 布尔 | REAPER开关 |

非对话框字段为 `use_reaper=true`、`reaper_bin_path`、`selected_parameter_keys=None`。新契约不接受客户端任意可执行路径；实际二进制由安装资源清单选择。`None` 表示旧服务不筛选；空列表在旧服务也不筛选，但 GUI 拒绝空选择，二者必须在兼容层与公开请求层分别测试。

## 需要独立记录的差异与缺口

| ID | 已确认事实/缺口 | 处理原则与验收门 |
| --- | --- | --- |
| M01-D01 | P03-K01 结果 sampling_rate 恒默认16000；实际输入有22050/44100 | 原样数值基线保留；结果元数据改真实采样率，单独差异测试/提交 |
| M01-D02 | 旧图降采样时间用 linspace 而非采样索引；层切换函数只改状态/文字，未直接触发重新绘制 | 新统一波形按索引绘制，层切换即更新。属于可定位的UI修正，不伪称旧行为已具备 |
| M01-D03 | 旧声学平滑保留NaN段；唇形先去NaN插值，可能桥接原缺失点，再用另一种rolling平滑 | 两条链分别回归；不为统一代码而改唇形数值。若需保留唇形缺口，另作科学行为修订 |
| M01-D04 | 旧单文件中断无检查；取消只能在下个文件前生效 | 独立计算子进程＋受控原生子进程，取消必须回收本任务树；已完成文件继续可下载 |
| M01-D05 | P07只支持PG文件任务、16输入/输出、300秒总期限，单任务全部输出一次提交；SQLite仅元数据探针 | 不修改 ZIP 支持上限来容纳 M01。设计独立逐文件任务＋持久批次汇总；schema 如确需增加须先提供具体DDL审阅 |
| M01-D06 | REAPER当前写路径和临时文件；普通临时目录加轮询不能强制阻止超额写入 | 先做输出强制有界的适配可行性验证；未通过不得对网页宣称native可用。Python回退不冒充native验收 |
| M01-D07 | XLSX后才写SQLite；第二步失败时旧XLSX可能已留下 | v3每文件两产物原子发布，失败清理未发布物；前一成功文件保持可见。新建导出库不等于迁移用户数据库 |
| M01-D08 | 旧切分标签过滤后可能重名；负时间、越界和已存在文件没有完整防护 | 保留有效区间量化，异常区间明确拒绝；生成资源ID/不覆盖命名，不能静默截断或覆盖源音频 |
| M01-D09 | P03缺四项唇形、GUI默认/子集/非默认14设置、回退成功分支、完整原GUI路径 | 先补独立捕获。合成唇形只证明时间/数值规则，不代表真实采集正确或自然语料验证 |
| M01-D10 | v3隔离环境无NumPy/SciPy/Pandas/Parselmouth/openpyxl/xlsxwriter发行元数据；core依赖为空 | 先审计原producer与可复用本机包缓存，建立项目内科学依赖锁/来源登记；不动 phonetic_311 |
| M01-D11 | 原生REAPER失败可能调用Python实现；WM的IRAPT失败可能调用Praat | 每任务记录请求后端、实际后端、失败/回退原因；不修改旧有限值来让后端看起来一致 |

P03-K02 的440Hz正弦 REAPER 中位输出约62.745Hz仍只是旧行为；本轮不修正其科学含义。空/单采样点返回空结果与批导出失败分别验收，不能把空表伪装成功。真实持续元音及唇形配套数据仍缺专门确认；在需要前先完成可独立验证部分。

## 来源与运行边界

复用现有12项来源索引：SRC-PRAAT、SRC-REAPER、SRC-IRAPT、SRC-WMPC、SRC-VOICESAUCE、SRC-OPENSAUCE、REF-CPP、REF-HNR、REF-SHR、REF-ISELI、REF-HAWKS、REF-SOE。本轮只核对本地用途，不新增下载、依赖或第三方移植，现有许可未决状态保持。新包迁入时逐文件补实际使用位置、关系类别和版权；完整发行许可仍另验。

本轮不启动 Codex 内置浏览器，不关闭其任何测试页；后续 UI 验证沿用独立 Chrome/Qt 自有进程。测试只写忽略的 `output/validation/m01/`，不更改相邻 v2、旧语料或原科学环境。
