# M03 EGG 信号分析实施计划

2026-09-13最新范围：井井要求暂停EXE及相关探针，下文F阶段为历史计划。E3六功能组复核、总览单击/提交取消已限定验证，见[功能复核报告](../testing/m03-function-review.md)。后续集中处理方法/代码来源未决项，不自动推进M04。

**目标：** 将旧 EGG 的六组功能完整迁入共同工作台，保留可复核的数值、声道和输出语义。

**架构：** 纯数组科学核心加显式配置，经既有任务/文件协议接桌面与网页。GUI中遗留的分析、CSV和绘图规则先分别捕获，正常算法迁移与科学行为修正分别提交。

**技术：** 项目内Python 3.11、NumPy/SciPy/Parselmouth、Pandas/Matplotlib、Vue/TypeScript、现有Qt宿主与P06/P07服务。A仅审计，B已创建独立科学环境并锁定实际Conda/MKL构建；GUI依赖和整合仍待后续阶段。

2026-09-12最新授权：井井在B阶段后回复“好，继续”。**M03-A/B/C已限定verified，完整M03仍in_progress。** B的独立核心、wheel及环境证据见[核心报告](../testing/m03-core-report.md)，C的任务/文件/数值导出证据见[任务报告](../testing/m03-jobs-report.md)。P04-FONT字体快照已接通，单文件/批次实际三PNG及数值CSV不变已限定验证，见[字体报告](../testing/p04-fonts-report.md)；D已限定完成，E1默认/导出/手势对齐见[联合收口记录](../testing/m03-report.md)，完整E仍in_progress，F尚未实施。无需重复002/005，本轮不执行DDL。

## 1. 输入证据和拟定行为

已读说明书3.1–3.4，逐组函数及9项差异见 [源码映射](../modules/evidence/M03-source-map.md)。该表是实施的必读入口，原六功能组见 [模块计划](modules/M03-egg-analysis.md)。

第一轮保留旧界面默认与公式：单文件和批次均GCI slope、GOI scale。A阶段发现旧EGGWidget.init_ui会覆盖EGGConfig的GOI slope默认，纠正上一轮仅据配置类作出的判断。SQ保留旧不对称指标并写出公式。原始/滤波和单文件/批次分别保留配置，不在迁移时强行统一。Praat真实帧时间、输出区间及逆滤波双WAV作为明确修正，须有前后差异证据；ROI重复滤波的科学统一放在兼容迁移之后，未经单独审阅不实施。

布局以本轮用户要求及[位置与组件复用约束](../design/m03-v2-layout.md)为准，覆盖此前右侧设置面板的概括方案。EGG内部保留v2四图、两行参数和下方总览关系；外壳与页面内所有控件、图表、状态均从第一版遵守v3 U2和共同tokens/组件，不能只统一外壳或留到收尾换样式。全部六功能组、30项最终验收继续保留。

三个实现方向已经比较：直接复用Qt页面不满足双端与核心解耦；全面重写事件算法会失去对照基准；选择拆出既有数组函数并分层适配，先保留差异，再有证据地修正。

## 2. 分阶段文件与退出门

### M03-A：独立行为基准与环境审计

拟新增 `scripts/m03_baseline_worker.py`、`scripts/capture_m03_baseline.py`、`tests/fixtures/m03/manifest.json`、`tests/parity/test_m03_capture_contract.py`、`docs/testing/m03-baseline-report.md`。

1. 只读核对P03 EGG三类现有基准、源码及配置哈希。扩展样例包含合成双声道、交换声道、静音、单声道错误、极短、事件缺失、截断ROI与两种数据类型。自然样例延用原明确授权，先核对文件哈希和声道，不导出原语料到仓库。
2. 基准子进程只在测试入口导入原v2，沿用 `scripts/baseline_worker.py` / `capture_v2_baseline.py` 的固定环境和有界输出，不从未来产品入口引用旧目录。
3. 捕获四种GCI/GOI组合、自动/手动门限、raw/filtered、完整/局部计算、两个padding策略、两类F0值与原/实际时间、单文件/批次CSV、静音mask、三PNG绘图参数及IF数组。图片像素检查与数值对照分开。
4. 同一输入双轮捕获，先检测重复性。来源哈希、依赖版本、实际后端、数组dtype/shape/NaN及取消/错误结果进入manifest。
5. 审计m09-ui已有依赖，拟建独立 `.venv/m03-ui` 并锁版本，避免修改m09/m10和v2运行环境。确认新环境前只审计，不安装。

已执行：`.venv/m09-ui/Scripts/python.exe -X utf8 scripts/capture_m03_baseline.py --freeze-public`，随后 `-m pytest -c tests/pytest.ini tests/parity/test_m03_capture_contract.py -q`。11样例双轮一致，脚本及公开fixture已创建。常规重捕获省略`--freeze-public`，冻结保护拒绝覆盖已有基准；详细结果见[M03-A报告](../testing/m03-baseline-report.md)。

退出：所有路径能定位源函数与独立expected，D01–D09前后边界明确。单/双轮返回差异必须先解释，不能增大容差后进入B。

### M03-B：纯数组核心与科学契约

已新增 `packages/phonetic_core/src/phonetic_core/egg/`、`tests/parity/test_egg_analysis.py`、`packages/phonetic_core/tests/test_egg_config.py`，正常兼容迁移先提交，再增加显式边界，见ADR-035。以下条目保留为本阶段设计依据。

1. 先为公开数组接口建立A01–A17回归，再移入现有函数，拆出文件/Qt/可变全局配置。逐字段保留峰谷阈值、滤波阶数/截止、局部窗口与事件位置。
2. 参数快照至少分为 `signal_mode`、`flip_channels`、`analysis_scope`、`roi_start/roi_end`、`gci_method/goi_method`、`peak_prominence/valley_prominence/auto_prominence`、`highpass_cutoff/lowpass_cutoff`、`spec_window_ms/spec_vmin/spec_vmax`、`lp_order`、`export_policy`。作用不同的配置不能共享一个被GUI修改的实例。
3. 结果分别保存采样时间、GCI/GOI/peak时间、CQ/SQ事件时间、Praat帧时间、GCI-F0中点时间。NaN/mask沿用协议，不把不同网格插值成一套后隐藏原值。
4. 保留当前SQ公式，并用独立解析事件测试：GCI=[0,.01,.02]、GOI=[.006,.016]、peak=[.002,.014]，应有CQ=[.6,.6]、SQ=[1/3,-1/3]。边界0.05/0.95、多个peak和无GOI需有独立缺失预期。
5. 明确raw/filtered局部兼容策略。正常算法移植提交通过后，再单独处理Praat实际时间与N/fs元数据修正，附差异摘要。不得把新科学方法或ROI滤波统一混进移植提交。

已执行：`scripts/Invoke-M03-Python.ps1 -X utf8 -m pytest -c tests/pytest.ini tests/parity/test_egg_analysis.py packages/phonetic_core/tests/test_egg_config.py tests/parity/test_m03_capture_contract.py -q`，最终80项通过。安装wheel后从独立目录执行同组回归和`scripts/verify_m03_core.py --include-private`，11样例31761项精确比较通过。真实使用`.venv/m03-compatible`，原拟`.venv/m03-ui`的PyPI SciPy构建无法逐位复现旧MKL去趋势，保留失败证据而不扩大容差。

### M03-C：有界任务、文件和导出

本轮完成状态及实测预算见[任务报告](../testing/m03-jobs-report.md)。采用固定的隔离科学bootstrap，不把EGG加入现有冻结EXE的通用worker入口；现有EXE尚未包含兼容运行库。通用任务表承载逐文件批次项，整批目录交互留给D，不改变005的M01批次限制。单文件/批次CSV科学规则保留并明确sample-aligned/1差异，FFT分块与原调用逐字节一致。字体专项后续已按下一段公共方案接入，报告范围以P04-FONT为准。

新增依赖（2026-09-12）：[P04-FONT](2026-09-12-global-fonts-design.md)规定三PNG及批次图片使用统一导出字体快照。C已有数值/文件工作可独立推进，字体契约接入前不得把图片字体专项标为verified。Matplotlib等后台渲染在适配层解析当前渲染端可用字体，按任务隔离配置；科学核心不读取用户字体或系统字体目录。该补充不授权DDL或更改已冻结的科学数值。

拟新增 `backend/src/ptb_api/egg_models.py`、`backend/src/ptb_worker/{egg_jobs,egg_child,egg_exports}.py`，修改 `job_models.py`、`jobs.py`、`ptb_worker/{store,acoustic_executor,process_entry}.py`、桌面 `task_bridge.py` 与 `frontend/src/platform/{research,desktop}.ts`。是否复用executor内部助手在C开始前检查，禁止复制另一套科学算法。

1. 新 `m03/1` 任务/结果模型由Pydantic生成contracts。仅以受控引用传入文件，读取和导出复用P06/P07 owner/hash/expiry/fencing/原子发布。
2. 子进程白名单涵盖EGG，限制输入样本、时长、内存、运行时间及导出像素。预算依据A阶段测量后写入计划，不先复制M01的240秒上限到60秒总览的EGG任务。长音频预览和科学任务预算分别解释。
3. 单文件三PNG+CSV、批次CSV和可选图、IF两WAV分别形成完整manifest。CSV网格策略及静音mask带标记，不把其中一图失败报告成全部成功。
4. IF输出为当前分析音频归一化片段及简化CP逆滤波结果，保留样本数、采样率、原ROI偏移、阶数/后端/失败信息，避免误标为原始文件字节切片。
5. 新operation如可复用既有jobs表则无DDL；若实际表约束要求DDL，先提交具体迁移审阅，此计划不构成执行授权。
6. C须先审阅兼容科学环境如何接入所属worker；不能把M03的Conda SciPy/DLL直接覆盖m09/m10宿主。独立子进程运行环境须有明确指纹，后续冻结/Qt整合时单独验证DLL搜索与既有模块数值，当前B不代表该整合已通过。

拟测试文件：`backend/tests/test_m03_jobs.py`、`backend/tests/test_m03_exports.py`、`desktop/tests/test_m03_bridge.py`、`tests/contracts/test_m03_contract.py`。退出：真实读写回读、取消/恢复/越权、失败全回收及IF双WAV一致，不止mock成功。

### M03-D：共同界面与真实操作

本轮已完成限定Windows开发态实现，报告见[页面验收](../testing/m03-ui-report.md)。井井针对按钮区的反馈已落实为两行功能分组，高低通集中EGG图上方。preview数据与IF四图数值边界见ADR-038，宏观任务范围不扩展为完整E/F。

字体退出门：先接入P04-FONT公共角色与偏好，再构建页面及四图，覆盖CQ/SQ双轴、谱图色条、微观事件标签、总览、参数行、批次与IF结果。不得先硬编码字体再等待收尾替换。除A28外追加[P04-FONT的F01–F12](2026-09-12-global-fonts-design.md)适用项，三PNG实际回读与字体缺失回退列入E联合验收。公共字体能力未就绪时，页面字体专项保持planned/in_progress。

拟新增 `frontend/src/modules/egg-analysis/{EggAnalysisPage,EggSignalPlots,EggParameters,EggBatchPanel,EggInverseResult}.vue`、`state.ts`，修改 `AppShell.vue` 和共用平台声明。新建 `frontend/tests/m03.test.ts`、`scripts/verify_m03_qt.py`、`tests/e2e/m03.cjs`。

页面实施前按[实际复用清单](../design/m03-v2-layout.md)接AppShell、tokens、AppIcon、AudioTransport、TaskPanel、ModalDialog及MethodReferences；WaveformViewport/公共状态、SpectrogramViewport按EGG科学语义作有界适配，WorkbenchColumns现有三栏不能机械套成四图。新增共享扩展需补现有M01/M02定向回归。第一版即提供真实数据的浅深主题并与现有v3页面比对，不推迟统一风格。

1. 顶部文件/声道/批处理/帮助工具。左上CQ/SQ、左下带顶部色条的语谱图；右上音频微观、右下EGG微观，原始/滤波和高通留在EGG图上方。图下两行参数，最后为60秒总览和长文件导航。播放复用共同能力，入口仍在第一行参数。保留单文件、批次、IF独立结果范围。
2. 新页面只消费结果，不计算另一套事件/F0。原始/滤波开关的旧科学影响明确可见，重算期间旧结果标为过期，阻止错配保存。
3. 微观事件与CQ图如沿用不同legacy策略必须显示来源，不能制造完全同步的假象。全局时间与毫秒相对时间分别标记。
4. 先测默认操作、每个设置/导出/关闭，再测浅深主题、390/1000/1440宽、IPA/长文件名、取消/错误和输入切换。独立Chrome/Qt，不调用Codex内置浏览器关闭。

拟执行：`npm --prefix frontend test`、`npm --prefix frontend run typecheck`、`npm --prefix frontend run build`、`.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m03_qt.py` 及项目拥有的Chrome验证启动器。

### M03-E：来源、说明书与逐项联合验收

拟新增 `docs/manual/egg-analysis.md`、`docs/testing/m03-report.md`，更新来源注册表、模块矩阵和任务账本。

1. 核对PENDING-EGG及手册引用原文的准确书目/页码、实际公式关系和代码许可。未决来源如实保留，不承诺完全原创或全部可公开再分发。
2. 联合单文件/批次/IF三路径，逐项解释默认、科学及导出差异；完整A01–A30须有文件/命令/平台/结果位置。
3. 后续科学行为修正若影响已导出文件，在版本/说明书中记录，不能用“优化”笼统覆盖。

### M03-F：Windows候选与停止点

在E通过且单文件打包范围获审阅后，使用独立候选文件名生成Windows单文件，不覆盖Research-Fix1/M10-R5。真实EXE复验冻结worker、全部三路径、重开恢复及自有进程清理。只有达到此范围才标M03 Windows verified，网页与跨平台按各自证据记录。

完成M03后停止，M04及其他模块不自动接续。源码/开发态通过不能替代完整EXE、声卡或跨平台验证。

## 3. 三十项验收设计清单（当前逐项状态见M03-report）

| ID | 正常路径 | 错误或边界 |
| --- | --- | --- |
| A01 | PCM16/FLOAT双声道按角色读取 | 单声道、多声道、损坏、空输入拒绝 |
| A02 | 交换后EGG/音频重算、试听一致 | 连续切文件/交换旧结果失效 |
| A03 | 各声道独立峰值0.7与dtype对应 | 零声道、非有限、负峰与原文件不变 |
| A04 | raw/filtered显示与分析来源 | 滤波失败不假称已滤波 |
| A05 | 25/1000Hz、非默认高低通 | Nyquist、极短、参数越界 |
| A06 | ±50/100ms局部padding分别对照 | 起止贴边、旧重复滤波差异 |
| A07 | 20/5/50ms谱窗及显示上下限 | 短于NFFT、上下限相反 |
| A08 | GCI/GOI四种组合 | 无峰/无谷、边界事件 |
| A09 | 自动200/100ms、手动峰谷 | 下限、重叠去重、非法阈值 |
| A10 | scale实际0.25与线性插值 | 不接受表面可调但无效参数 |
| A11 | 解析CQ/SQ及周期时间 | 0.05/0.95、重复GCI、多个peak、NaN |
| A12 | 当前显示事件/CQ各自来源 | ROI过期、padding外行不混淆 |
| A13 | F0变化启发式标记 | 无F0、同类0.1秒合并边界 |
| A14 | Praat原值与实际帧时间 | 旧0.005时间差、无声、失败 |
| A15 | GCI间隔倒数与中点 | MAD/std、低于100Hz保留、缺失 |
| A16 | 稳定元音简化CP逆滤波 | 缺GCI、阶数过大、少于3段、奇异矩阵 |
| A17 | IF两WAV及前后波形/谱对比 | 取消保存、冲突、任一写出失败 |
| A18 | 总览60秒、主ROI与点击联动 | 超60秒、EOF、非零起点 |
| A19 | 50ms微观与5–5000ms交互（E3恢复原滚轮范围） | 最小/最大、拖出边界、空区间 |
| A20 | ROI播放、停止及交换后音频 | 切模块/文件/取消不残留播放 |
| A21 | 单文件outer join CSV+3PNG | 时间网格、NaN、文件名、部分写出失败 |
| A22 | 批次GCI网格与两类F0可选 | 关闭某F0不删另一列 |
| A23 | 20ms绝对值包络/静音mask | 阈值相等、开关变化、图CSV策略差异 |
| A24 | 批次方法/高通/交换/可选图 | 单文件当前ROI不污染整批 |
| A25 | 处理中进度/取消/失败继续 | 文件内取消、旧worker迟到 |
| A26 | 结果保存及重开恢复/重试 | 配额满、到期、服务中断、不重复产物 |
| A27 | 桌面不登录、网页owner | 换账号/越权/不同参数快照隔离 |
| A28 | 浅深/窄窗/IPA/键盘 | 空态、长文件名、焦点与未保存状态 |
| A29 | 方法/公式/版本/来源显示一致 | 未核准文献不作准确性保证 |
| A30 | 实际桌面与网页三路径全回读 | 冻结worker、未知入口、退出清理独立验收 |

## 4. 当前执行结果

M03-A/B/C/D 已分别在原版基准、Windows安装核心、开发态任务/导出、开发态统一页面范围内验证。D证据见页面报告，包括井井要求的按钮分组修订；公共字体已从第一版接入。完整M03/P08仍in_progress，下一项E：30项逐项对照、自然语料页面、网页账号联合路径、输出命名及来源收口；F冻结EXE仍planned。D广域回归的一次M01 Scratch清理占用单列保留，不宣称所有回归稳定全绿。

2026-09-12 E1追加：井井在D后授权继续，已按ADR-039修正单文件F0显示与CSV列分离、独立批次默认/草稿、源名称/时间保存和四图手势。E1限定验证，完整E仍in_progress；下一批E2网页/自然录音/切换竞争，长文件与微观范围差异仍保留。

2026-09-12 E2追加：网页双账号/三路径/受控配额与到期、自然录音开头和较响ROI及迟到旧任务竞争已限定验收，见[报告](../testing/m03-e2-report.md)。完整E仍in_progress，下一项E3范围/来源及未覆盖边界。F需候选范围审阅，未授权自动扩展M04。

2026-09-12 E3本轮执行范围：按 ADR-040 恢复微观 5–5000 ms 及原版显示抽点，追加原 V2 独立基准、API/页面范围回归、实际宽窗口和长文件导航/明确拒绝测试。核实手册推荐论文的官方书目并区分代码许可与论文关系。长文件全段分析、未能获得的论文原文/页码、原生多屏设备和冻结 EXE 不以本轮局部通过代替。


2026-09-12 E3-B追加：120秒/576万帧有界长文件、原V2独立双轮对照、受限进程/Chrome/Qt末尾与全段导出已限定验证，见[长文件报告](../testing/m03-long-report.md)。上文60秒计算限制与长文件拒绝为历史证据；总览60秒视窗保持。完整E3/M03仍in_progress，下一项字体预检及剩余交互/来源，冻结EXE另行收口。
