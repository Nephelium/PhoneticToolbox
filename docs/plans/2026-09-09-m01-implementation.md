# P08 / M01 参数估计 Implementation Plan

> 执行者：秋叶。使用本机 `executing-plans` 技能按任务推进；不自动创建其他任务或子代理。已有 `codex/v3-rebuild` 工作树继续使用。

**Goal:** 将 v2 参数估计的6组功能迁入统一工作台，使桌面离线与网页任务共用同一科学核心，并逐项保留可验证的科研行为。

**Architecture:** 核心只接数组、配置、关联数据和声明式后端接口；解码、原生进程、导出、配额和任务持久化在外层。一个文件的 XLSX 与 `.ptb.sqlite` 原子发布，批次由逐文件任务汇总，保留已完成文件。页面复用现有工作台和平台接口，不另起页面服务器。

**Tech Stack:** 已有 Python 3.11.14、Vue 3/TypeScript/Vite、FastAPI/Pydantic、PostgreSQL/SQLite、PyQt6；M01-B 科学环境已按 P03/M01-A 实测版本独立锁定，具体环境和验收边界见下。

---

## 0. 本轮范围与状态

2026-09-09，井井在“下一步先形成 M01 实施计划与功能对照表”的说明后回复“好，继续”。计划最初交付完成**迁移前源码审阅、文件级计划和验收设计**，当时M01/P08为planned。后续M01-A已实施（见下段），但算法/UI仍未迁移。

本轮交付：[源码对照](../modules/evidence/M01-source-map.md)、[80参数/14设置及30文件哈希](../modules/evidence/M01-parameter-settings.json)、[细项验收表](../modules/evidence/M01-acceptance.csv)。既有 [M01模块计划](modules/M01-parameter-estimation.md) 保留为功能规格，本计划补足文件和执行顺序。

M01-A 已按井井后续“好，请继续”实施并通过限定基准验收，见 [M01-A报告](../testing/m01-baseline-report.md)。M01/P08 当前 in_progress，下一项为 M01-B 科学核心与数值对照。原规划的首个子任务为 **M01-A：补齐旧行为基准与科学环境审计**。它不改算法/产品页面，不写现存研究数据，不建新数据库表。M01-C/F 涉及原生输出与持久批次的具体设计门；只在可审阅方案具备后进入对应实现。任何实际数据库DDL、用户数据迁移、全局环境变更、push或发布仍按根规则单独授权，不能由此计划推定。

初次规划交付未引入依赖或算法源码。井井在 M01-A 后回复“继续”，授权 M01-B；本轮迁入纯计算与数组/配置接口，并更新实际来源登记，不以数值等价替代发行许可审查。

### M01-B 实际执行环境与子步骤

- `requirements-m01-science.in/.lock`：8 个科学运行包，全部精确匹配 P03/M01-A；`requirements-m01-test.in/.lock` 再锁定 10 个测试/构建包。只从 PyPI 安装有哈希的 wheel，不改 `phonetic_311`、现有 `.venv/v3-dev` 或全局依赖。
- 开发解释器：项目内 `.venv/m01-science/Scripts/python.exe`。独立 wheel 验收：`.venv/m01-wheelcheck/Scripts/python.exe`，从测试输出目录以 `-I` 运行，确认实际导入来自其 site-packages。
- 先原样保留科学运算和元数据旧默认值形成迁移提交，再单独修正实际采样率元数据；不覆盖 P03/M01-A 黄金文件。报告见 [M01-B 核心验收](../testing/m01-core-report.md)。
- 原 M01-C 的纯 REAPER PCM/EST 编码和已解析唇形/标签插值提前到 B，以便数组服务完成数值对照；文件解码入口、PKL 安全兼容、原生进程预算、导出与持久化仍在 C/F。
- B 的 REAPER 测试适配仅对 <=64,000 帧自有合成输入运行已核验二进制，目录来自独立测试进程；不得用于网页/用户任务。其成功不证明 C 的强制资源上限或生产取消能力。
- core 的 `acoustic` extra 声明科学依赖，基础工程探针仍可独立导入；完整运行锁固定传递版本。配置不可变和基本非法数值拒绝属于新数组 API，公开 JSON 契约的完整校验仍在 D。

## 1. 不可静默改变的约定

1. 主按钮处理**整个列表**，试听/切分使用选中项；切分层不改变批分析的全部 TextGrid 标签层。
2. 参数80键、14设置、p/r轨迹、校正/未校正、输出顺序、单位、NaN及时间网格分别锁定。GUI显式全选与服务 `selected_parameter_keys=None` 是不同基准。
3. 原始数值迁移、必要元数据/错误修正、UI交互更新分别提交。初次迁移不做计算裁剪、向量化、改变窗长或默认F0方法。
4. 桌面选目录/保存结果不要求网页账号；不套5GB或7天删除；网页输入/临时/结果均受配额和owner检查。
5. 科研文件成功以“产物写入＋回读校验＋manifest发布”为准；不能通过图上有曲线或进度100%认定成功。
6. 正式包禁止导入 `phonetic_toolbox`、相邻v2或旧网页项目。只读独立基线子进程按 ADR-014 可导入v2。
7. P07 已验证的 ZIP 限制、旧worker保护、过期和幂等保持回归；M01 不借扩展大列表放宽ZIP限制。

## 2. 三个实现前需要解决的接入点

### 2.1 逐文件结果与批次

旧 v2 逐文件成功后立即保存；P07 工程文件任务则整任务一次提交，失败清理未发布结果。拟使用 **一个输入WAV对应一个计算任务**，每任务引用自己的WAV、TextGrid和安全唇形数据，发布同一文件的两种导出；另建持久批次协调记录提交顺序、已创建子任务、取消与汇总。

批次创建必须幂等且可恢复，不能在浏览器循环发送N个请求后把本地数组当持久批次。失败汇总含已成功/失败/取消/未开始计数；`complete=false` 是批次语义，不修改P07的 `FileManifest.complete=true`。一项失败继续后续项；批次取消停止新建/认领并取消本批活动项，之前成功产物保留。部分失败重试仅为明确选定失败项创建新attempt。

先审阅是否需新增 batch/batch_item 表及SQLite等价结构，再写具体DDL；本计划不预先批准执行。每批和全账号的任务总量限制应可见且明确，不能静默截断列表。验证至少包含超过16个WAV的批次和中途重启，证明没有误套ZIP条目上限。重新评估P07的300秒总期限：运行预算由服务端受控策略决定，快照记入deadline，永远不超过输入最早到期时间；不能无限续租延长输入寿命。

### 2.2 原生REAPER与导出写入

REAPER 当前先转换16k单声道PCM16并写临时WAV，再让二进制写EST文件。不可将该目录直接交给网页worker，仅靠轮询文件大小无法保证不超额。

M01-C先用项目自有合成输入和**现有确切二进制**验证可强制有界的输出传输：首选实验为命名管道/等价受控句柄输出，由接收端限字节、背压并在取消时终止所属进程；不能假定二进制支持该路径或关闭行为。需要磁盘中间物时，必须在创建前纳入资源/预算，清理与fencing可追踪。若现有二进制不兼容，记录失败、比较有明确来源的适配补丁或受控文件系统方案并单独审阅，不安装系统驱动、不用定时查大小充当强制限额。

XLSX先验证受限可seek内存缓冲/写入器，禁用库的非受控临时文件；SQLite新导出库可实验内存构建、限制页数与临时存储、序列化后分块进入受控writer。内存亦有输入/输出上限与OOM隔离，不以“内存文件”免除生成资源预算。每文件的两份结果都在完整回读后一次发布。旧用户 `.ptb.sqlite` 不参与schema升级或覆盖测试。

此门不通过时仍可独立完成核心parity，但**不得标M01完整通过**，也不得将Python回退结果标为原生REAPER。

### 2.3 唇形兼容和桌面文件能力

安全唇形格式拟为版本化JSON：四个metric数组、原时间信息、音频起点锚点、manual_offset；nonfinite采用 `null + mask`，从旧格式转换不能损失偏移或时间原点。核心只处理解析后的结构。原 `pickle.load` 不进入网页；旧PKL兼容采用专门本地受限转换器，拒绝任意类构造/全局函数执行，先用受控合成旧记录证明四指标与伴随timestamp语义。真实旧格式不受支持时明确列出缺口，不能把无限制pickle当兼容捷径。

P04 的 `FileProvider.load(File)` 仅能加载WAV；`HostCapabilities.jobs=false`，desktop目前复用浏览器provider。M01需真实目录授权句柄、同目录/独立目录结果写入、文件列举和本地任务客户端。句柄绑定本次本地会话和授权根，防止越界/重解析点/用户给任意路径；网页仅传owner校验后的asset ID。完整科研页面还需接入当前 `/server/` 项目页与公共 AppShell，不能只在公开演示工作台画可点的任务按钮。

## 3. 文件级工作顺序

每个任务按“独立失败用例 → 最小实现 → 定向验证 → diff/来源核对 → 明确文件清单本地提交”执行。一次只推进一个可审阅子任务；下面的新文件/命令均为**拟建**，本轮不声称已存在或通过。遇到真实数值差异先定位，不扩大容差、不重写golden。

### M01-A · 基准补齐与环境审计（已完成限定验收）

**文件：** 新建 `scripts/capture_m01_baseline.py`、`scripts/m01_baseline_worker.py`、`tests/fixtures/m01/manifest.json`、`tests/parity/test_m01_capture_contract.py`；复用不覆盖 `scripts/baseline_support.py`、P03 fixtures；报告 `docs/testing/m01-baseline-report.md`。

1. 读取本机P03 producer中的依赖版本、实际加载模块及REAPER/IRAPT hash，比较本轮30文件清单。保存本阶段context-before，原运行环境仅只读。
2. 写失败用例证明P03尚未覆盖GUI80键筛选、只选pF0/rF0/Intensity/Energy兼容、空选择拒绝、14设置边界及四项唇形。期望来源为独立v2子进程或解析公式，不能调v3生成expected。
3. 调度器用v3 Python，旧算法子进程用原解释器 `-B -X utf8 -X faulthandler`；仅该子进程PATH补旧conda DLL目录，cwd/TEMP/TMP指向新测试目录。沿用P03隐私和输入hash前后核验。
4. 捕获同一小样本的服务None/GUI80键/子集配置；合成TextGrid层、解析唇形时轴和各设置单变量对照；真实回退路径通过可控后端失效触发，分别记native/python/IRAPT/Praat。不同配置各两次进程重复，检查实际调用而非只看标签。
5. 补导出回读：列重命名、顺序、NaN、text_标签、SQLite索引、空表失败；独立保存合成黄金结果，禁止覆盖P03现有golden。
6. 生成项目内科学依赖**审计候选**：原producer版本、已存在wheel/来源/许可、v3缺失项和平台限制；此步先报告，不安装或改lock。只有用于数字迁移的具体环境方案确定后进入B。
7. 执行 `& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/capture_m01_baseline.py`（已实现入口默认不覆盖已冻结文件；不操作现存/服务数据库，仅在新测试目录导出合成SQLite结果）；`& '.venv/v3-dev/Scripts/python.exe' -X utf8 -m pytest -c tests/pytest.ini tests/parity/test_m01_capture_contract.py -q`。

**退出条件：** 三条参数选择路径的真实列、四项唇形合成时间行为、14设置请求/实际传递及独立后端证据可定位；专门确认的真实持续元音/唇形缺口仍明确标出，不能伪造自然语料标签。

### M01-B · 科学核心与数值对照

**文件：** `packages/phonetic_core/src/phonetic_core/models/acoustic.py`、`acoustic/catalog.py`、`acoustic/alignment.py`、源码对照表所列同名算法、`services/acoustic.py`、`ports/acoustic.py`；`packages/phonetic_core/pyproject.toml`；拟建项目科学依赖 `.in/.lock`；`third_party/source-registry.json`；测试 `tests/parity/test_parameter_estimation.py`、`packages/phonetic_core/tests/test_acoustic_alignment.py`、`test_acoustic_config.py`。

1. 按A的确切版本在项目隔离环境安装/锁定依赖并登记来源；不改原环境。安装wheel到另一个项目内干净环境验证，不动态复用v2路径。
2. 先写以下行为回归，再迁对齐、平滑、配置和参数目录；断言红因必须是缺少迁入实现，不是导入旧包通过。

```python
# 新建核心测试中的代表性断言；不是本轮已实现函数。
out = align_track_to_grid([0.1, 0.2, 0.3], [1.0, float('nan'), 3.0],
                          [0.0, 0.1, 0.2, 0.3, 0.4])
assert np.isnan(out[[0, 2, 4]]).all()
np.testing.assert_array_equal(out[[1, 3]], [1.0, 3.0])
assert config.frameshift_ms == 5.0
assert config.windowsize_ms == 40.0
assert len(parameter_catalog) == 80
```

3. 将纯算法逐文件迁入，保留表达式顺序/注释/版权；对文件I/O和绝对旧导入做最小拆分，不同时优化公式。新API概念为 `analyze_audio(audio, config, associations, backends, cancellation) -> AcousticResult`，实际类型与错误在C/D完成契约后冻结；不把Path藏进array参数。
4. 对数值未变路径比较P03＋A基线；字段、shape、时间轴、mask精确；有限值沿用P03 `max(atol,rtol*scale)` 容差（1e-10/1e-7），不得直接换成更宽的allclose相加容差。元数据sampling_rate单独修正并列出预期差异。
5. 覆盖空/极短/损坏音频、声道转换、22050/44100重采样、非默认帧移、平滑缺口、输入数组不变与连续两配置不互相污染。重采样/原生输入量化另比较字节。
6. 运行 `& '.venv/v3-dev/Scripts/python.exe' -X utf8 -m pytest -c tests/pytest.ini packages/phonetic_core/tests tests/parity/test_parameter_estimation.py -q`；再运行 `scripts/check_architecture.py`。若科学环境采用新独立解释器，应先在本计划记录确切路径，不能改用旧环境跑v3验收。

**退出条件：** 核心可独立安装、不含Qt/HTTP/DB/旧包导入；数值差异全部解释；实际native路径待C后才能联合验收。

### M01-C · 原生、格式与文件预算适配

**文件：** `backend/src/ptb_worker/native/reaper.py`、`io/audio.py`、`io/parameter_exports.py`、`io/lip.py`；核心 `acoustic/reaper_codec.py`、`acoustic/reaper_python.py`、`acoustic/lip.py`、`acoustic/annotations.py`、`io/textgrid.py`；`scripts/verify_m01_native_io.py`、`backend/tests/test_m01_io.py`、`tests/security/test_m01_formats.py`；新建 `resources/manifests/acoustic.json`（当前resources仅有规则和架构文件）。

1. 先写取消/超限/错误回收用例，证明直接目录路径不被接收；实现2.2的独立有界输出探针，并实测原生二进制成功/失效/取消/异常退出。
2. 解码WAV保留原采样率/声道/量化，分别对照Praat与SciPy旧路径，防止统一单声道改变原结果。资源安装路径由宿主提供；native仅参数数组启动、自有进程句柄结束，不查杀端口或同名进程。
3. 在解析前限制输入大小/解码样本数/标签层和文本长度，明确返回错误；以超长文本、畸形WAV、受限pickle恶意构造测试，不能用大量分配后再判断大小。
4. 用合成唇形四指标验证音频锚点优先级、伴随时间、手动offset只加一次、排序去重、缺失插值与独立平滑；本地转换器只处理测试或明确授权的文件。
5. 构建XLSX和新SQLite双产物；加入第二文件失败、内存/字节预算不足、中文/IPA/公式样文本、任务取消场景；文本导出禁止把用户标签当公式执行，必要编码修正单列并回读。
6. 执行 `& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/verify_m01_native_io.py` 和 `& '.venv/v3-dev/Scripts/python.exe' -X utf8 -m pytest -c tests/pytest.ini backend/tests/test_m01_io.py tests/security/test_m01_formats.py -q`。

**退出条件：** 无任意目录输出旁路；既有REAPER二进制实际可用且受控；双产物回读等价；不能只凭模拟writer通过。

### M01-D · 契约与科学结果语义

**文件：** `backend/src/ptb_api/acoustic_models.py`、`job_models.py`、`models.py`；`contracts/openapi.json`、`generated/api.ts`、`versions.md`；`tests/contracts/test_m01_contract.py`。

1. 先写请求拒绝任意路径/native_tool/未知参数键/非有限值、同名关联歧义、跨owner资源、非正max_formant、min>=max的用例。旧边界允许但计算失败的配置保留在兼容数值基线；公开校验变化记录D01等差异。
2. 定义参数目录键与显示名分离、14设置快照、输入资源hash和后端策略；实际解码采样率/样本数/声道、algorithm/source IDs、实际后端、nonfinite mask/reason进入结果元数据。不从旧NaN推导不存在的原因。
3. 分离 `AcousticFileManifest`（单文件完整两产物）与 `AcousticBatchSummary`（计数/子任务/complete）；SOE额外输出标legacy service extension，不静默塞入80键列表或丢失。
4. 明确生成分析结果按成功时间起算最多7天；切分音频是原内容派生，截止不超过输入；引用到期时取消相关在途任务，不能沿用P07“除storage_check外都取输入期限”的工程分支直接实现分析TTL。
5. 用 `scripts/generate_contracts.py` 和 `npm --prefix frontend run contracts`生成，再运行两侧 `--check`/`contracts:check` 与定向契约测试；不手改生成类型。

**退出条件：** 旧P06/P07协议可读，三类manifest不混淆；实际行为修正有版本及测试依据。

### M01-E · 桌面目录与共同研究页

**文件：** 新建 `desktop/src/ptb_desktop/file_provider.py`、`host.py`，修改 `desktop/src/ptb_desktop/main.py`（保留既有诊断入口），参考但不覆盖 `desktop/experiments/p04_host.py`；`frontend/src/platform/types.ts/browser.ts/desktop.ts`；`frontend/src/modules/parameter-estimation/ParameterEstimationPage.vue`、`state.ts`；`frontend/src/app/WorkspaceView.vue`、`AppShell.vue`；`frontend/src/account/ServerPage.vue`；复用 `ParameterDrawer.vue/WaveformViewport.vue/AudioTransport.vue/MethodReferences.vue`，按需新增 `SettingsDrawer.vue/DirectoryOrAssetPicker.vue`；`desktop/tests/test_m01_files.py`、`frontend/tests/m01-state.test.ts`。

1. 写目录句柄限定范围、取消选择不改状态、同目录不覆盖WAV、两个宿主会话句柄不通用、符号链接/重解析点逃逸拒绝的测试。
2. 接入真实Qt选择器和本地服务会话；不以 `location.protocol` 冒充文件授权。当前P04静态scheme拦截器会阻止所有HTTP请求，正式host需只放行本次已握手的loopback origin，复用Host/Origin/会话校验；凭据不能进入URL、localStorage或日志。保留静态资源/外链权限隔离，不把测试页拦截器简单关闭。网页通过项目选择进入同一研究页，UI状态按owner/project/module隔离。
3. 先对文件列表与关联状态、80键全选/全不选取消、14设置草稿/应用写前端行为测试，再接共同组件。参数修改仅影响下一任务；切换主题/标签/项目保护未保存草稿。
4. 全列表处理、选中试听、选层切分三个范围分别显示；TextGrid缓存与批分析关联来源一致可查；没有真实结果不画占位科学曲线。
5. 校正文案“仅保留浊音(ZCR)”和min/max F0使用范围；显示默认窗与实际WM最小窗的差异，不改变公式。
6. 执行 `npm --prefix frontend run typecheck`、`npm --prefix frontend test`、`npm --prefix frontend run build` 与 `& '.venv/v3-dev/Scripts/python.exe' -X utf8 -m pytest -c tests/pytest.ini desktop/tests/test_m01_files.py -q`。

**退出条件：** 真目录授权和共同页面操作具备；完整数据任务需F/G后确认，暂不标页面verified。

### M01-F · 持久批次与双端任务

**文件：** `backend/src/ptb_worker/acoustic_executor.py`、`acoustic_batches.py`、`files.py`、`store.py`、`executor.py`、`cli.py`；`backend/src/ptb_api/jobs.py`、`main.py`、`server.py`；`desktop/src/ptb_desktop/local_service.py`；拟审阅 `backend/migrations/005_acoustic_batches.sql` 及本地schema方案；`scripts/run_m01_validation.py`、`backend/tests/test_m01_jobs.py`。

1. 提供batch创建、文件子任务关联、恢复与取消的精确DDL、旧行保存查询和清理范围；实际迁移待该具体操作授权，不能复用003/004授权。
2. 先验证幂等部分提交、17文件批次、第二文件失败、未开始即取消、计算中worker退出、租约过期旧worker、结果提交/输入删除竞态；使用真实PG/磁盘，不以mock成功替代。
3. 桌面SQLite元数据任务拓展文件句柄和本地结果发布流程，复用科学worker和导出器；不可 import server端业务作为桌面GUI依赖或要求启动PG登录。
4. 科学计算期间有独立心跳/取消通路，不能只在文件写入时续租；心跳失败后取消自身计算/原生子进程，旧代结果禁止发布。进程异常只影响所属任务。
5. 全批取消保留先前完整结果；单文件两产物失败一致回收；磁盘/权限/第二输出预算不足返回可解释错误，记录来源与实际后端，不泄露路径/标签至公共日志。
6. 执行 `& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/run_m01_validation.py`（拟建默认只复用已批准schema；任何清理限独立M01测试根）；同时定向回归P06/P07协议、owner与quota。

**退出条件：** 网页刷新/关页/服务重启可恢复批次，桌面离线可运行；确切文件字节、持久状态和进程退出均有证据。

### M01-G · 联合审阅与收口

**文件：** `tests/e2e/m01-parameter-estimation.cjs`、`scripts/verify_m01_desktop.py`、`docs/testing/m01-report.md`、`docs/manual/`对应操作章、来源清单/生成UI来源、验收表和任务账本。

1. 独立Chrome使用自有配置目录，Qt使用自有进程；不调用Codex内置浏览器关闭。验证实际上传WAV→计算→下载XLSX/SQLite→独立回读，桌面离线选择目录→计算→实际输出。
2. 全部6功能组正常/异常路径，浅/深主题、390px及桌面宽屏、中文/IPA、键盘、播放停止、标签切换草稿、取消/部分失败/过期/满额。截图只证明对应UI状态；试听计时与实际文件输出另验。
3. 跑B的parity、契约/owner/回收定向测试与前端typecheck/test/build；新的真实集成失败定位后只重跑相关检查。已通过不无意义重复全套。
4. 校验每条验收行都有关联报告；80键、14设置、SOE兼容、唇形4键及新增修正分别有证据。缺真实自然语料/原生路径/平台证据时写限制，不提前标全模块verified。
5. v2 427文件/HEAD/index/status/环境和输入语料hash保存性复核；仅staging审阅过的M01路径并本地提交，工作区干净。P08仍须其余9普通模块各自通过才能汇总verified。

## 4. 本轮文档验证与交付

以下是**当前已存在**、本轮可执行的命令，与上面的未来测试区分：

```powershell
& '.venv/v3-dev/Scripts/python.exe' -X utf8 -m pytest -c tests/pytest.ini tests/parity/test_baseline.py -q
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/validate_docs.py
git diff --check
```

另用只读AST核对80键/14控件、30文件hash和验收ID覆盖；`desktop.experiments.capture_context.capture()`比较v2保存性，只向 `output/validation/m01/`写本轮证据，不调用会覆盖P01输出的旧CLI入口。

实际结果记在 [本轮审阅报告](../testing/m01-planning-report.md)。上述未来脚本不纳入本轮已运行清单。M01-A完成后从M01-B开始；M02及M03等其他模块保持planned。
