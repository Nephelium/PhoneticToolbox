# 数据协议与版本 · API 1.1 / M01-D

M03-C增加`m03/1`：`EggRequest`仅接项目/资源ID、SHA-256与`EggTaskConfig`，通过`jobs/egg/create`创建通用`egg_analysis`任务。`EggManifest`仅允许完整CSV/三PNG/双WAV组合和来源JSON。`sample-aligned/1`明确原生时间outer join、实际Praat帧时间、批次GCI插值/CSV静音遮罩、采样对齐半开ROI和双FLOAT64 WAV语义。重试保持原配置，字体快照依赖P04-FONT尚未接入，完整接口/预算/差异见[M03-C记录](../docs/testing/m03-jobs-report.md)。未改既有002/005表结构。

应用发行版本的唯一维护入口为 [release/version.json](../release/version.json)，当前 Python 包 3.0.0a1，前端 3.0.0-alpha.1，API 1.1.0。运行时分别使用已安装包元数据和生成的前端版本文件。根目录旧 pyproject.toml 的 2.2.0 是历史迁移来源，不参与新包构建。

## 唯一生成链

1. 手写源：[后端模型](../backend/src/ptb_api/models.py)、[M01模型](../backend/src/ptb_api/acoustic_models.py)、[任务模型](../backend/src/ptb_api/job_models.py) 及 API 路由。
2. `python scripts/generate_contracts.py` 生成 [OpenAPI](openapi.json) 和 schemas 中的 Audio、Selection、Track、Viewport、AcousticRequest、AcousticResult、AcousticFileManifest、AcousticBatchSummary、ResultManifestEnvelope JSON Schema。
3. `npm --prefix frontend run contracts` 从固定 OpenAPI 快照生成 [TypeScript](generated/api.ts)。生成文件禁止手改；同名模型不另写一份前端 interface。
4. 两条命令分别加 `--check` 或改用 `contracts:check` 检查漂移；版本使用 `python scripts/sync_versions.py --check`。

## 科学语义

- sample_count 是每声道的帧数；采样帧为严格整数，最大 2^53−1，防止 JavaScript JSON 往返丢失整数精度。采样率明确为 Hz。
- Selection 使用原始采样率与半开区间 [start_sample,end_sample)，允许空选区和 EOF 单帧；Viewport 校验不能越界，不能改成播放整个文件。
- times_s 是非负、有限、严格递增的真实秒数组；与 values、validity、reason 等长。合法的 0 是真实数值，缺失不能填 0。
- validity 区分 valid / unvoiced / missing / failed。valid 必须有有限数值且 reason=null；其他状态值必须 null 且说明原因。NaN/Infinity 无法作为合法协议结果。
- unit 与 backend、analysis_config_hash、source_ids 必填。P02 不冻结 80 参数的单位映射或实际算法版本；P03/M01 按原源码和黄金样例核定，不能凭参数名猜单位。
- JSON Schema 与 TypeScript 提供结构约束；等长、单调、帧边界等跨字段约束由 Pydantic 执行，不能宣称浏览器类型检查已验证科研正确性。

P05 新增 auth challenge/login/me/logout 与 projects list/create/get/rename，模型源为 backend/src/ptb_api/account_models.py，科学模型不变。两个模式发布同一 OpenAPI；local 账号接口返回不可用，未配置服务器返回 503，不提供假账号。Viewport 仍是跨进程结构，任务与额度待 P06/P07。新增接口是 API 1 的扩展，未改变既有科学字段语义。

## 兼容规则

新增可选字段可以兼容扩展；单位、键名或状态语义改变必须升级 API major。应用版本与算法版本分开，科学依赖迁移要有 P03 证据。生成快照、包锁和来源登记与变更同批提交。

## M01-D 新协议及旧行为差异

2026-09-09至10实施，API 1.1新增共享模型，科学schema单独标记`m01/1`，算法版本`legacy-numeric/1`、适配版本`m01-adapter/1`。当前HTTP路径及旧任务模型保持不变，尚未开放acoustic_analysis操作。新结构不改P02 Track的validity语义；旧值中的NaN无法判定生理或失败原因，因此M01使用独立mask结构。详见[验收报告](../docs/testing/m01-contract-report.md)。

| 差异 | 新边界 | 原行为保留范围 |
| --- | --- | --- |
| D01 请求 | 只收项目/资源ID和hash；拒绝owner、路径、native_tool、未知字段 | 本地目录权限待E，不把旧裸路径透传服务端 |
| D02 设置 | 14项默认值/控件范围相同；max_formant严格正值、min_f0小于max_f0，拒绝非有限和不合类型 | 原有失败配置仍保存在历史基线；核心兼容测试不改期望 |
| D03 参数 | catalog仅80键，拒绝空选/重复/Energy别名/显示名；legacy_service显式空keys表示服务无筛选 | SOE_pF0/SOE_rF0作为legacy_service_extension保留，不增加80键 |
| D04 缺失 | 有限数包括0原值保留；NaN/+Inf/-Inf分别null加1/2/3，正常mask为0 | 原因只有legacy_nonfinite_unknown，不从NaN猜unvoiced/failed |
| D05 记录 | 真实解码帧数/声道/dtype/采样率、14设置及选择/策略hash、版本/来源/实际后端 | 固定16000元数据已在B单列修正；native必须有实际二进制hash，异常文本转安全code |
| D06 关联 | 同账号同项目ready资源，hash/类型/期限重查；同名多候选明确歧义 | 返回input_expired只是边界错误，运行任务自动取消需F接入租约/提交检查 |
| D07 完成 | 单文件恰有完整XLSX+SQLite；批次保存全列表子任务和计数，complete只表示全成功 | 批次关闭可含失败/取消/未开始，不能当完整单文件manifest |
| D08 保留 | 分析结果从成功起至多7天；切分不晚于输入截止；本地无服务器TTL | 当前为策略和可信快照校验，PG原子发布/到期物理回收仍需F集成 |

times_s沿用实际核心帧网格，验证为i×frameshift_ms/1000且处于真实音频范围内，不能由显示窗重算。每列携带目录显示名和A审计单位；Intensity不是测量SPL，唇形无输入单位信息时明确unknown。JSON Schema/TypeScript只保证结构；跨字段和科学一致性由后端验证和独立黄金结果对照负责。

## M01-F2 兼容增量

API仍为1.1.0；既有请求和结果均保留。增加`/jobs/batches/create`、`/jobs/batches/list`、`/jobs/batches/{id}`及cancel，复用原子任务get/retry；未配置持久批次返回503。新请求`BatchRequest.schema_version=m01-batch/1`只收资源ID/hash与配置，owner仍由会话/本机会话推导。1000项批次JSON边界为1 MB，其他账号/任务小请求上限不变。

`AcousticTaskManifest.kind=managed_acoustic_files`表示真实已公开资产；完整参数结果恰有XLSX、SQLite及用于同源参数派生的无损JSON。原`AcousticFileManifest`是D阶段科学清单，保持可读；新worker清单不冒充它。`BatchView`返回有序音频名及真实子任务计数；科学schema/单位/NaN规则未变。

`/jobs/local-inputs`和`/jobs/local-results/{id}`仅为桌面宿主提供有界二进制能力：本次bearer和Origin先于读取验证，普通网页登录不可调用。`/jobs/parents/latest`仅返回同账号项目、相同原音频hash、已完整发布且仍可访问的父结果引用。来源与授权边界见[ADR-027](../docs/decisions/ADR.md)。

## M03-D 交互显示增量

m03/1增加preview模式与micro_center/micro_width_ms，只作用于交互绘图，旧单文件/批次/逆滤波的完整文件集合不变。旧模式幂等哈希排除新增微观默认字段。egg-preview/1 JSON中缺失值为null，频谱栅格只做有界显示。egg-inverse-view/1记录原IF图的频率/dB与相对中心采样时间，WAV科学数值不改。两套显示结构由Pydantic生成OpenAPI/JSON Schema/TypeScript。

M03-E1：`egg.ptb.json`增加`input_name`和`export_names`保存建议，固定内部名称/manifest结构不变，旧结果无映射时回退旧名。单文件`keep_*_f0`作为绘图开关，CSV两类F0独立保留，修正D阶段混用的行为；批次列选择不变。
