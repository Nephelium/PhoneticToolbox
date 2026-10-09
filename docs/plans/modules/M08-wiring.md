# M08 公共接线需求

2026-10-04 M08-R1 更新：history 收录所有已生成且可读取的同源结果，新增可选 saveMany 导出现有 WAV。原 save 复制协议保留兼容，页面移除保存并编号。当前流程见 [ADR-M08-R1](../../decisions/ADR-M08-R1.md)和[验收报告](../../testing/2026-10-04-m08-r1-report.md)，以下旧保存 UI 描述被本轮取代，权限 / 到期 / 科研边界继续适用。

> 2026-09-27：井井授权本任务实施公共接线。现已接通正式 Windows 开发宿主，详细实现、测试、共享文件释放及 P07 saved_copy 到期交接见 [接线报告](../../testing/m08-wiring-report.md)。下文原接口约束继续适用，“尚无/待 owner 接入”等是此前状态，已由新报告取代。Linux 和真实 PG 验收仍有明确阻断。

2026-09-26。模块实现由本轮 M08 任务负责，公共文件修改由对应 owner 串行实施。此文不构成已接入或能力已开放的声明。

## P04 页面注册

- import `frontend/src/modules/pitch-manipulation/PitchManipulationPage.vue`。
- props：`context: ResearchContext`、`stateKey: string`、`active: boolean`；可选 `port: M08Port`，也可由 `context.files.m08` 提供模块适配器。
- emit：`references`，照既有模块打开 M08 来源。
- `defineExpose({save})` 返回真实 boolean。写入失败返回 false、保留 dirty。共享 `workspace(stateKey).dirty` 是关闭保护依据，草稿存原输入 hash、F0、参数、参考线与显示范围。
- 页面已使用 ModuleFrame/ModuleToolbar/ModuleSection/ModuleStatus、WaveformViewport、AudioTransport、TaskPanel、ModalDialog。v-show 保留编辑；页面无独立大标题/关闭按钮。
- 无 adapter 时只允许实际 WAV 预览和试听，明确显示未接线。不把可导航等同科学任务可用。

## P11 / 公共任务入口

精确模块契约：`frontend/src/modules/pitch-manipulation/port.ts`；Python 配置：`backend/src/ptb_api/m08_models.py`；计算 handler：`backend/src/ptb_worker/m08_jobs.py:execute`。

`execute(sound, config, stem, cancelled=..., emit=...)` 只能在受限科学子进程中调用。原生调用之间协作取消，原生调用中的硬取消/进程组限制由公共 executor 保证。不得将它放进 API 事件循环。产物逐一 emit，禁止聚合整个批次音频。

建议 operation `pitch_manipulation`、schema `m08/1`、source_ids `SRC-PRAAT`。config.action 包括 preview/synthesize/transform/linear。所有时间秒、F0 Hz。preview 返回 Praat 真时间数组和原 F0。synthesize 由真实原文件重提取时间网格，只接完整等长 modified_f0。transform 必须整段；linear 明确当前视野及内部控制区间。

输入通过现有 owner/project/hash/expiry 校验，公开入口只收 AcousticAssetRef，不收路径。先预留全部临时和最终空间，再由受限 writer 保存 WAV/快照。结果 complete 仅在 fenced manifest 与配额提交成功后标记。失败/取消时完整保留逐文件状态，未发布残留由现有 scratch 生命周期处理。

前端 M08Port 需要 preview、submit、jobs、cancel、audio、save、history、remove、rename、download。已有 ResearchTasks 尚无这些 M08 方法，不能强转 JobView 冒充已接。preview 的 wav 必须为原音解码后的 PCM 波形，支持 V2 WAV/MP3/FLAC；当前公共 FileProvider 的 fileKind 只识别 WAV，需由 owner 补齐授权音频适配，不能仅改 accept 属性。

save 为本次合成结果按 V2 前缀扫描最大尾号 +1，目录内原子分配，不覆盖。保存名绑定合成时 start/end 快照，不能用保存时的新视野命名。history 从已保存 PCM16 WAV 重提取 F0，横轴从保存音频零起点开始。管理仅收明确结果 ID，在宿主复核 owner、源资产与本次范围，download/delete 在满额时仍可用。rename 先核对全部名称与大小写冲突，保留明确逐项失败，不将部分完成报作整批成功。

## 预算与平台

模块边界先拒绝超过 64 MB 输入、8,000,000 帧、8 声道、32,000,000 预计输出帧、64 控制点、256 组合。它们是准入保护，不截断输入，不是已经实测的小服务器峰值。极端倍率/原生分配仍需进程硬限制。

Windows 模块组件与真实计算测试由 M08 自有测试入口完成。Linux 用 WSL 独立目录验证短公开合成数据；不自行在小服务器并发压测。正式 capability 继续由平台控制，不能因核心测试通过自动开放。

## 行为修正分离

- M08-FIX01：旧 `Shift frequencies(..., "Hz")` 在锁定 Parselmouth 0.4.7/Praat 6.1.38 明确失败。原样核心仍可复现失败，handler 显式使用 `Hertz`。独立测试覆盖乘法后加法，旧倍率阈值不变。
- M08-FIX02：坏倍率/NaN/越界/组合爆炸在边界明确拒绝，不沿用静默回落或吞异常。
- M08-FIX03：保存绑定生成快照、owner ID 与冲突预检，防止旧版移动视野后错命名或路径越界。

这些修正与三份直接迁移的科学文件分开评审。全局 ADR、台账及共享依赖清单留给统筹处理。
