# EGG 信号分析 · M03

本目录提供同步音频/EGG 的四图分析、参数导出与简化逆滤波界面。名称与持久模块 ID 见[模块注册表](../../app/registry.ts)，详细操作见[可编辑说明书章节](../../../../manual/chapters/m03.json)。章节按加载、时间定位、事件、语谱/F0、逆滤波、单文件导出、批次及恢复组织，操作图和真实导出图放在对应步骤旁边，依据当前源码及已记录验证范围核查。

## 功能概览

- CQ/SQ、语谱及 F0、音频微观、EGG 微观四图联动，支持总览定位与自动更新。
- GCI/GOI、滤波、微观时间窗和自动 dB 显示可调整。
- Praat、GCI 与 REAPER F0 可分别显示。新单文件预览三个开关均关闭；批次默认开启 Praat/GCI、关闭 REAPER，使用独立配置。
- 保存 CSV/三图，或查看与保存逆滤波的图形和音频。
- 页内帮助打开本模块的使用说明章节，可返回原工作台；方法与引用另提供来源窗口。

## 输入与输出

| 类型 | 内容 |
| --- | --- |
| 输入 | 双声道 WAV，默认左 EGG、右音频，可交换 |
| 参数输出 | CSV、PNG、任务来源 JSON |
| 逆滤波输出 | 归一化分析音频与 IF 估计 WAV、图形和来源记录 |
| 草稿 | 单文件、批次配置及 LP 阶数，不包含源音频 |

## 快速流程

1. 打开目录并选择双声道 WAV，核对接线角色；加载后自动分析。
2. 在总览选择宏观范围，在左图定位微观中心，调整滤波和事件参数。
3. 按需要开启 Praat/GCI/REAPER F0。新 Praat AC 与 REAPER 使用 30–800 Hz、10 ms。
4. 用底部播放栏试听选区，等待最新预览完成后保存 CSV/三图。
5. 逆滤波选择稳定短元音；批量分析在独立弹窗设置完整文件参数。

## 格式与限制

- Windows 本地输入限 2,000,000,000 字节、1800 秒、8–96 kHz 双声道，条件同时满足。逆滤波选区另限 10 秒、960000 帧。
- 旧 120 秒/576 万帧/64 MB 范围保留短文件路径。长文件采用 `egg-bounded/2`：全局幅度和趋势、20 秒块、SOS 零相位滤波及重叠裁剪，滤波上下文根据极点衰减确定。缓存最多 3 块，局部 CQ/事件/微观滤波也使用 SOS 并保留既有取窗，不宣称与整段算法逐位一致。
- 长文件实时宏观视野最多 60 秒，F0 在当前视野按块估计，GCI 异常值参考当前视野；完整导出使用全文件事件网格。总览为 1 kHz 显示数据，试听与分析选区保持原采样率。完整 CSV 保留原时间网格，三图波形为 4096 个 min/max 箱，语谱最多 2048 时间列按最大 PSD 聚合，压缩只作用于显示。
- 新本地任务清单 `m03/2` 允许最多 256 MB 结果集，超过 64 MB 的 CSV 从选择目录保存完整结果流式导出；其他显示与音频文件仍限 64 MB。服务器长文件路径尚未扩容验收。
- 各声道独立归一化至峰值 0.7。试听归一化分析音频，原 WAV 保留。
- CQ 严格保留 `0.05 < CQ < 0.95`。SQ 为 `(去接触时长−接触建立时长)/接触时长`，CQ/SQ 缺失值独立处理。
- 微观窗为 5–5000 ms，显示抽点不改变滤波、事件或 CSV/WAV。
- 简化 IF 使用 GCI 后固定 3 ms 自相关 LPC 片段，未根据 GOI 确认闭相，也未按下一 GCI 截断。10 秒选区仍使用同一组平均 LPC 系数，声道状态变化时应缩短选区。高 F0 时可跨周期，不能将结果当作直接测得的声门流。LP 自动阶数为 `floor(fs/1000)+6`，合法阶数不保证闭相取窗可靠。
- 新 F0 来源记为 `audio-f0/2`。旧任务保留原范围，REAPER 缺失时明确报错。
- 自动 dB 随可见区间 PSD 更新 50 dB 灰度，仅改显示，不改 PSD、F0、事件、CQ/SQ。手动模式导航保留上下限，无声区间不伪造范围。
- 单文件 CSV 保留各来源原时间网格并外连接，Praat/GCI 列始终保留。批次 CSV 插值到 GCI 网格，20 ms 平均绝对振幅仅遮罩 CSV。两类导出行时间不必一致。
- 单文件科学三图为 150 dpi，批次可选三图为 100 dpi；IF 的 6/4/4/2 图组合及单图 PNG 为 300 dpi。保存均从正式任务结果进行。
- 静态语谱/F0 PNG 当前用黑色 Praat、红色 GCI 和青绿色 REAPER 点，工作台的 Praat 用紫色。比较时核对各自坐标与任务参数快照。

## 源码结构

| 文件 | 职责 |
| --- | --- |
| [EggAnalysisPage.vue](EggAnalysisPage.vue) | 实时会话、任务与结果窗口 |
| [EggSignalPlots.vue](EggSignalPlots.vue)、[navigation.ts](navigation.ts) | 四图显示与时间导航 |
| [EggControls.vue](EggControls.vue)、[EggParameters.vue](EggParameters.vue)、[state.ts](state.ts) | 配置、默认值与校验 |
| [EggBatchPanel.vue](EggBatchPanel.vue)、[EggInverseResult.vue](EggInverseResult.vue) | 批次及 IF 呈现 |
| [科学核心](../../../../packages/phonetic_core/src/phonetic_core/egg/) | 预处理、事件、参数及 IF |

## 开发与定向验证

从仓库根目录运行[源码启动器](../../../../scripts/Start-M03-Workbench.ps1)，宿主使用统一主环境，科学子进程使用已有 M03 兼容环境。公共前端检查使用 `npm --prefix frontend run typecheck`、`npm --prefix frontend test` 和 `npm --prefix frontend run build`。

定向入口：[F0 Qt 检查](../../../../scripts/verify_m03_r5_qt.py)、[自动 dB 交互检查](../../../../tests/e2e/m03-r6.cjs)。[F0 报告](../../../../docs/testing/2026-10-04-m03-r5-report.md)、[公共显示报告](../../../../docs/testing/2026-10-04-p19-r5-m03-r6-report.md)与[页内帮助报告](../../../../docs/testing/2026-10-06-p19-r17-report.md)记录限定 Windows 源码/Qt/Chrome 验证；WSL 纯核心不等于完整 Linux 原生 REAPER 或 GUI 验收。实体设备、物理 DPI 与近期源码的成品功能未全面重验。

## 方法与来源

参见[方法核查](../../../../docs/references/m03-method-audit.md)、[参数核查](../../../../docs/references/m03-r3-algorithm-audit.md)、[来源映射](../../../../docs/modules/evidence/M03-source-map.md)与[统一来源登记](../../../../third_party/source-registry.json)。参考文献中的 DECOM 等方法不应自动视为本程序实现。

长录音及逆滤波边界见 [M03-R7 验证报告](../../../../docs/testing/2026-10-07-m03-r7-report.md)，成品功能需查对应包的验证范围。
