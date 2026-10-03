# ADR-M06-R3：复用 F0 后端与共享播放栏

2026-10-04，按当前任务授权实施。

配置 m06/2 新增可缺省字段 `f0_method`，枚举 praat_cc、praat_ac、reaper。旧 m06/2 读取补 praat_cc，不重写磁盘原文件。任务传输维持 m06/1，数据库不变。

Praat 使用公共 `compute_praat_f0_track`，REAPER 通过现有 `ReaperBackend` 注入纯核心，宿主在已预约临时文件与已有 Windows 进程预算内执行锁定二进制。不静默改用 Python REAPER 或 Praat。缺失/失败明确报错。

F0 按公共 `align_track_to_grid` 对齐实际秒数及 10 ms 网格，保留清音/无声空缺后再求有声掩码。旧 M06 丢弃 Praat 时间并按数组长度拉伸，可能插值跨过空缺。新提取单独标记 `m06-extract/2`，不宣称与旧提取逐位一致。用于编辑的连续 F0 仍填补空缺，AV 在未检出有声的帧为零；这类填补值不表示测得了 F0。同步修正 APQ5 百分数到 Shimmer 内部比例的换算（除以100），Jitter 仍为百分数。这是修复已有单位契约，旧参数文件和历史 WAV 不自动重算。其余测量到 Klatt 的经验映射保持不变，完整 M01 流程没有被搬入 M06。

完整共享 AudioTransport 放到三栏之外，切换源/结果时与 previewWave 一起切换，并保留选区、进度、音量和空格键逻辑。窗长仅使用横排 label，不改公共组件行为。

WORLD/PSOLA/声门声源模型等属于后续设计建议，需另行确定研究控制量与验收，不在当前实现中替换 Klatt。
