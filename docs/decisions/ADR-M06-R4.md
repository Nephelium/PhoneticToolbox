# ADR-M06-R4：保留原录音信息的可选重合成

日期：2026-10-04。状态：accepted，已实施；验证限定 Windows 开发态与 WSL 纯核心。井井在阅读 [M06-R3 核查](../references/m06-r3-resynthesis-audit.md)后要求按报告修改。

## 决策

保留 `klatt/2` 及其 24 参数编辑，新增 WORLD 和 Praat overlap-add（PSOLA）两个可选路径。采用同一页面、持久任务、源文件资源和共享播放器。自然路径要求原 WAV，不把有限 Klatt 控制量逆推成原频谱。声门神经模型、逐共振峰精细变换和自然语料感知实验仍为 planned。

纯核心 `synthesis/resynthesis.py` 只处理数组和配置，REAPER 通过既有端口注入。文件读写、受管原生进程、Praat 随机种子和任务发布由 worker 适配层负责。新增 `resynthesize` 动作，音频资源必填，输入 SHA-256 必须与配置绑定值一致。输出四个文件整体发布或失败，沿用 1,000,000,000 字节进程预算、120 秒任务预算及既有取消/迟到归属协议。

## 方法与数值语义

- F0 可选 Praat CC、Praat AC、既有原生 REAPER、WORLD Harvest；默认 CC。10 ms 帧移，保存实际后端、源/目标时间与 F0，缺测用 0 表示。编辑曲线可填补空缺，WORLD 只在原有声帧应用目标 F0。
- WORLD 锁定 PyWORLD 0.3.5，使用 CheapTrick 功率谱包络和 D4C 非周期幅度比。默认原 F0、谱包络频率比例 1、非周期幅度比例 1。变时长按源/目标时间线性映射。频率比例是整体谱包络频率变换，非周期比例只乘有声帧噪声幅度比并裁入有效区间；均不等价于 AH、HNR 或逐个共振峰参数。
- PSOLA 通过 Praat `To Manipulation` 估计源脉冲，选定 F0 仅替换目标 PitchTier，DurationTier 为统一时长比例。源脉冲与清浊复制仍由 Praat 决定，不能声称换成了 REAPER 的脉冲。无有声点且请求编辑 F0 时明确拒绝。
- Praat overlap-add 的清音处理涉及自身随机数。每个任务在隔离子进程设置 Praat 种子，另记 `praat_seed`；NumPy 种子不能替代。相同输入/配置/种子的任务 WAV 已逐字节复验。WORLD 记录其内部随机生成策略，不声称采用任务 NumPy 种子。
- Parselmouth 0.4.7 所含 Praat 的 Manipulation 会减去全段均值。对此记录实际均值，并只恢复源中连续至少 40 ms 的严格数字零区间：保留两侧各 10 ms 过渡，内侧各用 5 ms 渐变。时间均按目标比例缩放。无能量阈值门控，不截掉低幅语音；原区间另存数组。这是 PTB 明确增加的处理，不冒充上游原样输出。
- 两条新路径保持源采样率并以通道算术均值转单声道。正常幅度不放大、不做峰值归一化，不应用 Klatt 淡入淡出；峰值超过 0.99 才整体衰减，记录系数。输出为 FLOAT32 WAV，记录上游原始样本数及目标长度的裁补。Klatt 仍为原 PCM16 路径。

## 兼容、预算和依赖

配置保持 `m06/2`，新增带默认值的 `render` 对象。旧文件在内存补 Klatt 默认，不重写原文件；旧 `m06/1` 标尺继续明确拒绝。新计算分别记 `m06-world/1`、`m06-psola/1`，自然路径提取记 `m06-natural-extract/1`。恢复草稿后重选相同哈希的原 WAV 只恢复关联，保留编辑曲线；其他文件需显式重新提取。

输入/输出各限 0.1–10 秒、480000 样本，目标时长比 0.5–2。WORLD 限 16–48 kHz；两条新路径的 F0 范围及非零目标 F0 限 40–1000 Hz。WORLD 在分配谱矩阵前限制 `max(源帧数,目标帧数) × 频率格数 <= 1000000`，可能比总时长上限更早拒绝高采样率/低 F0 的组合。源 WAV 沿用 8 MB 准入，worker 输入包 16 MB、结果包 24 MB。缺依赖、范围超限或算法失败均显式报错，无静默回退/重采样。

PyWORLD 是可选依赖，精确哈希见 [锁文件](../../requirements-m06-world-additions.lock)。本次只安装到现有项目 `.venv/m09-ui` 及既有 WSL M06 环境；WSL 另用 Cython 3.1.5 和已有 GCC 构建。没有全局安装、系统配置变更或修改其他模块环境。封装 MIT 与所附 WORLD BSD 许可分别保存，实际归档内未提供精确 WORLD commit，保持 unknown。来源登记 `SRC-PYWORLD`、`SRC-WORLD`，已有 Praat/REAPER 来源保留。

## 证据与限制

结果包含 `synthesis.wav`、`m06.ptb.json`、`parameters.csv`、`analysis.npz`。NPZ 仅保存数值数组，用 `allow_pickle=False` 回读；WORLD 保留变换前的完整谱包络/非周期矩阵及源/目标 F0，元数据给出控制量，可重建变换。独立上游数值对照、真实持久任务、取消、保存回读、Chrome/Qt 和 WSL 检查见 [验收报告](../testing/2026-10-04-m06-r4-report.md)。

本轮仅合成输入，没有自然录音听辨、真实非模态发声性能或硬件/物理 DPI 验证。方法接入成功不构成自然度改善结论，Linux 正式任务资格继续关闭。未重打 EXE、push、公开发布或执行现存库 DDL。
