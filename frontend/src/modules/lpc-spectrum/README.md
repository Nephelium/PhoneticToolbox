# LPC 谱图 · M04

本目录提供 WAV 短片段的 LPC 谱包络分析、波形/TextGrid/Praat 定位、试听与结果保存界面。当前能力按 Windows 实际源码及已有验证范围说明。模块 ID 与入口见[模块注册表](../../app/registry.ts)，完整操作章节见[可编辑说明书](../../../../manual/chapters/m04.json)，既有模块说明见[操作文档](../../../../docs/manual/lpc-spectrum.md)。

## 用户流程与界面

1. 打开音频目录，或在具备导入能力的入口添加 WAV 和可选 TextGrid，再从音频下拉框逐个加载。
2. 核对唯一同名 TextGrid 的自动关联、第一层或手动选层。在波形/语谱图拖选，或在底部填写起止秒数。
3. 试听当前原始声道，设置阶数与频率上限，确认样本预算后开始分析。
4. 用波形/LPC 频谱切换、频谱缩放和平移查看结果。动态/固定纵轴及固定 dB 范围即时应用已有谱值。
5. 保存当前纵轴 PNG，或另选目录保存原任务 PNG/WAV/JSON。历史记录可查看、取消和按原配置重试。

左栏为文件、标注、声道及谱图参数，中区显示波形或 LPC 频谱，右栏为任务、保存与历史。底部共用 `AudioTransport`，放时间选区、全部、播放/暂停、停止、进度和音量。输入目录与输出目录分别授权。顶部“帮助”打开本章，点击“返回 LPC 谱图”继续原文件、选区与结果。

## 科研语义与限制

- WAV 限 64,000,000 字节、800 万帧、8 声道和 8–96 kHz；公共预览另限合计 3200 万采样值。单次 LPC 最多 48,000 样本，至少阶数 + 2 个样本。
- 默认 50 阶、显示频率上限 8000 Hz、固定轴 −5 至 35 dB。有效草稿可恢复已保存值。频率上限不改变 1024 点谱数组；超出 Nyquist 的区域保持空白。
- 多声道按逐样本算术均值计算，PCM 先换算成浮点，无峰值归一化。试听声道、音量及显示两个声道均不改变分析策略。
- 选区使用 `int(t * fs)` 和半开样本区间。未明确选择时分析范围跟随可见窗；明确拖选或编辑后，缩放不替代底部起止范围。
- LPC 固定 0.97 预加重、全选区 Hamming 窗、自相关 Toeplitz 求解、单位分子频率响应。当前没有可调窗长/窗函数/预加重、原音频 FFT 曲线、FFT 长度或全目录批处理入口。
- 本页只显示 LPC 包络，未乘预测误差增益，dB 未经声压校准。包络峰不自动成为人工核验的真实共振峰；科研解释须记录选区、阶数、采样率与录音条件。
- Praat 预览固定 Gaussian 5 ms、50 dB 相对灰度、6 dB/oct 显示预加重，频率最高 `min(5000, fs/2)`。改可见时间窗、文件或声道会刷新；只改选区不重复计算未改变的预览窗。
- TextGrid 标签保留当前末端区间排除、去重与 `+` 拼接规则。任一层非空标签越过音频范围会拒绝分析；空白末端可裁切选区，不改原标注。

## 输出与恢复

| 输出 | 语义 |
| --- | --- |
| 当前 PNG | 从已有谱值采用点击时纵轴、任务完整频率范围及任务字体快照生成；2400×1350、300 dpi、白底黑线，与窗口缩放/主题无关 |
| 完整任务 PNG | 任务最初生成的轴与字体，后来显示改轴不重写；与独立 PNG 的绘制实现可有差异 |
| 选区 WAV | 实际半开区间的 FLOAT64 单声道分析样本，保持原采样率 |
| JSON | `m04/1`、`v2-lpc-autocorrelation/1`、1024 点谱值、参数、原文件和 TextGrid 哈希、实际边界及字体来源 |
| 参数草稿 | 阶数、频率上限、纵轴模式/范围；不替代结果和源文件 |

保存窗口打开或浏览器接收仅表示交接，须确认实际文件。完整目录保存保留同名不同内容的原文件。历史结果未加载原 WAV 时试听单声道片段，以相对 0 s 起计，读取完成后自动选定片段全长并启用播放，打开结果不会自动播放；点击“全部”可重新选满片段，不能据此提交新分析；原 WAV 已加载时播放器继续使用当前原音频，需核对历史图的文件名。重试采用原任务输入与配置，需要改参数时提交新的分析。

## 源码结构

| 文件 | 职责 |
| --- | --- |
| [LpcSpectrumPage.vue](LpcSpectrumPage.vue) | 文件/标注、时间选区、任务状态、结果保存与历史 |
| [state.ts](state.ts) | 默认值、预算、草稿与结果格式校验 |
| [SpectrumPlot.vue](SpectrumPlot.vue)、[display.ts](display.ts) | 频率视窗和即时纵轴，保留谱数组 |
| [export.ts](export.ts)、[export-scene.ts](export-scene.ts) | 当前纵轴、纸面尺寸和字体快照 PNG |
| [科学核心](../../../../packages/phonetic_core/src/phonetic_core/lpc/spectrum.py)、[兼容计算](../../../../packages/phonetic_core/src/phonetic_core/lpc/_legacy.py) | 单声道/选区规则、自相关 LPC 与频率响应 |
| [任务产物](../../../../backend/src/ptb_worker/lpc_child.py)、[任务绘图](../../../../backend/src/ptb_worker/lpc_exports.py) | 原任务三件套、标签、导出命名与字体 |

## 开发与已记录验证范围

从仓库根目录使用[源码启动器](../../../../scripts/Start-M04-Workbench.ps1)。前端定向入口为[状态测试](../../../tests/lpc-state.test.ts)、[纵轴与导出测试](../../../tests/lpc-display.test.ts)，实际 Qt 入口为[定向检查](../../../../scripts/verify_m04_r2_qt.py)。常用前端检查为：

```powershell
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
```

[最新显示报告](../../../../docs/testing/2026-10-04-m04-display-report.md)记录 Windows 开发态 Chrome、实际 Qt、纵轴、改窗刷新、历史片段和 PNG 回读的限定验证。[已有 Qt 保存报告](../../../../docs/testing/p17-m04-r2-report.md)另有指定真实短录音的目录保存证据。上述报告不代表实体声卡、实际 DPI、自然语料的全部科学解释、Linux/macOS GUI、生产网页或后续 EXE 都已验证。本章图文核对使用最大化 Windows Qt 工作台与授权短录音副本，实际检查 TextGrid 切层/选区、Praat 改窗、LPC 任务、固定/动态轴、当前 PNG 与完整三件套回读。最新隔离 Vite 构建接入实际 Qt 后，历史恢复检查确认重载页面、未选原 WAV、打开历史结果即自动得到完整片段范围，无需点击“全部”；静音环境下直接播放、暂停、再播放与停止的状态检查通过，实体音频播放未据此标为通过。

## 方法与来源

John Makhoul（1975）*Linear Prediction: A Tutorial Review*. Proceedings of the IEEE, 63(4), 561–580，[DOI](https://doi.org/10.1109/PROC.1975.9792)为方法参考。直接计算代码来自项目 V2，更早出处尚无可核准证据。NumPy/SciPy 为运行依赖，Praat/Parselmouth 在本页仅用于语谱预览。详见[源码映射](../../../../docs/modules/evidence/M04-source-map.md)、[方法核查](../../../../docs/references/m04-method-audit.md)与[统一来源登记](../../../../third_party/source-registry.json)。
