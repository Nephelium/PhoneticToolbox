# 模块入口与迁移基线

当前使用和代码入口查模块 README，状态按模块 ID 查询[任务台账](../plans/task-ledger.json)与相应报告，不在本页复制各轮 verified/下一步。

| ID | 模块 | 操作与实现入口 |
| --- | --- | --- |
| M01 | 参数估计 | [README](../../frontend/src/modules/parameter-estimation/README.md) |
| M02 | 参数显示 | [README](../../frontend/src/modules/parameter-display/README.md) |
| M03 | EGG 信号分析 | [README](../../frontend/src/modules/egg-analysis/README.md) |
| M04 | LPC 谱图 | [README](../../frontend/src/modules/lpc-spectrum/README.md) |
| M05 | 唇形提取 | [README](../../frontend/src/modules/lip-extraction/README.md) |
| M06 | 声学参数合成 | [README](../../frontend/src/modules/speech-synthesis/README.md) |
| M07 | 发声类型合成 | [README](../../frontend/src/modules/phonation-synthesis/README.md) |
| M08 | 变速变调 | [README](../../frontend/src/modules/pitch-manipulation/README.md) |
| M09 | 语谱图转音频 | [README](../../frontend/src/modules/spectrogram-to-audio/README.md) |
| M10 | 生理参数合成 | [README](../../frontend/src/modules/vocal-tract/README.md) |
| M11 | MFA 自动标注 | [README](../../frontend/src/modules/mfa/README.md) |
| M12 | TextGrid 标注 | [README](../../frontend/src/modules/annotation/README.md) |
| M13 | 汉字转国际音标 | [README](../../frontend/src/modules/mandarin-ipa/README.md) |
| M14 | 音系归纳 | [README](../../frontend/src/modules/phonology-induction/README.md) |
| M15 | 感知实验 | [README](../../frontend/src/modules/perception/README.md) |
| M16 | 录音 | [README](../../frontend/src/modules/recording/README.md) |
| M17 | 国际音标表 Plus | [README](../../frontend/src/modules/ipa-plus/README.md) |
| M18 | 语音学论文精读 | [README](../../frontend/src/modules/paper-reading/README.md) |

## 迁移覆盖

[原功能矩阵](legacy-acceptance.csv)、[双端补充](dual-platform-acceptance.csv)、[V2 功能细节基线](v2-feature-baseline.md)及[说明书覆盖](v2-manual-coverage.md)用于逐项核对，历史暂停和验收状态不继承为当前指令。

原 197 项由 83 功能组、20 全局项、80 参数项和 14 设置项构成，并非全部实际按钮或功能数量；新增模块按实际能力补充自己的验收。布局可调整，科学语义和既有功能不能遗漏。

迁移前核对对应 v2 说明书与源码，记录旧操作、新入口、行为/结果证据及明确差异。Windows、Linux 服务、设备、远程与最终成品分别标范围，公共界面通过不等于完整模块通过。
