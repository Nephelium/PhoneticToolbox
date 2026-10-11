# README 配图来源

根 [README](../../../README.md) 当前展示18个模块、首页和设置，共20张实际Windows工作台截图，统一采用 **Absolutely浅色配色**。图片原尺寸为2561 × 1601，页面缩放100%，屏幕DPR为1.5。当前配图采于2026-10-12的源码工作台。

## 当前展示图片

| 页面 | 图片尺寸 |
| --- | --- |
| [工作台首页](absolutely-home.png) | 2561 × 1601 |
| [参数估计](absolutely-acoustic-analysis.png) | 2561 × 1601 |
| [参数显示](absolutely-parameter-display.png) | 2561 × 1601 |
| [LPC 谱图](absolutely-lpc-analysis.png) | 2561 × 1601 |
| [EGG 信号分析](absolutely-egg-analysis.png) | 2561 × 1601 |
| [TextGrid 标注](absolutely-textgrid-annotation.png) | 2561 × 1601 |
| [声学参数合成](absolutely-speech-synthesis.png) | 2561 × 1601 |
| [发声类型合成](absolutely-phonation-synthesis.png) | 2561 × 1601 |
| [变速变调](absolutely-pitch-manipulation.png) | 2561 × 1601 |
| [语谱图转音频](absolutely-spectrogram-to-audio.png) | 2561 × 1601 |
| [生理参数合成](absolutely-vocal-tract.png) | 2561 × 1601 |
| [唇形提取](absolutely-lip-extraction.png) | 2561 × 1601 |
| [MFA 自动标注](absolutely-mfa-alignment.png) | 2561 × 1601 |
| [汉字转国际音标](absolutely-mandarin-ipa.png) | 2561 × 1601 |
| [国际音标表Plus](absolutely-ipa-chart.png) | 2561 × 1601 |
| [音系归纳](absolutely-phonology-induction.png) | 2561 × 1601 |
| [感知实验](absolutely-perception-experiment.png) | 2561 × 1601 |
| [录音](absolutely-recording.png) | 2561 × 1601 |
| [语音学论文精读](absolutely-paper-reading.png) | 2561 × 1601 |
| [工作台设置](absolutely-settings.png) | 2561 × 1601 |

每张图均由任务专属的实际Qt工作台采集，公开副本与采集原图逐字节一致。界面通过自己的设置选择配色，图片没有裁剪、缩放、拼贴或重绘。图片摘要、输入摘要及采集范围见 [provenance.json](provenance.json)。

## 示例输入与公开范围

- 参数估计、参数显示、LPC、声学参数合成、变速变调、频谱编辑和感知实验使用正式说明书中的自然语句音频。公开文件名统一为语句示例。
- TextGrid使用本次编写的单层演示标注；该标注不表示人工精细语音切分。参数显示中的三条曲线来自同一次实际参数估计。
- EGG使用正式说明书原始双声道媒体，图中为40–41秒的有声段，50 ms微观窗口。
- 发声类型合成使用说明书中的两段自然单音，显示实际提取的F0对比。
- 音系归纳使用仓库公开的最小调查字表，实际生成三份结果；录音展示任务清单，感知实验展示刺激导入与设计界面。
- 唇形提取展示采集准备界面，没有启动摄像头或麦克风。论文阅读使用项目已有的公开协同发音论文示例，展示实际全屏阅读界面，文件夹路径处于视图外。

用户在本轮明确要求使用说明书素材制作GitHub配图。公开候选只包含静态截图及必要来源说明，原始录音、TextGrid、数据库、处理输出与任务配置留在本机。输入在临时副本中采用通用名称，原媒体字节未改变。来源登记为 `DOC-README-NATURAL-AUDIO`，见 [来源登记](../../../third_party/source-registry.json)。

## 保留的既有证据

2026-10-11的五张 `release-300-*.png` 来自已公开的Windows 3.0.0 R7最终EXE，应用SHA-256为 `162c8fa542b7b3b1203ce535e3021235e261f0dfaeb095659f9946baec055edb`。此前六张源码工作台图也保留。它们当前不在根README展示，其原输入授权、尺寸、摘要和EXE身份保存在provenance的历史节点中。

[website-button.svg](website-button.svg) 为300 × 58的静态黄蓝官网按钮，不依赖脚本、外部字体或动态图片服务。

当前图片记录源码工作台的界面和示例计算，已公开R7成品的字节身份单独记录于 [发行检查](../../testing/windows-current-release.md)。本轮文档和配图没有重建EXE。检查结果见 [README检查报告](../../testing/github-readme.md)。
